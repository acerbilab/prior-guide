# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# from https://github.com/toshas/torch_truncnorm
from __future__ import annotations

import copy
import logging
import math
import random
from numbers import Number

import corner
import matplotlib.ticker as ticker
import numpy as np
import scipy as sp
import scipy.stats as stats
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from scipy.linalg import block_diag
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class TruncatedStandardNormal(Distribution):
    """Truncated Standard Normal distribution.

    Source: https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True
    eps = 1e-6

    def __init__(self, a, b, validate_args=None, device=None):
        self.a, self.b = broadcast_all(a, b)
        self.a = self.a.to(device)
        self.b = self.b.to(device)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        eps = self.eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp(eps, 1 - eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def deterministic_sample(self):
        return self.mean

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    def _big_phi(self, x):
        phi = 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())
        return phi.clamp(self.eps, 1 - self.eps)

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        y = self._big_phi_a + value * self._Z
        y = y.clamp(self.eps, 1 - self.eps)
        return self._inv_big_phi(y)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=None):
        if sample_shape is None:
            sample_shape = torch.Size([])
        shape = self._extended_shape(sample_shape)
        p = torch.empty(shape, device=self.a.device).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return self.icdf(p)


class TruncatedNormal(TruncatedStandardNormal):
    """Truncated Normal distribution.

    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    def __init__(self, loc, scale, a, b, validate_args=None, device=None):
        scale = scale.clamp_min(self.eps)
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = a.to(device)
        b = b.to(device)
        self._non_std_a = a
        self._non_std_b = b
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        super().__init__(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return super().cdf(self._to_std_rv(value))

    def icdf(self, value):
        sample = self._from_std_rv(super().icdf(value))

        # clamp data but keep gradients
        sample_clip = torch.stack(
            [sample.detach(), self._non_std_a.detach().expand_as(sample)], 0
        ).max(0)[0]
        sample_clip = torch.stack(
            [sample_clip, self._non_std_b.detach().expand_as(sample)], 0
        ).min(0)[0]
        sample.data.copy_(sample_clip)
        return sample

    def log_prob(self, value):
        value = self._to_std_rv(value)
        return super().log_prob(value) - self._log_scale


def set_seed(seed: int):
    """This methods just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def robust_cholesky(A):
    eps = 1e-6
    L = None
    jitter = 0
    for i in range(3):
        try:
            L = sp.linalg.cholesky(A)
            logging.info(f"Successful Cholesky decomposion after jittering={jitter}")
            break
        except np.linalg.LinAlgError as e:
            pass
        jitter = 10**i * eps
        A = A + jitter * np.eye(A.shape[0])
    if L is None:
        raise ValueError("Cholesky decomposition failed after jittering.")
    return L, A


class MergedVP:

    def __init__(self, vp_list, ELCBO=False, sample=False):

        self.vp_list = [
            vp
            for vp in vp_list
            if (vp.stats["stable"] and np.max(vp.stats["J_sjk"]) < 5)
        ]
        self.vp_list = vp_list

        self.K = [vp.mu.shape[1] for vp in self.vp_list]
        self.mk = np.sum(self.K)
        self.M = len(self.vp_list)
        self.mu = np.concatenate([vp.mu for vp in self.vp_list], axis=1)
        self.sd = np.concatenate([vp.lambd * vp.sigma for vp in self.vp_list], axis=1)
        self.w = np.concatenate(
            [np.reshape(vp.w, (1, vp.mu.shape[1])) for vp in self.vp_list], axis=1
        )
        self.w = self.w / np.sum(self.w)
        if sample:
            self.I = []
            for vp in self.vp_list:
                # g_sk = np.zeros(vp.stats['I_sk'].shape)
                idx = random.sample(list(range(vp.stats["I_sk"].shape[0])), 1)[0]
                # print(idx)
                g_sk = np.zeros((1, vp.stats["I_sk"].shape[1]))
                # for s in range(g_sk.shape[0]):
                # mean = np.mean(vp.stats['I_sk'], axis=0)
                # cov_matrix = np.mean(vp.stats['J_sjk'], axis=0)
                mean = vp.stats["I_sk"][idx, :]
                cov_matrix = vp.stats["J_sjk"][
                    idx, :, :
                ]  # *np.eye(vp.stats['I_sk'].shape[1])*100000

                # sdts = np.sqrt(np.diag(cov_matrix))
                # s = np.random.normal(loc = 0, scale = 1, size = (1))
                # g_sk[0, :] = mean + sdts*s

                try:
                    _, cov_matrix = robust_cholesky(cov_matrix)
                    g_sk[0, :] = stats.multivariate_normal.rvs(
                        mean=mean, cov=cov_matrix
                    )
                except ValueError as e:
                    if "Cholesky decomposition failed after jittering." in str(e):
                        logging.error(str(e))
                    g_sk[0, :] = mean
                # try:
                #     g_sk[0, :] = stats.multivariate_normal.rvs(mean = mean, cov = cov_matrix)
                # except:
                #     g_sk[0, :] = stats.multivariate_normal.rvs(mean = mean, cov = cov_matrix + np.eye(vp.stats['J_sjk'].shape[1])*1e-6)

                self.I.append(g_sk)
            self.I = np.concatenate(self.I, axis=1)
        else:
            self.I = np.concatenate(
                [
                    np.mean(vp.stats["I_sk"], axis=0, keepdims=True)
                    for vp in self.vp_list
                ],
                axis=1,
            )
            # self.I = np.concatenate([vp.stats['I_sk'][0, :].reshape((1, vp.stats['I_sk'].shape[1])) for vp in self.vp_list], axis = 1)

        self.ELCBO = ELCBO

        self.J = [np.mean(vp.stats["J_sjk"], axis=0) for vp in self.vp_list]
        self.J = torch.tensor(block_diag(*self.J))

        self.individual_elbos = [vp.stats["elbo"] for vp in self.vp_list]

    def merged_entropy_efficient(
        self,
        mu: torch.Tensor,  # shape (D, K_total)
        sigma: torch.Tensor,  # shape (D, K_total)
        w: torch.Tensor,  # shape (K_total,)
        n_samples: int = 1024,
    ):
        """
        Vectorized MC estimate of the mixture entropy for K_total components,
        each with its own ParameterTransformer. Avoids O(K^2*N) nested loops in Python.
        Instead does O(K*N*K) but in a vectorized manner, i.e. only K big transform calls.

        self.vp_list: list of length M, each with parameter_transformer
        self.K: list of length M with K[m]
        Summation of K[m] = K_total.

        Returns: A scalar torch.Tensor for the entropy estimate.
        """

        device = mu.device
        dtype = mu.dtype

        # 1) Gather subcomps
        w = w.flatten()
        w = w / w.sum()  # ensure normalized
        subcomps = []
        idx = 0
        J_corrections = np.zeros((w.shape[0]))

        for vp_i, vp in enumerate(self.vp_list):
            K_m = self.K[vp_i]
            for _k in range(K_m):
                subcomps.append(
                    {
                        "transform": vp.parameter_transformer,  # x->z
                        "mu": mu[:, idx],
                        "sigma": sigma[:, idx],
                        "weight": w[idx],
                    }
                )
                idx += 1
        K_total = len(subcomps)  # same as sum(K)

        # 2) Build big array of base-space samples, Nx = K_total*N
        #    and track comp_index
        Nx = K_total * n_samples
        D = mu.shape[0]
        X_base = torch.empty(Nx, D, device=device, dtype=dtype)
        comp_index = torch.empty(Nx, dtype=torch.long, device=device)

        row_offset = 0
        for j, sc in enumerate(subcomps):
            mu_j = sc["mu"]
            sigma_j = sc["sigma"]
            transform_j = sc["transform"]

            # sample in z_j-space
            z_samp = torch.randn(n_samples, D, device=device, dtype=dtype)
            z_samp = z_samp * sigma_j.unsqueeze(0) + mu_j.unsqueeze(0)

            # invert to base space
            z_np = z_samp.cpu().numpy()
            J_corrections[j] = np.mean(transform_j.log_abs_det_jacobian(z_np))
            x_np = transform_j.inverse(z_np)  # shape (n_samples, D)
            x_b = torch.from_numpy(x_np).to(device, dtype=dtype)

            X_base[row_offset : row_offset + n_samples] = x_b
            comp_index[row_offset : row_offset + n_samples] = j
            row_offset += n_samples

        # 3) Evaluate log p(x) for all x in X_base
        #    => p(x) = sum_{j=1..K_total} w_j * p_j(x)
        # We'll build log_p_matrix in shape [Nx, K_total]
        log_p_matrix = torch.empty(Nx, K_total, device=device, dtype=dtype)

        # So we do K_total big transforms, each of shape Nx
        X_np_all = X_base.detach().cpu().numpy()

        for j, sc in enumerate(subcomps):
            trans_j = sc["transform"]
            mu_j = sc["mu"]
            sigma_j = sc["sigma"]

            # forward transform: x->z_j
            z_j_np = trans_j(X_np_all)  # shape [Nx, D]
            log_jac_np = trans_j.log_abs_det_jacobian(z_j_np)  # shape [Nx]
            z_j = torch.from_numpy(z_j_np).to(device=device, dtype=dtype)
            log_j = torch.from_numpy(log_jac_np).to(device=device, dtype=dtype)

            # diagonal normal logpdf
            dist_j = torch.distributions.Normal(loc=mu_j, scale=sigma_j)
            logpdf_z_j = dist_j.log_prob(z_j).sum(dim=1)
            # p_j(x) = p_j(z_j) * |det dT_j/dx|
            logpdf_x_j = logpdf_z_j - log_j

            log_p_matrix[:, j] = logpdf_x_j

        # add log(w_j)
        w_t = torch.tensor(
            [sc["weight"] for sc in subcomps], device=device, dtype=dtype
        )
        log_w = torch.log(w_t + 1e-40)
        log_p_matrix = log_p_matrix + log_w.unsqueeze(0)

        log_p_x = torch.logsumexp(log_p_matrix, dim=1)  # shape [Nx]

        # 4) group by comp_index => average log p(x)
        sum_log_p = torch.zeros(K_total, device=device, dtype=dtype)
        count_log_p = torch.zeros(K_total, device=device, dtype=dtype)

        for j in range(K_total):
            mask = comp_index == j
            count_j = mask.sum()
            if count_j > 0:
                sum_log_p[j] = log_p_x[mask].sum()
                count_log_p[j] = count_j

        # average => E_{x~comp_j}[log p(x)]
        avg_log_p = sum_log_p / (count_log_p + 1e-40)

        # 5) sum_j [ w_j * E_{x~comp_j}[ log p(x) ] ]
        total_sum = 0.0
        for j, sc in enumerate(subcomps):
            w_j = sc["weight"]
            total_sum = total_sum + w_j * avg_log_p[j]

        # final => negative
        H_estimate = -total_sum
        return H_estimate, J_corrections

    def merged_ELBO(self, I, mu, sigma, w, n_samples=10000):

        # H = self.merged_entropy(mu, sigma, w, n_samples)
        # H = self.mixture_entropy_mc(mu, sigma, w, n_samples)
        H, J = self.merged_entropy_efficient(mu, sigma, w, n_samples)
        J = torch.tensor(J)
        # print((I-J)*w)

        if self.ELCBO:
            beta = torch.sqrt(2 * torch.log(torch.tensor(self.M)))
            var = torch.matmul(
                torch.matmul(torch.reshape(w, (1, self.mk)), self.J),
                torch.reshape(w, (self.mk, 1)),
            )[0, 0]
            G = torch.sum((I - J) * w) - beta * torch.sqrt(var)
        else:
            G = torch.sum((I - J) * w)

        ELBO = G + H
        return ELBO, H

    def maximize_ELBO(self, n_samples, lr, max_steps, ablation):
        """
        Maximize merged_ELBO(I, mu, sigma, w) only w.r.t. w,
        by parameterizing w via softmax of unconstrained logits.

        Args:
            n_samples:   int (for the Monte-Carlo in merged_entropy)
            lr:          float, learning rate
            steps:       int, number of gradient steps

        Returns:
            w_final:   torch.Tensor of shape (K,)  (the optimized mixture weights)
            elbo
        """
        I = torch.tensor(self.I)
        sigma = torch.tensor(self.sd)
        mu = torch.tensor(self.mu)
        w_init = torch.tensor(self.w).flatten()
        logw = torch.log(w_init)
        broadcasted_elbos = np.concatenate(
            [np.ones((self.K[m])) * self.individual_elbos[m] for m in range(self.M)],
            axis=0,
        )
        broadcasted_elbos = torch.tensor(broadcasted_elbos)
        broadcasted_logK = np.concatenate(
            [np.ones((self.K[m])) * np.log(self.K[m]) for m in range(self.M)], axis=0
        )
        broadcasted_logK = torch.tensor(broadcasted_logK)

        # w_logits_init = torch.log(w[:-1]) + broadcasted_elbos[:-1] - (torch.log(w[-1]) + broadcasted_elbos[-1])
        # w_fixed = 0

        w_logits_init = torch.log(w_init) + broadcasted_elbos - broadcasted_logK
        w_logits_init = w_logits_init - torch.max(w_logits_init)

        if ablation:
            global_w_init = np.array([self.individual_elbos[m] for m in range(self.M)])
            global_w_init = torch.tensor(global_w_init)
            global_w_init = global_w_init - torch.max(global_w_init)
            global_w_logits = global_w_init.detach().clone()
            global_w_logits.requires_grad_(True)
            optimizer = optim.Adam([global_w_logits], lr=lr)
        else:
            # We treat w_logits as the raw, unconstrained parameter to be optimized:
            w_logits = w_logits_init.detach().clone()
            w_logits.requires_grad_(True)
            # Set up an optimizer that will *only* update w_logits
            optimizer = optim.Adam([w_logits], lr=lr)

        # for step in range(steps):
        convergence_counter = 0
        loss_old = 1e8
        w_best = copy.deepcopy(w_logits)
        elbo_best = None

        for step in range(max_steps):

            optimizer.zero_grad()

            # Convert logits -> valid weights via softmax

            # w = torch.cat((w_logits, torch.tensor([w_fixed])))
            # w = torch.softmax(w, dim=-1)
            if ablation:
                w = (
                    torch.repeat_interleave(
                        global_w_logits, repeats=torch.tensor(self.K), dim=0
                    )
                    + logw
                )
                w = torch.softmax(w, dim=-1)
            else:
                w = torch.softmax(w_logits, dim=-1)

            # Compute the (negative) objective we want to minimize
            # Since we want to maximize ELBO, we minimize -ELBO.
            elbo_value, h = self.merged_ELBO(I, mu, sigma, w, n_samples=n_samples)

            # if elbo_best == None:
            #     elbo_best = copy.deepcopy(elbo_value)
            if step == 0:
                print(f"Initial elbo = {elbo_value}")
            if (step + 1) % 5 == 0:
                print(f"iter {step+1}: elbo = {elbo_value}, entropy: {h}")

            loss = -elbo_value

            loss_new = torch.round(loss * 1e5) / 1e5

            if loss_new >= loss_old:
                convergence_counter += 1
            else:
                convergence_counter = 0
                w_best = copy.deepcopy(w_logits)
                elbo_best = -loss
                loss_old = loss_new

            if convergence_counter >= 5:
                # w_final = torch.cat((w_logits, torch.tensor([w_fixed])))
                # w_final = torch.softmax(w_final, dim=-1)
                if ablation:
                    global_w_logits = (
                        torch.repeat_interleave(
                            global_w_logits, repeats=torch.tensor(self.K), dim=0
                        )
                        + logw
                    )
                    w_final = torch.softmax(global_w_logits, dim=-1)
                    return w_final, elbo_value
                else:
                    w_final = torch.softmax(w_best, dim=-1)
                    return w_final, elbo_best

            # Backprop and take an optimization step
            loss.backward()
            optimizer.step()

        # Return the final mixture weights
        # w_final = torch.cat((w_logits, torch.tensor([w_fixed])))
        # w_final = torch.softmax(w_final, dim=-1)
        if ablation:
            global_w_logits = (
                torch.repeat_interleave(
                    global_w_logits, repeats=torch.tensor(self.K), dim=0
                )
                + logw
            )
            w_final = torch.softmax(global_w_logits, dim=-1)
        else:
            w_final = torch.softmax(w_best, dim=-1)

        # elbo_value, _ = self.merged_ELBO(I, mu, sigma, w_final, n_samples)

        return w_final, elbo_best

    def optimize_w(self, n_samples=10000, lr=1e-2, max_steps=200, ablation=False):

        w, elbo = self.maximize_ELBO(
            n_samples=n_samples, lr=lr, max_steps=max_steps, ablation=ablation
        )

        self.w = w.detach().numpy()

        self.elbo = elbo

    def sample(self, n_samples):

        idx = 0
        Xs = []
        for vp in self.vp_list:

            K = vp.mu.shape[1]

            sum_w = np.sum(self.w[idx : idx + K])

            vp.w = self.w[idx : idx + K] / sum_w

            samples, _ = vp.sample(int(np.round(n_samples * sum_w)))
            Xs.append(samples)
            idx += K

        return np.concatenate(Xs, axis=0)

    def corner_plot(self, n_samples):

        Xs = self.sample(n_samples)
        flat_samples = Xs
        labels = ["Ra", "gpas_soma", "ffact", "k", "v_init"]

        Ra_approx = np.mean(flat_samples[:, 0])
        gpas_soma_approx = np.mean(flat_samples[:, 1])
        ffact_approx = np.mean(flat_samples[:, 2])
        k_approx = np.mean(flat_samples[:, 3])
        v_init_approx = np.mean(flat_samples[:, 4])

        fig = corner.corner(
            Xs,
            labels=labels,
            truths=[Ra_approx, gpas_soma_approx, ffact_approx, k_approx, v_init_approx],
            color=None,
            show_titles=True,
            title_fmt=".5f",
            fill_contours=False,
            plot_datapoints=False,
            plot_density=False,
            # bins = 50,
            contour_kwargs={
                "levels": 8,
                "locator": ticker.MaxNLocator(prune="lower"),
                "colors": None,
                "cmap": "rainbow",
                "linewidths": 1.5,
            },  # Set the colormap and linewidths
        )

        axes = np.array(fig.axes).reshape(
            (5, 5)
        )  # Assuming a 2x2 grid, adjust accordingly
        for ax in axes.flatten():
            ax.grid(True)

        # plt.savefig(path_to_plots + 'pyvbmc_single_plot_160603_%i_new.png' % idx)
        plt.show()
        plt.close()
