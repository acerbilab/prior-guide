import random
from typing import Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from numpyro.distributions import Distribution, Independent, Normal, Uniform
from pyro.distributions import Distribution as PyroDistribution

from priorg.sim.tasks.bav import BAV_X_MEAN, BAV_X_STD, sample_bav_responses_flat
from priorg.sim.tasks.sbibm_task import GaussianLinear, TwoMoons
from priorg.sim.utils.conditional_mask import get_condition_mask_fn


def set_seed(seed: int):
    """This method just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_task(name: str, p_dist: Union[PyroDistribution, Distribution] = None):
    if name == "two_moons":
        return SBIBMTask(name="two_moons", p_dist=p_dist)
    elif name == "sir":
        return SBIBMTask(name="sir", p_dist=p_dist)
    elif name == "slcp":
        return SBIBMTask(name="slcp", p_dist=p_dist)
    elif name == "lotka_volterra":
        return SBIBMTask(name="lotka_volterra", p_dist=p_dist)
    elif name == "gaussian_linear":
        return SBIBMTask(name="gaussian_linear", p_dist=p_dist)
    elif name == "gaussian_linear_high":
        return SBIBMTask(name="gaussian_linear_high", p_dist=p_dist)
    elif name == "toy_gaussian":
        return (
            ToyGaussianTask(prior=p_dist) if p_dist is not None else ToyGaussianTask()
        )
    elif name == "oup":
        return OUPTask(prior=p_dist) if p_dist is not None else OUPTask()
    elif name == "turin":
        return TurinTask(prior=p_dist) if p_dist is not None else TurinTask()
    elif name == "bav":
        return BAVTask(prior=p_dist) if p_dist is not None else BAVTask()
    else:
        raise NotImplementedError(f"Task {name} not implemented")


def base_batch_sampler(key, batch_size, data, node_id, meta_data=None, num_devices=1):
    assert data.ndim == 3, "Data must be 3D, (num_samples, num_nodes, dim)"
    assert (
        node_id.ndim == 2 or node_id.ndim == 1
    ), "Node id must be 2D or 1D, (num_nodes, dim) or (num_nodes,)"

    index = jax.random.randint(
        key,
        shape=(
            num_devices,
            batch_size,
        ),
        minval=0,
        maxval=data.shape[0],
    )
    data_batch = data[index, ...]
    node_id_batch = jnp.repeat(node_id[None, ...], num_devices, axis=0).astype(
        jnp.int32
    )
    if meta_data is not None:
        if meta_data.ndim == 3:
            meta_data_batch = meta_data[index, ...]
        else:
            meta_data_batch = jnp.repeat(meta_data[None, ...], num_devices, axis=0)
    else:
        meta_data_batch = None
    return data_batch, node_id_batch, meta_data_batch


class SBIBMTask:
    observations = range(1, 11)

    def __init__(
        self, name: str, p_dist: PyroDistribution = None, backend: str = "jax"
    ) -> None:
        if p_dist is not None:
            assert isinstance(
                p_dist, PyroDistribution
            ), "SBIBMTask requires Pyro Distribution"
        self.name = name
        self.backend = backend
        if name == "two_moons":
            self.task = TwoMoons(p_dist=p_dist)
        elif name == "gaussian_linear":
            self.task = GaussianLinear(p_dist=p_dist, dim=10)
        elif name == "gaussian_linear_high":
            self.task = GaussianLinear(p_dist=p_dist, dim=20)
        else:
            raise NotImplementedError(f"Task {name} not implemented")

    def get_theta_dim(self):
        return self.task.dim_parameters

    def get_x_dim(self):
        return self.task.dim_data

    def get_prior(self):
        if self.backend == "torch":
            return self.task.get_prior_dist()
        else:
            raise NotImplementedError()

    def change_prior(self, prior: PyroDistribution):
        assert isinstance(prior, PyroDistribution), "Prior must be a Pyro Distribution"
        assert (
            prior.event_shape[0] == self.get_theta_dim()
        ), "Prior must have same dim as theta"
        self.task.prior_dist = prior

    def get_simulator(self):
        if self.backend == "torch":
            return self.task.get_simulator()
        else:
            raise NotImplementedError()

    def get_node_id(self):
        dim = self.get_theta_dim() + self.get_x_dim()
        if self.backend == "torch":
            return torch.arange(dim)
        else:
            return jnp.arange(dim)

    def _simulate(self, key, theta: jnp.ndarray):
        try:
            simulator = self.get_simulator()
            theta = torch.tensor(np.array(theta))
            theta = torch.atleast_2d(theta)
            xs = simulator(theta)
            return xs
        except:
            # If not implemented in JAX, use PyTorch
            old_backed = self.backend
            self.backend = "torch"
            simulator = self.get_simulator()
            theta = torch.tensor(np.array(theta))
            theta = torch.atleast_2d(theta)
            xs = simulator(theta)
            self.backend = old_backed
            if self.backend == "numpy":
                xs = xs.numpy()
            elif self.backend == "jax":
                xs = jnp.array(xs)
            return xs

    def get_data(self, num_samples: int, **kwargs):
        try:
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            return {"theta": thetas, "x": xs}
        except:
            # If not implemented in JAX, use PyTorch
            old_backed = self.backend
            self.backend = "torch"
            prior = self.get_prior()
            simulator = self.get_simulator()
            thetas = prior.sample((num_samples,))
            xs = simulator(thetas)
            self.backend = old_backed
            if self.backend == "numpy":
                thetas = thetas.numpy()
                xs = xs.numpy()
            elif self.backend == "jax":
                thetas = jnp.array(thetas)
                xs = jnp.array(xs)
            return {"theta": thetas, "x": xs}

    def get_observation(self, index: int):
        if self.backend == "torch":
            return self.task.get_observation(index)
        else:
            out = self.task.get_observation(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_reference_posterior_samples(self, index: int):
        if self.backend == "torch":
            return self.task.get_reference_posterior_samples(index)
        else:
            out = self.task.get_reference_posterior_samples(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_true_parameters(self, index: int):
        if self.backend == "torch":
            return self.task.get_true_parameters(index)
        else:
            out = self.task.get_true_parameters(index)
            if self.backend == "numpy":
                return out.numpy()
            elif self.backend == "jax":
                return jnp.array(out)

    def get_base_mask_fn(self):
        theta_dim = self.task.dim_parameters
        x_dim = self.task.dim_data
        thetas_mask = jnp.eye(theta_dim, dtype=jnp.bool_)
        x_mask = jnp.tril(jnp.ones((theta_dim, x_dim), dtype=jnp.bool_))
        base_mask = jnp.block(
            [
                [thetas_mask, jnp.zeros((theta_dim, x_dim))],
                [jnp.ones((x_dim, theta_dim)), x_mask],
            ]
        )
        base_mask = base_mask.astype(jnp.bool_)

        def base_mask_fn(node_ids, node_meta_data):
            return base_mask[node_ids, :][:, node_ids]

        return base_mask_fn

    def get_batch_sampler(self):
        def base_batch_sampler(
            key, batch_size, data, node_id, meta_data=None, num_devices=1
        ):
            assert data.ndim == 3, "Data must be 3D, (num_samples, num_nodes, dim)"
            assert (
                node_id.ndim == 2 or node_id.ndim == 1
            ), "Node id must be 2D or 1D, (num_nodes, dim) or (num_nodes,)"

            index = jax.random.randint(
                key,
                shape=(
                    num_devices,
                    batch_size,
                ),
                minval=0,
                maxval=data.shape[0],
            )
            data_batch = data[index, ...]
            node_id_batch = jnp.repeat(node_id[None, ...], num_devices, axis=0).astype(
                jnp.int32
            )
            if meta_data is not None:
                if meta_data.ndim == 3:
                    meta_data_batch = meta_data[index, ...]
                else:
                    meta_data_batch = jnp.repeat(
                        meta_data[None, ...], num_devices, axis=0
                    )
            else:
                meta_data_batch = None
            return data_batch, node_id_batch, meta_data_batch

        return base_batch_sampler


class BaseTask:
    def __init__(self, name: str, prior: Distribution, x_dim: int) -> None:
        self.name = name
        self.prior = prior
        self.theta_dim = prior.event_shape[0]
        self.x_dim = x_dim
        self.var_names = [f"theta{i}" for i in range(self.theta_dim)] + [
            f"x{i}" for i in range(self.x_dim)
        ]

    def _sample_theta(self, key, num_samples: int):
        return self.prior.sample(key, sample_shape=(num_samples,))

    def _simulate(self, key, theta: jnp.ndarray):
        raise NotImplementedError()

    def _sample_joint(self, key, num_samples: int):
        data = self.get_data(num_samples, key)
        theta = data["theta"]
        x = data["x"]
        params = {"theta{i}".format(i=i): theta[..., i] for i in range(self.theta_dim)}
        data = {"x{i}".format(i=i): x[..., i] for i in range(self.x_dim)}
        return {**params, **data}

    def get_prior(self):
        return self.prior

    def change_prior(self, prior: Distribution):
        assert isinstance(prior, Distribution), "Prior must be a Numpyro Distribution"
        assert (
            prior.event_shape[0] == self.theta_dim
        ), "Prior must have same dim as theta"
        self.prior = prior

    def get_theta_dim(self):
        return self.theta_dim

    def get_x_dim(self):
        return self.x_dim

    def get_data(self, num_samples: int, key):
        key_theta, key_x = jax.random.split(key)
        theta = self._sample_theta(key_theta, num_samples)
        x = self._simulate(key_x, theta)
        return {"theta": theta, "x": x}

    def get_posterior_observations(self, key, num_observations):
        key_mask, key_data = jax.random.split(key, 2)
        condition_mask_fn = get_condition_mask_fn("posterior")
        condition_masks = condition_mask_fn(
            key_mask, num_observations, self.get_theta_dim(), self.get_x_dim()
        )
        data = self.get_data(num_observations, key_data)
        theta = data["theta"]
        x = data["x"]
        return (condition_masks, x, theta)

    def get_observation_generator(self, condition_mask_fn="structured_random"):
        """
        Better to generate observations one by one for other tasks than posterior of theta.
        """
        condition_mask_fn = get_condition_mask_fn(condition_mask_fn)

        def observation_generator(key):
            while True:
                key, key_sample, key_condition_mask = jax.random.split(key, 3)
                condition_mask = condition_mask_fn(
                    key_condition_mask, 1, self.get_theta_dim(), self.get_x_dim()
                )[0]
                sample = self._sample_joint(key_sample, num_samples=1)
                conditioned_names = [
                    self.var_names[i]
                    for i in range(len(self.var_names))
                    if condition_mask[i]
                ]
                try:
                    x_o = jnp.concatenate(
                        [sample[var] for var in conditioned_names], axis=-1
                    )
                except:
                    x_o = jnp.array([])
                theta_o = jnp.concatenate(
                    [
                        sample[var]
                        for var in self.var_names
                        if var not in conditioned_names
                    ],
                    axis=-1,
                )
                yield (condition_mask, x_o, theta_o)

        return observation_generator

    def get_node_id(self):
        dim = self.theta_dim + self.x_dim
        return jnp.arange(dim)

    def get_batch_sampler(self):
        """
        used in sim.methods.run_score_transformer
        """
        return base_batch_sampler


class ToyGaussianTask(BaseTask):
    def __init__(
        self,
        prior: Distribution = Independent(
            Uniform(low=jnp.array([0.0, 0.01]), high=jnp.array([1.0, 1.0])),
            reinterpreted_batch_ndims=1,
        ),
        x_dim: int = 10,
    ):
        assert prior.event_shape[0] == 2, "Toy Gaussian has 2 parameters"
        super().__init__(name="toy_gaussian", prior=prior, x_dim=x_dim)

    def _simulate(self, key, theta: jnp.ndarray):
        num_samples = theta.shape[0]
        return (
            jax.random.normal(key, shape=(num_samples, self.x_dim))
            * theta[..., 1][..., None]
            + theta[..., 0][..., None]
        )


class OUPTask(BaseTask):
    def __init__(
        self,
        # uniform prior in original scale
        # prior: Distribution = Independent(
        #     Uniform(low=jnp.array([0.0, -2.0]), high=jnp.array([2.0, 2.0])),
        #     reinterpreted_batch_ndims=1,
        # ),
        prior: Distribution = Independent(
            Uniform(low=jnp.array([-1.0, -1.0]), high=jnp.array([1.0, 1.0])),
            reinterpreted_batch_ndims=1,
        ),
        x0: float = 10.0,
        dt: float = 0.2,
        num_points: int = 25,
        theta_shift: jnp.ndarray = jnp.array([1.0, 0.0]),
        theta_rescale: jnp.ndarray = jnp.array([1.0, 2.0]),
    ):
        assert prior.event_shape[0] == 2, "OUP has 2 parameters"
        super().__init__(name="oup", prior=prior, x_dim=num_points)
        self.x0 = x0
        self.dt = dt

        # calculated from 1 million simulations
        self.x_mean = 4.18954
        self.x_std = 3.0685966

        self.theta_shift = theta_shift
        self.theta_rescale = theta_rescale

    def _simulate(self, key, theta: jnp.ndarray):
        if theta.ndim == 1:
            theta = jnp.expand_dims(theta, 0)

        theta = theta * self.theta_rescale + self.theta_shift
        # noises
        x = jnp.zeros((*theta.shape[:-1], self.x_dim))
        x = x.at[..., 0].set(self.x0)
        dt = self.dt

        theta1_exp = jnp.exp(theta[..., 1])

        w = jax.random.normal(key, shape=(*theta.shape[:-1], self.x_dim))

        for t in range(self.x_dim - 1):
            mu, sigma = (
                theta[..., 0] * (theta1_exp - x[..., t]) * dt,
                0.5 * (dt**0.5) * w[..., t],
            )
            x = x.at[..., t + 1].set(x[..., t] + mu + sigma)
        x = (x - self.x_mean) / self.x_std
        return x

    def get_data(self, num_samples: int, key):
        key_theta, key_x = jax.random.split(key)
        theta = self._sample_theta(key_theta, num_samples)
        x = self._simulate(key_x, theta)
        return {"theta": theta, "x": x}


class TurinTask(BaseTask):
    def __init__(
        self,
        # uniform prior in original scale
        # prior: Distribution = Independent(
        #     Uniform(
        #         low=jnp.array([1e-9, 1e-9, 1e7, 1e-10]),
        #         high=jnp.array([1e-8, 1e-8, 5e9, 1e-9]),
        #     ),
        #     reinterpreted_batch_ndims=1,
        # ),
        prior: Distribution = Independent(
            Uniform(
                low=jnp.array([0.0, 0.0, 0.0, 0.0]),
                high=jnp.array([1.0, 1.0, 1.0, 1.0]),
            ),
            reinterpreted_batch_ndims=1,
        ),
        B: float = 5e8,
        tau0: float = 0.0,
        num_points: int = 101,
        theta_shift: jnp.ndarray = jnp.array([1e-9, 1e-9, 1e7, 1e-10]),
        theta_rescale: jnp.ndarray = jnp.array(
            [1e-8 - 1e-9, 1e-8 - 1e-9, 5e9 - 1e7, 1e-9 - 1e-10]
        ),
    ):
        assert prior.event_shape[0] == 4, "Turin has 4 parameters"
        super().__init__(name="turin", prior=prior, x_dim=num_points)
        self.B = B
        self.tau0 = tau0

        self.theta_shift = theta_shift
        self.theta_rescale = theta_rescale

    def _simulate(self, key, theta: jnp.ndarray) -> jnp.ndarray:
        if theta.ndim == 1:
            theta = jnp.expand_dims(theta, 0)

        theta = theta * self.theta_rescale + self.theta_shift

        simulations = []
        num_samples = theta.shape[0]

        for i in range(num_samples):
            params = theta[i]
            G0, T, lambda_0, sigma2_N = params[0], params[1], params[2], params[3]
            delta_f = self.B / (self.x_dim - 1)
            t_max = 1 / delta_f

            mu_poisson = lambda_0 * t_max

            key, subkey_poisson = jax.random.split(key)
            num_delay_points = int(
                jax.random.poisson(subkey_poisson, lam=mu_poisson, shape=())
            )

            key, subkey_delays = jax.random.split(key)
            delays = (
                jax.random.uniform(subkey_delays, shape=(num_delay_points,)) * t_max
            )
            delays = jnp.sort(delays)

            alpha = jnp.zeros((num_delay_points,), dtype=jnp.complex64)

            sigma2 = G0 * jnp.exp(-delays / T) / lambda_0 * self.B

            for l in range(num_delay_points):
                if delays[l] < self.tau0:
                    alpha = alpha.at[l].set(0.0 + 0.0j)
                else:
                    std_val = jnp.sqrt(sigma2[l] / 2)
                    std = jnp.where(std_val > 0, std_val, 1e-7)
                    key, subkey1 = jax.random.split(key)
                    key, subkey2 = jax.random.split(key)
                    real_part = jax.random.normal(subkey1, shape=()) * std
                    imag_part = jax.random.normal(subkey2, shape=()) * std
                    alpha = alpha.at[l].set(real_part + 1j * imag_part)

            phase_matrix = jnp.exp(
                -1j * 2 * jnp.pi * delta_f * jnp.outer(jnp.arange(self.x_dim), delays)
            )
            H = jnp.matmul(phase_matrix, alpha)

            key, subkey_noise1 = jax.random.split(key)
            key, subkey_noise2 = jax.random.split(key)
            noise_real = jax.random.normal(
                subkey_noise1, shape=(self.x_dim,)
            ) * jnp.sqrt(sigma2_N / 2)
            noise_imag = jax.random.normal(
                subkey_noise2, shape=(self.x_dim,)
            ) * jnp.sqrt(sigma2_N / 2)
            Noise = noise_real + 1j * noise_imag

            Y = H + Noise
            y = jnp.fft.ifft(Y)
            p = jnp.abs(y) ** 2
            out = 10 * jnp.log10(p)
            out_normalized = (out + 140.0) / 60.0
            simulations.append(out_normalized)

        return jnp.stack(simulations)

    def get_data(self, num_samples: int, key):
        key_theta, key_x = jax.random.split(key)
        theta = self._sample_theta(key_theta, num_samples)
        x = self._simulate(key_x, theta)
        return {"theta": theta, "x": x}


class BAVTask(BaseTask):
    def __init__(
        self,
        # gaussian prior in original scale
        # prior: Distribution = Independent(
        #     Normal(
        #         loc=jnp.array([jnp.log(2), jnp.log(2), jnp.log(5), jnp.log(0.3), 0]),
        #         scale=jnp.array([0.35, 0.35, 0.5, 0.35, 1]),
        #     ),
        #     reinterpreted_batch_ndims=1,
        # ),
        prior: Distribution = Independent(
            Normal(
                loc=jnp.array([0.0] * 5),
                scale=jnp.array([1.0] * 5),
            ),
            reinterpreted_batch_ndims=1,
        ),
        theta_shift: jnp.ndarray = jnp.array(
            [jnp.log(2), jnp.log(2), jnp.log(5), jnp.log(0.3), 0]
        ),
        theta_rescale: jnp.ndarray = jnp.array([0.35, 0.35, 0.5, 0.35, 1]),
    ):
        assert prior.event_shape[0] == 5, "BAV has 5 parameters"
        super().__init__(name="bav", prior=prior, x_dim=98)

        self.x_mean = jnp.array(BAV_X_MEAN)
        self.x_std = jnp.array(BAV_X_STD)

        self.theta_shift = theta_shift
        self.theta_rescale = theta_rescale

    def _simulate(self, key, theta: jnp.ndarray):
        theta = jnp.atleast_2d(theta)
        theta = theta * self.theta_rescale + self.theta_shift
        theta_cpu = jax.device_put(theta, device=jax.devices("cpu")[0])
        theta_torch = torch.tensor(np.array(theta_cpu), device="cpu")

        set_seed(int(key[1]))

        xs = []
        for params in theta_torch:
            x = sample_bav_responses_flat(theta=params, N=1).squeeze()
            xs.append(jnp.array(x.numpy()))
        xs = jnp.array(xs)
        xs = (xs - self.x_mean) / self.x_std
        return xs


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    task = get_task("bav")
    data = task.get_data(num_samples=10000, key=key)
    print(data["theta"].shape, data["x"].shape)
