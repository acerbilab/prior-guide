import jax
import jax.numpy as jnp

# from sim.utils.sdeint import register_method
# from functools import partial
from jax import grad, jacfwd


# Sort the dimesnionality of the score function
def predict_x0(score_fn, model, t, x, condition_mask=None):
    """Predicts x0 using the score function and model parameters."""
    score = score_fn(t, x)  #
    x0_mean_full = (
        x + model.sde.marginal_stddev(t, jnp.array([1.0])) ** 2 * score
    ) / model.sde.marginal_mean(t, jnp.array([1.0]))

    return x0_mean_full


# Replace jax.lax.scan with this during debugging
def debug_scan(f, init, xs):
    carry = init
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, jnp.array(ys)


def prior_guide_theta_prior_only(
    model,
    key,
    condition_mask,
    x_o,
    x_T,
    theta_prior_log_weights,  # shape (k,)
    theta_prior_mean,  # shape (k, l_p,)
    theta_prior_cov,  # shape (k, l_p,l_p)
    num_steps=500,
    rho=2.0,
    prior_dim=2,
    theta_original_prior_cov=None,  # shape (k, l_p,)
    clipping_range=None,  # shape (2, l_p,)
    verbose=False,
    langevin_steps=0,  # Number of Langevin steps to perform between diffusion steps
    langevin_ratio=0.5,  # Ratio of Langevin noise to diffusion noise
):
    """
    A variant that:
      - Assumes the first prior_dim coordinates of x_T are 'theta',
      - Applies a Gaussian prior only to those dimensions,
      - Leaves the other coords unaffected by the prior directly,
        but they can still be indirectly influenced.
      - Interleaves Langevin dynamics steps between diffusion reverse steps.
    """

    # Time discretization
    ts = jnp.linspace(model.T_min, model.T_max, num_steps) ** rho

    # Score function initialization (no guidance mask here)
    score_fn = model._init_score(
        node_id=model.node_id,
        condition_mask=jnp.zeros_like(condition_mask),
        edge_mask=None,
        meta_data=None,
    )

    # Helper function to compute the guided score at a given point and time
    def compute_guided_score(x, t):
        # Predict the x0-mean
        x_0_mean = predict_x0(score_fn, model, t, x)
        sigma = model.sde.marginal_stddev(t, jnp.array([1.0]))

        if theta_original_prior_cov is not None:
            # Sigma_post based on original prior covariance
            Sigma_post = jnp.linalg.inv(
                jnp.linalg.inv(theta_original_prior_cov)
                + (1 / sigma**2) * jnp.eye(prior_dim)
            )
        else:
            # Sigma_post isotropic assumption
            x0_var = sigma**2 / (1 + sigma**2)
            Sigma_post = x0_var * jnp.eye(theta_prior_mean.shape[-1])

        Sigmas_combined = (
            theta_prior_cov + Sigma_post[None, :, :]
        )  # (k,4,4) as (k,4,4) @ (1,4,4)  - k is the number of mixtures
        cur_theta = x_0_mean[:prior_dim]

        if verbose and t == ts[-1]:  # Only print for first step to avoid flooding
            jax.debug.print("Verbose mode --------------------------------------")
            jax.debug.print("t {x}", x=t)
            jax.debug.print("x_0_mean {x}", x=x_0_mean)
            jax.debug.print("sigma {x}", x=sigma)
            jax.debug.print("Sigma_post {x}", x=Sigma_post)
            jax.debug.print("Sigmas_combined {x}", x=Sigmas_combined)
            jax.debug.print("cur_theta {x}", x=cur_theta)

        # Computations to be used in weights
        residual = (
            theta_prior_mean - cur_theta[None, :]
        )  # shape (k, l_p) - k is the number of mixtu
        Sigma_combined_inv = jnp.linalg.inv(Sigmas_combined)  # shape (k, l_p, l_p)
        residual_times_inverse = jnp.matmul(
            residual[:, None, :], Sigma_combined_inv
        )  # shape (k,1,l_p)

        # Build the weight matrix term by term for k mixtures
        first_term = -0.5 * jnp.matmul(
            residual_times_inverse, residual[..., None]
        )  # (k,1,1)
        second_term = -0.5 * jnp.linalg.slogdet(Sigmas_combined)[1]  # (k,)
        weights = jax.nn.softmax(
            first_term.squeeze() + second_term + theta_prior_log_weights
        )  # (k,)

        grad_x0_mean = jacfwd(lambda x_in: predict_x0(score_fn, model, t, x_in))(
            x
        )  # shape (D, D) the full jacobian where D is concat
        grad_x0_mean = grad_x0_mean[
            :prior_dim, :
        ]  # shape (l_p, D) As we only want the prior part

        # Building eqn 18
        # Einsum magic its the sums of the residuals times the inverse of the covariance for each mixture
        # Compute scaled residuals: (2, d) -> (2, d) after applying (d, d) covariance inverse
        scaled_residuals = jnp.einsum("bi, bij -> bj", residual, Sigma_combined_inv)
        weighted_residual = jnp.sum(
            weights[:, None] * scaled_residuals, axis=0
        )  # Final shape: (l_p) rhs of eqn 18 without gradient

        p_y_xt_grad = jnp.matmul(
            weighted_residual[None, :], grad_x0_mean
        ).squeeze()  # shape (D,) full dimension

        # eqn 18 rhs is adjusted by SD and some correction
        x_0_mean_new = x_0_mean + p_y_xt_grad * model.sde.marginal_stddev(
            t, jnp.array([1.0])
        ) ** 2 / model.sde.marginal_mean(t, jnp.array([1.0]))

        if clipping_range is not None:
            x_0_mean_new = jnp.clip(x_0_mean_new, clipping_range[0], clipping_range[1])
        else:
            x_0_mean_new = jnp.clip(
                x_0_mean_new, -50, 50
            )  # change this to something sensible

        if verbose and t == ts[-1]:
            jax.debug.print("x_0_mean_new {x}", x=x_0_mean_new)

        # Tweedies formula conversion for guided score
        guided_score = (
            x_0_mean_new * model.sde.marginal_mean(t, jnp.array([1.0])) - x
        ) / model.sde.marginal_stddev(t, jnp.array([1.0])) ** 2

        return guided_score

    # We'll define the scanning function that goes from t1 -> t0
    def scan_fn(carry, t0):
        key, t1, x1 = carry
        dt = t0 - t1

        # --- First run Langevin dynamics steps (at time t1) ---
        # Expand x_o to full dimensions for conditioning
        x_o_full = jnp.zeros_like(x1)
        x_o_full = x_o_full.at[condition_mask].set(x_o)

        # Calculate adaptive step size based on diffusion characteristics
        # We'll use the diffusion coefficient at t1 to set the step size
        diffusion_coeff = model.sde.diffusion(t1, x1)
        diffusion_variance = diffusion_coeff**2 * jnp.abs(dt)
        langevin_step_size = (langevin_ratio * diffusion_variance) / 2.0

        # Run multiple Langevin steps at time t1
        x_langevin = x1  # Start from the current point

        def langevin_step(state_carry, _):
            x_state, step_key = state_carry
            step_key, noise_key = jax.random.split(step_key)
            # Use the shared guided score function
            guided_score = compute_guided_score(x_state, t1)

            # Langevin update (only on unconditioned dimensions)
            noise = jax.random.normal(noise_key, shape=x_state.shape)
            grad_step = (1 - condition_mask) * langevin_step_size * guided_score
            eps = 1e-8
            noise_term = (
                (1 - condition_mask) * jnp.sqrt(2 * langevin_step_size + eps) * noise
            )

            x_updated = x_state + grad_step + noise_term

            # Ensure conditions are still respected
            x_updated = condition_mask * x_o_full + (1 - condition_mask) * x_updated
            return (x_updated, step_key), x_updated

        # Perform multiple Langevin steps
        key, langevin_key = jax.random.split(key)
        init_carry = (x_langevin, langevin_key)
        (final_langevin_x, final_langevin_key), _ = jax.lax.scan(
            langevin_step, init_carry, jnp.arange(langevin_steps)
        )
        # (final_langevin_x, final_langevin_key), _ = debug_scan(
        #     langevin_step, init_carry, jnp.arange(langevin_steps)
        # )

        # Use the result from Langevin steps for the diffusion update
        x1_after_langevin = final_langevin_x
        key = final_langevin_key

        # --- Now perform regular diffusion reverse step ---
        key, subkey = jax.random.split(key)

        # Compute guided score using the same function
        guided_score = compute_guided_score(x1_after_langevin, t1)

        # VE SDE update
        drift_backward = (1 - condition_mask) * (
            model.sde.drift(t1, x1_after_langevin)
            - model.sde.diffusion(t1, x1_after_langevin) ** 2 * guided_score
        )
        diffusion_backward = (1 - condition_mask) * (
            model.sde.diffusion(t1, x1_after_langevin)
        )
        xt0_uncond = (
            x1_after_langevin
            + drift_backward * dt
            + diffusion_backward
            * jnp.sqrt(jnp.abs(dt))
            * jax.random.normal(subkey, shape=x1_after_langevin.shape)
        )

        # Apply conditioning
        xt0 = condition_mask * x_o_full + (1 - condition_mask) * xt0_uncond

        return (key, t0, xt0), xt0

    # Run the chain backwards from T to 0
    carry_init = (key, ts[-1], x_T)
    _, traj = jax.lax.scan(scan_fn, carry_init, ts[::-1][1:])
    # _, traj = debug_scan(scan_fn, carry_init, ts[::-1][1:])
    return traj[-1]
