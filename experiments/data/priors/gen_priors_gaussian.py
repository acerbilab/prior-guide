"""
Generate test-time priors under Gaussian training priors.
"""

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logit

BAV_PRIOR = {
    "task": "bav",
    "dist": "gaussian",
    "type": "training",
    "mu": [0.0] * 5,
    "sigma": [1.0] * 5,
    "mu_ori": [np.log(5), np.log(5), np.log(5), np.log(1), logit(0.02), 0, 0],
    "sigma_ori": [0.5, 0.5, 0.5, 1, 0.5, 1, 5],
}
GL_PRIOR = {
    "task": "gaussian_linear",
    "dist": "gaussian",
    "type": "training",
    "mu": [0.0] * 10,
    "sigma": [np.sqrt(0.1)] * 10,
}
GL_HIGH_PRIOR = {
    "task": "gaussian_linear_high",
    "dist": "gaussian",
    "type": "training",
    "mu": [0.0] * 20,
    "sigma": [np.sqrt(0.1)] * 20,
}


def main():
    gaussian_priors = [BAV_PRIOR, GL_PRIOR, GL_HIGH_PRIOR]

    seed = 0
    key = jax.random.PRNGKey(seed)

    for prior in gaussian_priors:
        # Generate the prior parameters
        task = prior["task"]
        mu = jnp.array(prior["mu"])
        sigma = jnp.array(prior["sigma"])
        low = mu - 2 * sigma
        high = mu + 2 * sigma

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), task)
        os.makedirs(path, exist_ok=True)

        json.dump(
            prior,
            open(os.path.join(path, "training.json"), "w"),
            indent=4,
        )

        for i in range(10):
            key, key_mild, key_strong, key_mixture = jax.random.split(key, 4)
            sigma_mild = 0.5 * sigma
            mu_mild = jax.random.uniform(
                key=key_mild,
                shape=sigma_mild.shape,
                minval=low + 3 * sigma_mild,
                maxval=high - 3 * sigma_mild,
            )
            sigma_strong = 0.2 * sigma
            mu_strong = jax.random.uniform(
                key=key_strong,
                shape=sigma_strong.shape,
                minval=low + 3 * sigma_strong,
                maxval=high - 3 * sigma_strong,
            )
            sigma_mixture = 0.2 * sigma
            key_mixture_1, key_mixture_2 = jax.random.split(key_mixture)
            mu1 = jax.random.uniform(
                key=key_mixture_1,
                shape=sigma_mixture.shape,
                minval=low + 3 * sigma_mixture,
                maxval=high - 3 * sigma_mixture,
            )
            mu2 = jax.random.uniform(
                key=key_mixture_2,
                shape=sigma_mixture.shape,
                minval=low + 3 * sigma_mixture,
                maxval=high - 3 * sigma_mixture,
            )
            pi = jax.random.uniform(key=key_mixture, minval=0.2, maxval=0.8)
            q_mild = {
                "task": task,
                "dist": "gaussian",
                "type": "mild",
                "mu": mu_mild.tolist(),
                "sigma": sigma_mild.tolist(),
            }
            q_strong = {
                "task": task,
                "dist": "gaussian",
                "type": "strong",
                "mu": mu_strong.tolist(),
                "sigma": sigma_strong.tolist(),
            }
            q_mixture = {
                "task": task,
                "dist": "mixture",
                "type": "mixture",
                "mu": jnp.array([mu1, mu2]).tolist(),
                "sigma": jnp.array([sigma_mixture, sigma_mixture]).tolist(),
                "pi": jnp.array([pi, 1 - pi]).tolist(),
            }

            json.dump(
                q_mild,
                open(os.path.join(path, f"mild_{i}.json"), "w"),
                indent=4,
            )
            json.dump(
                q_strong,
                open(os.path.join(path, f"strong_{i}.json"), "w"),
                indent=4,
            )
            json.dump(
                q_mixture,
                open(os.path.join(path, f"mixture_{i}.json"), "w"),
                indent=4,
            )


if __name__ == "__main__":
    main()
