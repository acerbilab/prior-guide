"""
Generate test-time priors under uniform training priors.
"""

import json
import os

import jax
import jax.numpy as jnp

TWO_MOONS_PRIOR = {
    "task": "two_moons",
    "dist": "uniform",
    "type": "training",
    "low": [-1.0] * 2,
    "high": [1.0] * 2,
}
OUP_PRIOR = {
    "task": "oup",
    "dist": "uniform",
    "type": "training",
    "low_ori": [0.0, -2.0],
    "high_ori": [2.0, 2.0],
    "low": [-1.0, -1.0],
    "high": [1.0, 1.0],
}
TURIN_PRIOR = {
    "task": "turin",
    "dist": "uniform",
    "type": "training",
    "low_ori": [1e-9, 1e-9, 1e7, 1e-10],
    "high_ori": [1e-8, 1e-8, 5e9, 1e-9],
    "low": [0.0] * 4,
    "high": [1.0] * 4,
}


def main():
    # Create dictionaries to hold the priors
    uniform_priors = [OUP_PRIOR, TURIN_PRIOR, TWO_MOONS_PRIOR]

    seed = 0
    key = jax.random.PRNGKey(seed)

    for prior in uniform_priors:
        # Generate the prior parameters
        task = prior["task"]
        low = jnp.array(prior["low"])
        high = jnp.array(prior["high"])

        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), task)

        os.makedirs(path, exist_ok=True)

        json.dump(
            prior,
            open(os.path.join(path, "training.json"), "w"),
            indent=4,
        )

        for i in range(10):
            key, key_mild, key_strong, key_mixture = jax.random.split(key, 4)
            sigma_mild = 0.5 * (high - low) / jnp.sqrt(12)
            mu_mild = jax.random.uniform(
                key=key_mild,
                shape=sigma_mild.shape,
                minval=low + 3 * sigma_mild,
                maxval=high - 3 * sigma_mild,
            )
            sigma_strong = 0.2 * (high - low) / jnp.sqrt(12)
            mu_strong = jax.random.uniform(
                key=key_strong,
                shape=sigma_strong.shape,
                minval=low + 3 * sigma_strong,
                maxval=high - 3 * sigma_strong,
            )
            sigma_mixture = 0.2 * (high - low) / jnp.sqrt(12)
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
