import argparse
import json
import logging
import os
from pathlib import Path

import jax
import jax.numpy as jnp

observation_seeds = [
    1000000,  # observation 1
    1000001,  # observation 2
    1000002,  # observation 3
    1000003,  # observation 4
    1000004,  # observation 5
    1000005,  # observation 6
    1000010,  # observation 7
    1000012,  # observation 8
    1000008,  # observation 9
    1000009,  # observation 10
]


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Run prior guide")
    parser.add_argument(
        "--frac_conditioned",
        type=float,
        default=0.3,
        help="Fraction of conditioned x (default: 0.3)",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    args = parser.parse_args()

    logger.info(f"frac_conditioned: {args.frac_conditioned}, seed: {args.seed}")

    exp_path = Path(__file__).parent.parent

    key = jax.random.PRNGKey(args.seed)

    for task in ["oup", "turin"]:
        for prior_type in ["mild", "strong", "mixture"]:
            for prior_id in range(10):
                condition_mask_path = (
                    exp_path
                    / "posterior_predictive"
                    / "condition_masks"
                    / task
                    / f"{prior_type}_{prior_id}"
                )
                os.makedirs(condition_mask_path, exist_ok=True)
                key, key_select_obs = jax.random.split(key, 2)
                # select 5 observations from the 10 available
                selected_obs_seeds = jax.random.choice(
                    key_select_obs,
                    jnp.array(observation_seeds),
                    shape=(5,),
                    replace=False,
                )
                for i, obs_seed in enumerate(observation_seeds):
                    obs_path = (
                        exp_path
                        / "data"
                        / "observations"
                        / task
                        / f"{prior_type}_{prior_id}"
                        / f"obs_{obs_seed}.json"
                    )
                    obs = json.load(open(obs_path, "r"))
                    theta_o = jnp.atleast_1d(jnp.array(obs["theta"]))
                    x_o = jnp.atleast_1d(jnp.array(obs["x"]))
                    theta_dim = theta_o.shape[0]
                    x_dim = x_o.shape[0]
                    key, key_condition_mask = jax.random.split(key, 2)
                    condition_mask = jnp.zeros(x_dim, dtype=jnp.bool)

                    if obs_seed in selected_obs_seeds:
                        # Condition on the first 30% of the x dimension
                        conditioned_indices = jnp.arange(
                            int(round(x_dim * args.frac_conditioned))
                        )
                    else:
                        # Condition on the last 30% of the x dimension
                        conditioned_indices = jnp.arange(
                            -int(round(x_dim * args.frac_conditioned)), 0
                        )
                    # conditioned_indices = jax.random.choice(
                    #     key_condition_mask,
                    #     x_dim,
                    #     shape=(int(round(x_dim * args.frac_conditioned)),),
                    #     replace=False,
                    # )
                    condition_mask = condition_mask.at[conditioned_indices].set(True)
                    condition_mask = jnp.concatenate(
                        [
                            jnp.array([False] * theta_dim, dtype=jnp.bool),
                            condition_mask,
                        ]
                    )
                    json.dump(
                        condition_mask.tolist(),
                        open(
                            os.path.join(
                                condition_mask_path,
                                f"mask_obs{obs_seed}.json",
                            ),
                            "w",
                        ),
                        indent=4,
                    )
