import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from priorg.sim.methods.guidance_gmm import prior_guide_theta_prior_only

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


def evaluate():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    parser = argparse.ArgumentParser(description="Run prior guide")
    parser.add_argument(
        "--task",
        type=str,
        default="oup",
        choices=["oup", "turin"],
        help="Task name (default: oup)",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        default="mild",
        choices=["mild", "strong", "mixture"],
        help="Prior type (default: mild)",
    )
    parser.add_argument(
        "--prior_id",
        type=int,
        default=0,
        choices=list(range(10)),
        help="Prior ID (default: 0)",
    )
    parser.add_argument(
        "--model_id",
        type=int,
        default=0,
        choices=list(range(5)),
        help="Model ID (default: 0)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to generate (default: 100)",
    )
    parser.add_argument(
        "--num_steps", type=int, default=25, help="Number of steps in prior guide"
    )
    parser.add_argument("--rho", type=float, default=2.0, help="Rho in prior guide")
    parser.add_argument(
        "--langevin_steps",
        type=int,
        default=8,
        help="Number of Langevin steps in prior guide",
    )
    parser.add_argument(
        "--langevin_ratio",
        type=float,
        default=0.5,
        help="Langevin ratio in prior guide",
    )

    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="number of batches to split the num_samples into (default: 2)",
    )
    parser.add_argument(
        "--run_prior_guide",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Run PriorGuide sampling (default: True)",
    )
    parser.add_argument(
        "--run_simformer",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Run Simformer sampling (without guide) (default: True)",
    )
    parser.add_argument(
        "--obs_seed",
        type=int,
        default=None,
        help="Observation seed (default: None, in which case all observation seeds are used as listed at the top of the script)",
    )

    parser.add_argument("--seed", type=int, default=0, help="Random seed (default: 0)")

    args = parser.parse_args()

    logger.info(
        f"Running prior guide with task: {args.task}, prior_type: {args.prior_type}, prior_id: {args.prior_id}, model_id: {args.model_id}, num_samples: {args.num_samples}"
    )

    exp_path = Path(__file__).parent.parent

    prior_path = (
        exp_path
        / "data"
        / "priors"
        / args.task
        / f"{args.prior_type}_{args.prior_id}.json"
    )

    condition_mask_path = (
        exp_path
        / "posterior_predictive"
        / "condition_masks"
        / args.task
        / f"{args.prior_type}_{args.prior_id}"
    )

    prior_guide_path = (
        exp_path
        / "posterior_predictive"
        / "prior_guide"
        / args.task
        / f"model_{args.model_id}"
        / f"{args.prior_type}_{args.prior_id}"
    )
    os.makedirs(prior_guide_path, exist_ok=True)

    simformer_path = (
        exp_path
        / "posterior_predictive"
        / "simformer"
        / args.task
        / f"model_{args.model_id}"
        / f"{args.prior_type}_{args.prior_id}"
    )
    os.makedirs(simformer_path, exist_ok=True)

    model_path = exp_path / "models" / args.task / f"model_{args.model_id}.pkl"

    # Load the model
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    q_params = json.load(open(prior_path, "r"))

    if args.prior_type == "mild" or args.prior_type == "strong":
        mu = jnp.array(q_params["mu"])
        sigma = jnp.array(q_params["sigma"])
        cov = jnp.diag(sigma**2)
        pi = jnp.array([1.0])
    elif args.prior_type == "mixture":
        pi = jnp.array(q_params["pi"])
        mu = jnp.array(q_params["mu"])
        sigma = jnp.array(q_params["sigma"])
        cov = (sigma**2)[..., None] * jnp.eye(sigma.shape[-1])
    else:
        raise ValueError(f"Unknown prior type: {args.prior_type}")

    key = jax.random.PRNGKey(args.seed)

    for obs_seed in observation_seeds if args.obs_seed is None else [args.obs_seed]:
        key, key_obs = jax.random.split(key, 2)
        obs_path = (
            exp_path
            / "data"
            / "observations"
            / args.task
            / f"{args.prior_type}_{args.prior_id}"
            / f"obs_{obs_seed}.json"
        )
        obs = json.load(open(obs_path, "r"))
        theta_o = jnp.atleast_1d(jnp.array(obs["theta"]))
        x_o = jnp.atleast_1d(jnp.array(obs["x"]))

        theta_dim = theta_o.shape[0]

        condition_mask = jnp.array(
            json.load(
                open(
                    os.path.join(condition_mask_path, f"mask_obs{obs_seed}.json"),
                    "r",
                )
            )
        )

        x_conditioned = x_o[condition_mask[theta_dim:]]

        @jax.jit
        def sample_once_prior(key):
            x_T = model.sde.marginal_stddev(
                jnp.array([model.T_max]), jnp.array([1.0])
            ) * jax.random.normal(key, shape=(int(theta_o.shape[0] + x_o.shape[0]),))
            return prior_guide_theta_prior_only(
                model=model,
                key=key,
                condition_mask=condition_mask,
                x_o=x_conditioned,
                x_T=x_T,
                theta_prior_log_weights=jnp.log(pi),
                theta_prior_mean=mu,
                theta_prior_cov=cov,
                prior_dim=theta_o.shape[0],
                num_steps=args.num_steps,
                rho=args.rho,
                langevin_steps=args.langevin_steps,
                langevin_ratio=args.langevin_ratio,
            )

        samples_prior_guide_batchified = []
        samples_wo_guide_batchified = []

        start_time = time.time()

        for batch in range(args.num_batches):
            batch_size = int(args.num_samples / args.num_batches)
            logger.info(f"Processing batch {batch + 1}/{args.num_batches}")

            key_obs, key_prior_guide, key_simformer = jax.random.split(key_obs, 3)

            # Sample posterior using PriorGuide
            if args.run_prior_guide:
                keys = jax.random.split(key_prior_guide, batch_size)
                batched_sample_prior = jax.vmap(sample_once_prior)
                samples_with_prior = batched_sample_prior(keys)
                samples_with_prior = jnp.array(samples_with_prior)
                # Append to batchified list
                samples_prior_guide_batchified.append(samples_with_prior)

            # Sample posterior without prior (Simformer)
            if args.run_simformer:
                simformer_samples = model.sample(
                    batch_size,
                    x_o=x_conditioned,
                    condition_mask=condition_mask,
                    rng=key_simformer,
                )
                # Append to batchified list
                samples_wo_guide_batchified.append(simformer_samples)

        end_time = time.time()
        logger.info(f"Time taken in total: {end_time - start_time} seconds")

        # Process and save results for each method separately
        if args.run_prior_guide:
            # Concatenate the batchified samples
            samples_prior_guide = jnp.concatenate(
                samples_prior_guide_batchified, axis=0
            )
            logger.info(f"Samples with prior shape: {samples_prior_guide.shape}")
            jnp.savez(
                os.path.join(
                    prior_guide_path,
                    f"samples_{args.num_samples}_obs_{obs_seed}.npz",
                ),
                samples=samples_prior_guide,
                theta=theta_o,
                x=x_o,
            )

        if args.run_simformer:
            # Concatenate the batchified samples
            samples_wo_guide = jnp.concatenate(samples_wo_guide_batchified, axis=0)
            logger.info(f"Samples without prior shape: {samples_wo_guide.shape}")
            jnp.savez(
                os.path.join(
                    simformer_path, f"samples_{args.num_samples}_obs_{obs_seed}.npz"
                ),
                samples=samples_wo_guide,
                theta=theta_o,
                x=x_o,
            )


if __name__ == "__main__":
    evaluate()
