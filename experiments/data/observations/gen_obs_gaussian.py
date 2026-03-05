"""
Generate observations under Gaussian training priors.
Loading prior specifications from the priors/ directory.
"""

import argparse
import os

os.environ["JAX_PLATFORMS"] = "cpu"
import json
import logging
from pathlib import Path

import jax
import jax.numpy as jnp
import pyro.distributions as pdist
import torch
from numpyro.distributions import Categorical, Independent, Mixture, Normal

from experiments.utils import set_seed
from priorg.sim.tasks.task import BAVTask, SBIBMTask

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


def main():
    parser = argparse.ArgumentParser(description="Generate observations")
    parser.add_argument(
        "--task",
        type=str,
        default="gaussian_linear",
        choices=["bav", "gaussian_linear", "gaussian_linear_high"],
        help="Task name (default: bav)",
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

    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(
        f"Task: {args.task}, Prior Type: {args.prior_type}, Prior ID: {args.prior_id}"
    )

    prior_path = os.path.join(
        Path(__file__).parent.parent,
        f"priors/{args.task}/{args.prior_type}_{args.prior_id}.json",
    )

    obs_path = os.path.join(
        Path(__file__).parent.parent,
        f"observations/{args.task}/{args.prior_type}_{args.prior_id}",
    )
    os.makedirs(obs_path, exist_ok=True)

    if args.task == "bav":
        prior_params = json.load(open(prior_path, "r"))
        if args.prior_type == "mild" or args.prior_type == "strong":
            mu = jnp.array(prior_params["mu"])
            sigma = jnp.array(prior_params["sigma"])
            prior_dist = Independent(
                Normal(loc=jnp.array(mu), scale=jnp.array(sigma)),
                reinterpreted_batch_ndims=1,
            )
        elif args.prior_type == "mixture":
            weights = jnp.array(prior_params["pi"])
            means = jnp.array(prior_params["mu"])
            stds = jnp.array(prior_params["sigma"])
            prior_dist = Mixture(
                mixing_distribution=Categorical(probs=weights),
                component_distributions=Independent(
                    Normal(loc=means, scale=stds),
                    reinterpreted_batch_ndims=1,
                ),
            )
        else:
            raise ValueError(f"Unknown prior type: {args.prior_type}")
        task = BAVTask(prior=prior_dist)
    elif args.task == "gaussian_linear":
        prior_params = json.load(open(prior_path, "r"))
        if args.prior_type == "mild" or args.prior_type == "strong":
            mu = torch.tensor(prior_params["mu"])
            sigma = torch.tensor(prior_params["sigma"])
            cov = torch.diag_embed(sigma**2)
            prior_dist = pdist.MultivariateNormal(loc=mu, covariance_matrix=cov)
        elif args.prior_type == "mixture":
            weights = torch.tensor(prior_params["pi"])
            means = torch.tensor(prior_params["mu"])
            stds = torch.tensor(prior_params["sigma"])
            covs = torch.diag_embed(stds**2)
            prior_dist = pdist.MixtureSameFamily(
                mixture_distribution=pdist.Categorical(probs=weights),
                component_distribution=pdist.MultivariateNormal(
                    loc=means, covariance_matrix=covs
                ),
            )
        else:
            raise ValueError(f"Unknown prior type: {args.prior_type}")
        task = SBIBMTask(name="gaussian_linear", p_dist=prior_dist)
    elif args.task == "gaussian_linear_high":
        prior_params = json.load(open(prior_path, "r"))
        if args.prior_type == "mild" or args.prior_type == "strong":
            mu = torch.tensor(prior_params["mu"])
            sigma = torch.tensor(prior_params["sigma"])
            cov = torch.diag_embed(sigma**2)
            prior_dist = pdist.MultivariateNormal(loc=mu, covariance_matrix=cov)
        elif args.prior_type == "mixture":
            weights = torch.tensor(prior_params["pi"])
            means = torch.tensor(prior_params["mu"])
            stds = torch.tensor(prior_params["sigma"])
            covs = torch.diag_embed(stds**2)
            prior_dist = pdist.MixtureSameFamily(
                mixture_distribution=pdist.Categorical(probs=weights),
                component_distribution=pdist.MultivariateNormal(
                    loc=means, covariance_matrix=covs
                ),
            )
        else:
            raise ValueError(f"Unknown prior type: {args.prior_type}")
        task = SBIBMTask(name="gaussian_linear_high", p_dist=prior_dist)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    for obs_seed in observation_seeds:
        key = jax.random.PRNGKey(obs_seed)
        set_seed(obs_seed)
        data = task.get_data(num_samples=1, key=key)
        theta_o = data["theta"]
        x_o = data["x"]
        obs = {
            "seed": obs_seed,
            "theta": theta_o.squeeze().tolist(),
            "x": x_o.squeeze().tolist(),
        }
        json.dump(
            obs,
            open(os.path.join(obs_path, f"obs_{obs_seed}.json"), "w"),
            indent=4,
        )
        logger.info(
            f"obs_seed: {obs_seed}, theta shape: {theta_o.shape}, x shape: {x_o.shape}"
        )


if __name__ == "__main__":
    main()
