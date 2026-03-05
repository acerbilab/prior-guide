"""
Generate observations under uniform training priors.
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
import torch

# use numpyro for custom tasks: OUP, Turin, BAV
from numpyro.distributions import Categorical, Independent, Mixture, TruncatedNormal

# use pyro/torch for SBIBM tasks: Two Moons, Gaussian Linear
from pyro import distributions as pdist

from experiments.utils import TruncatedNormal as TruncatedNormalTorch
from experiments.utils import set_seed
from priorg.sim.tasks.task import OUPTask, SBIBMTask, TurinTask

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
    parser = argparse.ArgumentParser(
        description="Generate observations under uniform training priors"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="two_moons",
        choices=["two_moons", "oup", "turin"],
        help="Task name (default: two_moons)",
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

    training_prior_path = os.path.join(
        Path(__file__).parent.parent,
        f"priors/{args.task}/training.json",
    )

    training_prior_params = json.load(open(training_prior_path, "r"))

    posterior_path = os.path.join(
        Path(__file__).parent.parent,
        f"observations/{args.task}/{args.prior_type}_{args.prior_id}",
    )
    os.makedirs(posterior_path, exist_ok=True)

    low = training_prior_params["low"]
    high = training_prior_params["high"]

    if args.task == "oup":
        prior_params = json.load(open(prior_path, "r"))
        if args.prior_type == "mild" or args.prior_type == "strong":
            mu = jnp.array(prior_params["mu"])
            sigma = jnp.array(prior_params["sigma"])
            prior_dist = Independent(
                TruncatedNormal(
                    loc=jnp.array(mu),
                    scale=jnp.array(sigma),
                    low=jnp.array(low),
                    high=jnp.array(high),
                ),
                reinterpreted_batch_ndims=1,
            )
        elif args.prior_type == "mixture":
            weights = jnp.array(prior_params["pi"])
            means = jnp.array(prior_params["mu"])
            stds = jnp.array(prior_params["sigma"])
            prior_dist = Mixture(
                mixing_distribution=Categorical(probs=weights),
                component_distributions=Independent(
                    TruncatedNormal(
                        loc=means, scale=stds, low=jnp.array(low), high=jnp.array(high)
                    ),
                    reinterpreted_batch_ndims=1,
                ),
            )
        else:
            raise ValueError(f"Unknown prior type: {args.prior_type}")
        task = OUPTask(prior=prior_dist)
    elif args.task == "turin":
        prior_params = json.load(open(prior_path, "r"))
        if args.prior_type == "mild" or args.prior_type == "strong":
            mu = jnp.array(prior_params["mu"])
            sigma = jnp.array(prior_params["sigma"])
            prior_dist = Independent(
                TruncatedNormal(
                    loc=jnp.array(mu),
                    scale=jnp.array(sigma),
                    low=jnp.array(low),
                    high=jnp.array(high),
                ),
                reinterpreted_batch_ndims=1,
            )
        elif args.prior_type == "mixture":
            weights = jnp.array(prior_params["pi"])
            means = jnp.array(prior_params["mu"])
            stds = jnp.array(prior_params["sigma"])
            prior_dist = Mixture(
                mixing_distribution=Categorical(probs=weights),
                component_distributions=Independent(
                    TruncatedNormal(
                        loc=means, scale=stds, low=jnp.array(low), high=jnp.array(high)
                    ),
                    reinterpreted_batch_ndims=1,
                ),
            )
        else:
            raise ValueError(f"Unknown prior type: {args.prior_type}")
        task = TurinTask(prior=prior_dist)
    elif args.task == "two_moons":
        prior_params = json.load(open(prior_path, "r"))
        if args.prior_type == "mild" or args.prior_type == "strong":
            mu = torch.tensor(prior_params["mu"])
            sigma = torch.tensor(prior_params["sigma"])
            prior_dist = pdist.Independent(
                TruncatedNormalTorch(
                    loc=mu, scale=sigma, a=torch.tensor(low), b=torch.tensor(high)
                ),
                reinterpreted_batch_ndims=1,
            )
        elif args.prior_type == "mixture":
            weights = torch.tensor(prior_params["pi"])
            means = torch.tensor(prior_params["mu"])
            stds = torch.tensor(prior_params["sigma"])
            prior_dist = pdist.MixtureSameFamily(
                mixture_distribution=pdist.Categorical(probs=weights),
                component_distribution=pdist.Independent(
                    TruncatedNormalTorch(
                        loc=means, scale=stds, a=torch.tensor(low), b=torch.tensor(high)
                    ),
                    reinterpreted_batch_ndims=1,
                ),
            )
        else:
            raise ValueError(f"Unknown prior type: {args.prior_type}")
        task = SBIBMTask(name=args.task, p_dist=prior_dist)
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
            open(os.path.join(posterior_path, f"obs_{obs_seed}.json"), "w"),
            indent=4,
        )
        logger.info(
            f"obs_seed: {obs_seed}, theta shape: {theta_o.shape}, x shape: {x_o.shape}"
        )


if __name__ == "__main__":
    main()
