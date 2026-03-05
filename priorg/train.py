import logging
import os
import pickle
import random
import socket
import time

import hydra
import jax
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from priorg.sim.methods.methods import get_method
from priorg.sim.tasks.task import get_task


def main():
    """Main script to run"""
    score_sbi()


def set_seed(seed: int):
    """This methods just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    with jax.default_device(jax.devices("cpu")[0]):
        key = jax.random.PRNGKey(seed)
    return key


def init_dir(dir_path: str):
    """Initializes a directory for storing models and summary.csv"""
    if not os.path.exists(dir_path + os.sep + "models"):
        os.makedirs(dir_path + os.sep + "models")

    if not os.path.exists(dir_path + os.sep + "summary.csv"):
        df = pd.DataFrame(
            columns=[
                "method",
                "task",
                "num_simulations",
                "seed",
                "model_id",
                "metric",
                "value",
                "time_train",
                "time_eval",
                "cfg",
            ]
        )
        df.to_csv(dir_path + os.sep + "summary.csv", index=False)


@hydra.main(version_base=None, config_path="cfg", config_name="train.yaml")
def score_sbi(cfg: DictConfig):
    """Evaluate score based inference"""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Jax devices: {jax.devices()}")
    log.info(f"Torch devices: {torch.cuda.device_count()}")

    seed = cfg.seed
    key = set_seed(seed)
    log.info(f"Seed: {seed}")

    init_dir(output_dir)

    # Set up the task # TODO: maybe add data_device
    if cfg.model_id is None:
        log.info(f"Task: {cfg.task.name}")
        task = get_task(name=cfg.task.name)
        key, key_data = jax.random.split(key)
        data = task.get_data(cfg.task.num_simulations, key=key_data)
        # Run method
        log.info(f"Running method: {cfg.method.name}")
        method_run = get_method(cfg.method.name)
        key, key_train = jax.random.split(key)
        start_time = time.time()
        model = method_run(task, data, cfg.method, rng=key_train)
        time_train = time.time() - start_time
        log.info(f"Training time: {time_train:.2f} seconds")

    # Evaluate
    # log.info(f"Evaluating method: {cfg.method.name}")
    # metrics_results = {}

    # metric_values, eval_time = eval_negative_log_likelihood(task, model, metric_params, rng_eval)
    # log.info(f"Negative log likelihood values: {metric_values}")
    # log.info(f"Evaluation time: {eval_time:.2f} seconds")

    # if metric_values is not None:
    #     metrics_results['nll'] = metric_values

    is_save_model = cfg.save_model
    if is_save_model and cfg.model_id is None:
        log.info(f"Saving model")
        file_name = os.path.join(output_dir, "model.pkl")
        with open(file_name, "wb") as file:
            pickle.dump(model, file)
        log.info(f"Model saved to {file_name}")


if __name__ == "__main__":
    main()
