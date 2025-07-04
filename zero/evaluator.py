import argparse
import numpy as np
import copy
import hydra
import os
from termcolor import cprint

from natsort import natsorted
from omegaconf import OmegaConf
from equi_diffpo.env_runner.robomimic_image_runner_tmp import RobomimicImageRunner
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import torch
import wandb


def seed_everything(seed: int):
    """
    Set the seed for reproducibility.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cprint(f"Seed set to {seed}", "yellow")


def resolve_output_dir(output_dir: str):

    # 1. run_name
    run_name = output_dir.split("_")[1:]
    run_name = "_".join(run_name)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    # 2. checkpoints
    checkpoint_all = natsorted(os.listdir(checkpoint_dir))[1:]  # exclude 'laste.ckpt'

    # 3. cfg
    config_path = os.path.join(output_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)

    return cfg, checkpoint_all, run_name


def evaluate_run(seed: int = 42,
                 run_dir: str = "data/outputs/Normal/23.27.09_normal_ACK_1000",
                 results_dir: str = "data/outputs/eval_results",
                 n_envs: int = 28,
                 n_test_vis: int = 6,
                 n_train_vis: int = 3,
                 n_train: int = 6,
                 n_test: int = 50):
    seed_everything(seed)

    cfg, checkpoint_all, run_name = resolve_output_dir(run_dir)

    cfg_env_runner = []
    dataset_path = []
    print(cfg.train_tasks_meta)
    for key, value in cfg.train_tasks_meta.items():
        this_dataset_path = f"data/robomimic/datasets/{key}/{key}_abs_{cfg.dataset_tail}.hdf5"
        this_env_runner_cfg = copy.deepcopy(cfg.task.env_runner)
        this_env_runner_cfg.dataset_path = this_dataset_path
        this_env_runner_cfg.max_steps = value

        OmegaConf.resolve(this_env_runner_cfg)
        dataset_path.append(this_dataset_path)
        cfg_env_runner.append(this_env_runner_cfg)

    eval_result_dir = os.path.join(results_dir, run_name)  # 建议为评估结果创建一个独立的子目录
    media_dir = os.path.join(eval_result_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    cprint(f"Evaluation output will be saved to: {eval_result_dir}", "blue")

    # --- WandB Initialization ---
    wandb_project_name = "Eval"
    wandb_run_name = f"eval_{run_name}"

    wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        mode="online",
        config=OmegaConf.to_container(cfg, resolve=False),
        dir=eval_result_dir,
    )
    cprint("WandB initialized successfully!", "green")
    # --- Environment Runner Execution ---
    for env_cfg in cfg_env_runner:
        # debug
        cprint('debugging code is on', 'red')
        # /debug
        task_name = env_cfg.dataset_path.split("/")[-2]
        env_runner: RobomimicImageRunner = hydra.utils.instantiate(
            config=env_cfg,
            output_dir=eval_result_dir,
            n_envs=n_envs,
            n_test_vis=n_test_vis,
            n_train_vis=n_train_vis,
            n_train=n_train,
            n_test=n_test,
        )

        for ckpt in checkpoint_all:
            policy: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)
            policy.load_state_dict(
                torch.load(os.path.join(run_dir, "checkpoints", ckpt), map_location="cpu")["state_dict"]
            )
            policy.to("cuda" if torch.cuda.is_available() else "cpu")
            policy.eval()
            epoch = int(ckpt.split("=")[-1].split(".")[0])
            print(f"Evaluating policy at epoch {epoch} with checkpoint {ckpt}...")

            evaluation_results = env_runner.run(policy)
            new_results = dict()
            for key, value in evaluation_results.items():
                new_results[f"{task_name}/{key}"] = value

            print("\nEvaluation Results:")
            print(new_results)

            # raise EOFError
            cprint("Logging results to WandB...", "green")
            wandb.log(new_results, step=epoch)  # 直接将字典传递给 wandb.log
            cprint("Results logged to WandB successfully!", "green")

            # --- Finish WandB run ---

        del env_runner

    wandb.finish()
    cprint("WandB run finished.", "green")
