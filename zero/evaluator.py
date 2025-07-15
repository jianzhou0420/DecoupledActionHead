import time
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
import mimicgen


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
    tasks_meta = {
        "A": {"name": "stack_d1", "average_steps": 108, },
        "B": {"name": "square_d2", "average_steps": 153, },
        "C": {"name": "coffee_d2", "average_steps": 224, },
        "D": {"name": "threading_d2", "average_steps": 227, },
        "E": {"name": "stack_three_d1", "average_steps": 255, },
        "F": {"name": "hammer_cleanup_d1", "average_steps": 286, },
        "G": {"name": "three_piece_assembly_d2", "average_steps": 335, },
        "H": {"name": "mug_cleanup_d1", "average_steps": 338, },
        "I": {"name": "nut_assembly_d0", "average_steps": 358, },
        "J": {"name": "kitchen_d1", "average_steps": 619, },
        "K": {"name": "pick_place_d0", "average_steps": 677, },
        "L": {"name": "coffee_preparation_d1", "average_steps": 687, },
    }
    # 1. run_name
    run_name = output_dir.split("_")[1:]
    run_name = "_".join(run_name)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    # 2. checkpoints
    checkpoint_all = natsorted([ckpt for ckpt in os.listdir(checkpoint_dir) if ckpt != "last.ckpt"])

    # 3. cfg
    config_path = os.path.join(output_dir, "config.yaml")
    cfg = OmegaConf.load(config_path)
    task_alphabet_list = natsorted(cfg.task_alphabet)
    task_name_list = {key: tasks_meta[key]["name"] for key in task_alphabet_list}

    return cfg, checkpoint_all, task_alphabet_list, task_name_list


def evaluate_run(seed: int = 42,
                 run_dir: str = "data/outputs/Normal/23.27.09_normal_ACK_1000",
                 results_dir: str = "data/outputs/eval_results",
                 n_envs: int = 28,
                 n_test_vis: int = 6,
                 n_train_vis: int = 3,
                 n_train: int = 6,
                 n_test: int = 50,
                 wandb_mode: str = "offline"):
    seed_everything(seed)

    cfg, checkpoint_all, task_alphabet_list, task_name_list = resolve_output_dir(run_dir)

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

    date_time = time.strftime("%y.%d.%m_%H.%M.%S", time.localtime())
    eval_result_dir = os.path.join(results_dir, date_time)
    media_dir = os.path.join(eval_result_dir, "media")
    os.makedirs(media_dir, exist_ok=True)

    cprint(f"Evaluation output will be saved to: {eval_result_dir}", "blue")

    cprint("WandB initialized successfully!", "green")
    # --- Environment Runner Execution ---
    for i, env_cfg in enumerate(cfg_env_runner):
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
        # --- WandB Initialization ---
        wandb_project_name = "Eval"
        wandb_run_name = f"{cfg.train_mode}_{task_alphabet_list[i]}_{task_name_list[task_alphabet_list[i]]}"

        wandb.init(
            project=wandb_project_name,
            name=wandb_run_name,
            mode=wandb_mode,
            config=OmegaConf.to_container(cfg, resolve=False),
            dir=eval_result_dir,
            group=cfg.run_name,
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
            evaluation_results['epoch'] = epoch
            # raise EOFError
            cprint("Logging results to WandB...", "green")

            wandb.log(evaluation_results, step=epoch)
            cprint("Results logged to WandB successfully!", "green")

            # --- Finish WandB run ---

        del env_runner

        wandb.finish()
    cprint("WandB run finished.", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a policy on Robomimic tasks.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--results_dir", type=str, default="data/outputs/eval_results",)
    parser.add_argument("--run_dir", type=str, default="/media/jian/data/outputs/Normal/23.27.09_normal_ACK_1000",)
    parser.add_argument("--n_envs", type=int, default=28, help="Number of environments to evaluate.")
    parser.add_argument("--n_test_vis", type=int, default=6, help="Number of test environments to visualize.")
    parser.add_argument("--n_train_vis", type=int, default=3, help="Number of training environments to visualize.")
    parser.add_argument("--n_train", type=int, default=6, help="Number of training episodes to run.")
    parser.add_argument("--n_test", type=int, default=50, help="Number of test episodes to run.")
    parser.add_argument("--wandb_mode", type=str, default="offline", help="WandB mode for logging.")
    args = parser.parse_args()

    evaluate_run(seed=args.seed,
                 run_dir=args.run_dir,
                 results_dir=args.results_dir,
                 n_envs=args.n_envs,
                 n_test_vis=args.n_test_vis,
                 n_train_vis=args.n_train_vis,
                 n_train=args.n_train,
                 n_test=args.n_test,
                 wandb_mode=args.wandb_mode
                 )
