import hydra
import os
from termcolor import cprint

from natsort import natsorted
from omegaconf import OmegaConf
from equi_diffpo.env_runner.robomimic_image_runner_tmp import RobomimicImageRunner
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import torch


def resolve_output_dir(output_dir: str):

    # 1. run_name
    run_name = output_dir.split("_")[1:]
    checkpoint_dir = os.path.join(output_dir, "checkpoints")

    # 2. checkpoints
    checkpoint_all = natsorted(os.listdir(checkpoint_dir))

    # 3. cfg
    config_path = os.path.join(output_dir, ".hydra", "config.yaml")
    cfg = OmegaConf.load(config_path)

    return cfg, checkpoint_all, run_name


if __name__ == "__main__":
    # 0.some config resolvers
    def get_ws_x_center(task_name):
        if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
            return -0.2
        else:
            return 0.

    def get_ws_y_center(task_name):
        return 0.

    OmegaConf.register_new_resolver("get_max_steps", lambda x: max_steps[x], replace=True)
    OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)

    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    # 1.dataset_path
    output_path = "/media/jian/ssd4t/DP/first/data/outputs/2025.06.10/13.46.04_DecoupleActhonHead_normal_stack_d1_10"
    cfg, checkpoint_all, run_name = resolve_output_dir(output_path)

    max_steps = {
        'stack_d1': 400,
        'stack_three_d1': 400,
        'square_d2': 400,
        'threading_d2': 400,
        'coffee_d2': 400,
        'three_piece_assembly_d2': 500,
        'hammer_cleanup_d1': 500,
        'mug_cleanup_d1': 500,
        'kitchen_d1': 800,
        'nut_assembly_d0': 500,
        'pick_place_d0': 1000,
        'coffee_preparation_d1': 800,
        'tool_hang': 700,
        'can': 400,
        'lift': 400,
        'square': 400,
    }

    policy: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(
        torch.load(os.path.join(output_path, "checkpoints", checkpoint_all[-1]), map_location="cpu")["state_dict"]
    )
    policy.to("cuda" if torch.cuda.is_available() else "cpu")
    policy.eval()
    cprint("Policy loaded successfully!", "green")

    env_runner: RobomimicImageRunner = hydra.utils.instantiate(
        config=cfg.task.env_runner,
        output_dir="./data/outputs",
    )

    test = env_runner.run(policy)
    print(test)
