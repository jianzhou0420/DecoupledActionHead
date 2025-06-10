import hydra
import os
from termcolor import cprint

from natsort import natsorted
from omegaconf import OmegaConf
from equi_diffpo.env_runner.robomimic_image_runner_tmp import RobomimicImageRunner
from equi_diffpo.policy.base_image_policy import BaseImagePolicy
import torch
import wandb


def resolve_output_dir(output_dir: str):

    # 1. run_name
    run_name = output_dir.split("_")[1:]
    run_name = "_".join(run_name)
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
    output_path = "data/outputs/Archive/20.10.57_pretrain_JPee_stage2_coffee_d2_1000"
    cfg, checkpoint_all, run_name = resolve_output_dir(output_path)

    max_steps = {
        'stack_d1': 400,
        'stack_three_d1': 400,
        'square_d2': 400,
        'threading_d2': 400,
        'coffee_d2': 1200,
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
    # from torchview import draw_graph

    # dummy_input_dict = {
    #     'obs': {
    #         "agentview_image": torch.randn(1, 2, 3, 84, 84),  # Batch size 1, 3 channels (RGB), 100x100 image
    #         "robot0_eye_in_hand_image": torch.randn(1, 2, 3, 84, 84),  # Batch size 1, 3 channels (RGB), 100x100 image
    #         "robot0_eef_pos": torch.randn(1, 2, 3),  # Batch size 1, 3 features for end-effector position
    #         "robot0_eef_quat": torch.randn(1, 2, 4),  # Batch size 1, 4 features for end-effector quaternion
    #         "robot0_gripper_qpos": torch.randn(1, 2, 2),  # Batch size 1, 2 features for gripper position
    #     },
    #     'action': torch.randn(1, 16, 10),  # Batch size 1, 8 features for action
    # }
    # graph = draw_graph(
    #     policy,
    #     input_data=dummy_input_dict,
    #     expand_nested=True,
    #     # device="meta",
    #     save_graph=True,           # write out a .gv file
    #     filename="mymodel-graph"   # without extension; .gv and .png will be created
    # )

    # # 4) Render or display
    # #    If you want to render to PNG right away:
    # graph.visual_graph.render(filename="mymodel-graph", format="png", cleanup=True)

    # # 5) In a Jupyter notebook you can just do
    # graph.visual_graph
    cprint("Policy loaded successfully!", "green")

    output_dir = os.path.join("./data/outputs", run_name, "eval_results")  # 建议为评估结果创建一个独立的子目录
    media_dir = os.path.join(output_dir, "media")
    os.makedirs(media_dir, exist_ok=True)
    cprint(f"Evaluation output will be saved to: {output_dir}", "blue")

    # --- WandB Initialization ---
    # Customize these parameters for your WandB run
    wandb_project_name = "Eval"  # 你的WandB项目名称
    wandb_run_name = f"eval_{run_name}"  # 每次评估的唯一名称，包含源output_path信息

    wandb.init(
        project=wandb_project_name,
        name=wandb_run_name,
        config=OmegaConf.to_container(cfg, resolve=False),  # 将Hydra配置作为WandB配置记录
        dir=output_dir,  # 将wandb的日志文件也保存到这个目录下
    )
    cprint("WandB initialized successfully!", "green")

    env_runner: RobomimicImageRunner = hydra.utils.instantiate(
        config=cfg.task.env_runner,
        output_dir=output_dir,  # 确保这里的output_dir是为当前评估设置的
        n_envs=28,
        n_test_vis=50,  # 确保这个值足够大，以录制你想要的测试视频数量
        n_train_vis=6,
    )

    # --- Run evaluation and log to WandB ---
    cprint("Starting environment runner...", "yellow")
    # env_runner.run(policy) 返回的 log_data 字典包含了 wandb.Video 对象和奖励数据
    evaluation_results = env_runner.run(policy)
    cprint("Environment runner finished.", "yellow")

    print("\nEvaluation Results:")
    print(evaluation_results)

    # Log results to WandB
    cprint("Logging results to WandB...", "green")
    wandb.log(evaluation_results)  # 直接将字典传递给 wandb.log
    cprint("Results logged to WandB successfully!", "green")

    # --- Finish WandB run ---
    wandb.finish()
    cprint("WandB run finished.", "green")
