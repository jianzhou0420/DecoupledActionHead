# helper package
# try:
#     import warnings
#     warnings.filterwarnings("ignore", message="Gimbal lock detected. Setting third angle to zero")
#     warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_bwd.*")
#     warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.cuda.amp.custom_fwd.*")
# except:
#     pass

from zero.evaluator import evaluate_run
import wandb
import pathlib
import gym  # Or your custom environment library
import time
from datetime import datetime
import os
import os
from typing import Type, Dict, Any
import copy

# framework package
import argparse
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
import pytorch_lightning as pl
from torch.utils.data import Dataset
# equidiff package
from equi_diffpo.workspace.base_workspace import BaseWorkspace
from equi_diffpo.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from equi_diffpo.dataset.base_dataset import BaseImageDataset
from equi_diffpo.env_runner.base_image_runner import BaseImageRunner
from equi_diffpo.common.checkpoint_util import TopKCheckpointManager
from equi_diffpo.common.json_logger import JsonLogger
from equi_diffpo.common.pytorch_util import dict_apply, optimizer_to
from equi_diffpo.model.diffusion.ema_model import EMAModel
from equi_diffpo.model.common.lr_scheduler import get_scheduler
# Hydra specific imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from equi_diffpo.config.config_hint import AppConfig

from equi_diffpo.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
import sys
from termcolor import cprint
import mimicgen
from natsort import natsorted
from zero.z_utils.scp_utils import scp_to_another_computer

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
os.environ['HYDRA_FULL_ERROR'] = "1"

torch.set_float32_matmul_precision('medium')

# ---------------------------------------------------------------
# region 1. Trainer


def load_pretrained_weights(model, ckpt_path):
    """
    加载预训练权重，并根据策略冻结参数。

    此函数执行以下操作：
    1. 记录模型中初始状态就为冻结的参数。
    2. 从指定的 `ckpt_path` 加载权重。
    3. 将权重加载到模型中匹配的层。
    4. 强制冻结模型中所有的 'clip' 子模块参数，并将其设置为评估模式。
    5. 冻结所有其他从检查点成功加载的参数。
    6. 确保初始状态为冻结的参数保持冻结。
    7. 保持模型中其余参数（如新的分类头）为可训练状态。

    Args:
        model (nn.Module): 需要加载权重和设置梯度的模型。
        ckpt_path (str): 预训练权重文件（.pth）的路径。

    Returns:
        nn.Module: 处理完成后的模型。
    """
    # --------------------------------------------------------------------------
    # ✨ 新增步骤：记录初始冻结状态
    # --------------------------------------------------------------------------
    initially_frozen_keys = {name for name, param in model.named_parameters() if not param.requires_grad}
    if initially_frozen_keys:
        print(f"检测到 {len(initially_frozen_keys)} 个参数在函数调用前已被设置为冻结状态。这些参数将保持冻结。")
        # for name in initially_frozen_keys:
        #     print(f"  - 初始冻结: {name}")

    if not ckpt_path:
        print("未提供权重路径，跳过权重加载过程。")
        # 即使不加载权重，我们仍然需要冻结CLIP和保持初始冻结状态
        if hasattr(model, 'clip'):
            print("正在冻结 CLIP 模块并设置为评估模式...")
            model.clip.eval()
            for name, param in model.clip.named_parameters():
                param.requires_grad = False
                print(f"🧊 [强制冻结] {name}")
        return model

    # --------------------------------------------------------------------------
    # 第1步：识别出可以安全加载的权重 (逻辑不变)
    # --------------------------------------------------------------------------
    print(f"正在从 '{ckpt_path}' 加载权重...")
    pretrained_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

    new_model_dict = model.state_dict()
    loadable_keys = set()
    filtered_dict = {}

    print("正在筛选兼容的权重...")
    for k, v in pretrained_dict.items():
        if k in new_model_dict and new_model_dict[k].shape == v.shape:
            filtered_dict[k] = v
            loadable_keys.add(k)
    print(f"识别出 {len(loadable_keys)} 个参数可以从 checkpoint 安全加载。")

    # --------------------------------------------------------------------------
    # 第2步：加载筛选后的权重 (逻辑不变)
    # --------------------------------------------------------------------------
    new_model_dict.update(filtered_dict)
    model.load_state_dict(new_model_dict)
    print("已成功加载所有兼容的权重。")

    # --------------------------------------------------------------------------
    # 第3步：强制设置 CLIP 模块为评估模式 (逻辑不变)
    # --------------------------------------------------------------------------
    if hasattr(model, 'clip'):
        print("正在将 CLIP 模块设置为评估模式 (model.clip.eval())...")
        model.clip.eval()
    else:
        print("⚠️  警告: 模型中未找到名为 'clip' 的属性，无法设置为评估模式。")

    # --------------------------------------------------------------------------
    # 第4步（已修改）：根据加载情况、模块名称和初始状态智能地设置梯度
    # --------------------------------------------------------------------------
    print("正在智能地设置参数的梯度计算状态...")
    trainable_params = 0
    frozen_params = 0

    for name, param in model.named_parameters():
        # 检查参数是否在初始时就已冻结
        is_initially_frozen = name in initially_frozen_keys
        # 检查参数是否属于CLIP模块
        is_clip_param = name.startswith('clip.')
        # 检查参数是否从checkpoint加载
        is_loaded_from_ckpt = name in loadable_keys

        # 智能逻辑：检查偏置对应的权重是否也被加载了
        if name.endswith('.bias') and not is_clip_param and not is_initially_frozen:
            weight_name = name.replace('.bias', '.weight')
            if weight_name not in loadable_keys:
                is_loaded_from_ckpt = False
                print(f"ℹ️  注意: 偏置 '{name}' 将保持可训练，因为其对应的权重 '{weight_name}' 未被加载。")

        # 最终冻结决策：只要是初始冻结、CLIP参数或从checkpoint加载的参数，就冻结
        if is_initially_frozen or is_clip_param or is_loaded_from_ckpt:
            param.requires_grad = False
            frozen_params += 1
        else:
            param.requires_grad = True
            trainable_params += 1

    print(f"策略执行完毕：{frozen_params} 个参数被冻结，{trainable_params} 个参数保持可训练。")

    # --------------------------------------------------------------------------
    # 第5步：最终验证 (已修改)
    # --------------------------------------------------------------------------
    print("\n--- 最终模型梯度状态验证 ---")
    for name, param in model.named_parameters():
        status = "🧊 [已冻结]" if not param.requires_grad else "✅ [可训练]"
        reason = ""
        if not param.requires_grad:
            if name in initially_frozen_keys:
                reason = "(原因: 初始状态为冻结)"
            elif name.startswith('clip.'):
                reason = "(原因: CLIP模块)"
            elif name in loadable_keys:
                reason = "(原因: 从ckpt加载)"
        print(f"{status} {name} {reason}")
    print("---------------------------------")

    return model


class Trainer_all(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        task_type = cfg.train_mode

        if task_type == 'stage2' or task_type == 'stage2_rollout':
            ckpt_path = cfg.ckpt_path
            policy: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
            policy = load_pretrained_weights(policy, ckpt_path)
            policy_ema = copy.deepcopy(policy)
        elif task_type == 'stage1':
            policy: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
            policy_ema: DiffusionUnetHybridImagePolicy = copy.deepcopy(policy)
        elif task_type == 'normal':
            policy: DiffusionUnetHybridImagePolicy = hydra.utils.instantiate(cfg.policy)
            policy_ema: DiffusionUnetHybridImagePolicy = copy.deepcopy(policy)
        else:
            raise ValueError(f"Unsupported task type: {task_type}, check config.train_mode")

        if cfg.training.use_ema:
            ema_handler: EMAModel = hydra.utils.instantiate(
                cfg.ema,
                model=policy_ema,)

        self.policy = policy.to(self.device)
        self.policy_ema = policy_ema.to(self.device)
        self.ema_handler = ema_handler
        self.train_sampling_batch = None

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.normalizer = self.trainer.datamodule.normalizer
            self.policy.set_normalizer(self.normalizer)
            self.policy_ema.set_normalizer(self.normalizer) if self.cfg.training.use_ema else None

        return

    def training_step(self, batch):
        # model = self.policy
        # print("\n--- 最终模型梯度状态验证 ---")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"✅ [可训练] {name}")
        #     else:
        #         print(f"🧊 [已冻结] {name}")
        # print("---------------------------------")

        if self.train_sampling_batch is None:
            self.train_sampling_batch = batch

        loss = self.policy.compute_loss(batch)
        self.logger.experiment.log({
            'train/train_loss': loss.item(),
            'train/lr': self.optimizers().param_groups[0]['lr'],
            'trainer/global_step': self.global_step,
            'trainer/epoch': self.current_epoch,
        }, step=self.global_step)
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """
        This hook is called after the training step and optimizer update.
        It's the perfect place to update the EMA weights.
        """
        self.ema_handler.step(self.policy)

    def validation_step(self, batch):
        loss = self.policy_ema.compute_loss(batch)
        self.logger.experiment.log({
            'train/val_loss': loss.item(),
        }, step=self.global_step)
        return loss

    def configure_optimizers(self):
        cfg = self.cfg
        num_training_steps = self.trainer.estimated_stepping_batches

        optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.policy.parameters())
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=int((
                num_training_steps)
                // cfg.training.gradient_accumulate_every),
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",  # Make sure to step the scheduler every batch/step
                "frequency": 1,
            },
        }

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        This hook is called when a checkpoint is saved.
        We replace the full state_dict with ONLY the state_dict of the policy
        we want to save (either the training policy or the EMA one).
        """
        if self.cfg.training.use_ema:
            # Get the state_dict from your EMA model
            policy_state_to_save = self.policy_ema.state_dict()
        else:
            # Get the state_dict from the standard training model
            policy_state_to_save = self.policy.state_dict()

        # Overwrite the complete state_dict with only the policy's state
        checkpoint['state_dict'] = policy_state_to_save
# endregion
# ---------------------------------------------------------------


# ---------------------------------------------------------------
# region 2. DataModule
class MyDataModule(pl.LightningDataModule):
    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        if stage == 'fit':
            cfg = self.cfg
            dataset: BaseImageDataset
            dataset = hydra.utils.instantiate(cfg.task.dataset)
            val_dataset = dataset.get_validation_dataset()

            assert isinstance(dataset, BaseImageDataset)
            normalizer = dataset.get_normalizer()

            self.normalizer = normalizer
            self.dataset = dataset
            self.val_dataset = val_dataset

    def train_dataloader(self):
        train_dataloader = DataLoader(self.dataset, **self.cfg.dataloader)
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(self.val_dataset, **self.cfg.val_dataloader)
        return val_dataloader


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region 3. Callback


class RolloutCallback(pl.Callback):
    """
    A Callback to run a policy rollout in an environment periodically.
    """

    def __init__(self, env_runner_cfg: DictConfig, rollout_every_n_epochs: int = 1):
        super().__init__()
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            env_runner_cfg,
            output_dir='data/outputs'
        )  # TODO:fix it

        assert isinstance(env_runner, BaseImageRunner)
        self.rollout_every_n_epochs = rollout_every_n_epochs
        self.env_runner = env_runner

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Trainer_all):
        """
        This hook is called after every validation epoch.
        """

        # Ensure we only run this every N epochs
        if (trainer.current_epoch + 1) % self.rollout_every_n_epochs != 0:
            return
        if pl_module.global_step <= 0:
            return
        runner_log = self.env_runner.run(pl_module.policy_ema)
        trainer.logger.experiment.log(runner_log, step=trainer.global_step)
        # cprint(f"Rollout completed at epoch {trainer.current_epoch}, step {trainer.global_step}.", "green", attrs=['bold'])
        # cprint(f"Rollout log: {runner_log}", "blue", attrs=['bold'])

# region ActionMseLossForDiffusion


class ActionMseLossForDiffusion(pl.Callback):
    """
    A Callback to compute the MSE loss of actions in the diffusion model.
    This is useful for training the diffusion model with action data.
    """

    def __init__(self, cfg: AppConfig):
        super().__init__()
        self.cfg = cfg

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: Trainer_all):
        """
        This hook is called after every validation epoch.
        """
        if pl_module.global_step <= 0:
            return
        train_sampling_batch = pl_module.train_sampling_batch

        batch = dict_apply(train_sampling_batch, lambda x: x.to(pl_module.device, non_blocking=True))
        obs_dict = batch['obs']
        gt_action = batch['action']
        result = pl_module.policy_ema.predict_action(obs_dict)
        pred_action = result['action_pred']
        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
        trainer.logger.experiment.log({
            'train/action_mse_loss': mse,
        }, step=trainer.global_step)


# endregion
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# region Main


def train(cfg: AppConfig):

    # 0. extra config processing

    cfg_env_runner = []
    dataset_path = []
    for key, value in cfg.train_tasks_meta.items():
        this_dataset_path = f"data/robomimic/datasets/{key}/{key}_abs_{cfg.dataset_tail}.hdf5"
        this_env_runner_cfg = copy.deepcopy(cfg.task.env_runner)
        this_env_runner_cfg.dataset_path = this_dataset_path
        this_env_runner_cfg.max_steps = value

        OmegaConf.resolve(this_env_runner_cfg)
        dataset_path.append(this_dataset_path)
        cfg_env_runner.append(this_env_runner_cfg)

    cfg.task.dataset.dataset_path = OmegaConf.create(dataset_path)

    OmegaConf.save(cfg, os.path.join(cfg.run_dir, 'config.yaml'))
    # 1. Define a unique name and directory for this specific run

    this_run_dir = cfg.run_dir
    run_name = cfg.run_name

    os.makedirs(os.path.join(this_run_dir, 'wandb'), exist_ok=True)  # Ensure the output directory exists

    ckpt_path = os.path.join(this_run_dir, 'checkpoints')
    # 2. Configure ModelCheckpoint to save in that specific directory

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename=f'{run_name}' + '_{epoch:03d}',
        every_n_epochs=cfg.training.checkpoint_every,
        save_top_k=-1,
        save_last=False,
        save_weights_only=True,
        save_on_train_epoch_end=True
    )

    # 3. Configure WandbLogger to use the same directory
    wandb_logger = WandbLogger(
        save_dir=this_run_dir,  # <-- Use save_dir to point to the same path
        config=OmegaConf.to_container(cfg, resolve=True),
        **cfg.logging,
    )

    #

    if cfg.train_mode == 'stage1':
        callback_list = [checkpoint_callback,
                         ActionMseLossForDiffusion(cfg),
                         ]
    elif cfg.train_mode == 'stage2':
        callback_list = [checkpoint_callback,
                         ActionMseLossForDiffusion(cfg),
                         ]

    elif cfg.train_mode == 'normal':
        callback_list = [checkpoint_callback,
                         ActionMseLossForDiffusion(cfg),
                         ]
    elif cfg.train_mode == 'stage2_rollout':
        rollout_callback_list = [RolloutCallback(cfg_env_runner[i], rollout_every_n_epochs=cfg.training.rollout_every) for i in range(len(cfg_env_runner))]
        callback_list = [checkpoint_callback,
                         ActionMseLossForDiffusion(cfg),
                         ]
        callback_list.extend(rollout_callback_list)
    else:
        raise ValueError(f"Unsupported task type: {cfg.train_mode}, check config.name")

    trainer = pl.Trainer(callbacks=callback_list,
                         max_epochs=int(cfg.training.num_epochs),
                         devices='auto',
                         strategy='ddp_find_unused_parameters_true',
                         logger=[wandb_logger],
                         use_distributed_sampler=False,
                         check_val_every_n_epoch=cfg.training.val_every,
                         )
    trainer_model = Trainer_all(cfg)
    data_module = MyDataModule(cfg)
    trainer.fit(trainer_model, datamodule=data_module)

    # if cfg.train_mode == 'stage2':
    #     scp_to_another_computer(
    #         local_path=this_run_dir,
    #         remote_path=os.path.join('/media/jian/ssd4t/tmp', run_name),
    #         hostname='10.12.65.19',
    #         username='jian',
    #     )
    #     wandb.finish()
    #     evaluate_run(
    #         seed=42,
    #         run_dir=this_run_dir,)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'equi_diffpo', 'config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.

    OmegaConf.resolve(cfg)
    train(cfg)


# endregion
# ---------------------------------------------------------------
if __name__ == '__main__':
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

    max_steps = {meta['name']: int(meta['average_steps'] * 2.5) for task, meta in tasks_meta.items()}
    print(f"max_steps: {max_steps}")

    def get_ws_x_center(task_name):
        if task_name.startswith('kitchen_') or task_name.startswith('hammer_cleanup_'):
            return -0.2
        else:
            return 0.

    def get_ws_y_center(task_name):
        return 0.

    def get_train_tasks_meta(task_alphabet):
        task_alphabet_list = natsorted(task_alphabet)
        train_tasks_meta = dict()
        for task_alphabet in task_alphabet_list:
            task_name = tasks_meta[task_alphabet]['name']
            task_max_steps = max_steps[task_name]
            train_tasks_meta.update({task_name: task_max_steps})
        train_tasks_meta = OmegaConf.create(train_tasks_meta)
        return train_tasks_meta

    OmegaConf.register_new_resolver("get_train_tasks_meta", get_train_tasks_meta, replace=True)
    OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)

    # allows arbitrary python code execution in configs using the ${eval:''} resolver
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    main()
