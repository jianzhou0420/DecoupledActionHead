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
from config_hint import AppConfig

from equi_diffpo.workspace.base_workspace import BaseWorkspace
import pathlib
from omegaconf import OmegaConf
import hydra
import sys
from termcolor import cprint


def resolve_output_dir():
    pass
