from trainer_pl_all import Trainer_all
from omegaconf import OmegaConf
import yaml
import torch
import hydra
from jiandecouple.policy.base_image_policy import BaseImagePolicy
import os


config_path = "/media/jian/ssd4t/workspace_DP/first/equi_diffpo/config/dummy/config_stage1.yaml"
ckpt_path = "/media/jian/ssd4t/workspace_DP/first/data/robomimic/Stage1/tmp/uninitialized.ckpt"
cfg = OmegaConf.load(config_path)

# save
policy = hydra.utils.instantiate(cfg.policy)
torch.save({'state_dict': policy.state_dict()}, ckpt_path)

# load
policy: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)
print(f"Policy type: {type(policy)}")
policy.load_state_dict(
    torch.load(ckpt_path, map_location="cpu")['state_dict']
)
policy.to("cuda" if torch.cuda.is_available() else "cpu")
policy.eval()
