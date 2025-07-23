import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import natsort

run_dir = "/media/jian/data/cached_from_sub_machine/runtime/12.55.17_Exp_Normal_1000__A"
ckpt_path = "/media/jian/data/cached_from_sub_machine/runtime/12.55.17_Exp_Normal_1000__A/checkpoints/Exp_Normal_1000__A_epoch=049.ckpt"


test_ckpt = torch.load(ckpt_path, map_location='cpu')
print(test_ckpt.keys())
obs_encoder_dict = {}

for key in test_ckpt['state_dict'].keys():
    if 'obs_encoder' in key:
        obs_encoder_dict[key] = test_ckpt['state_dict'][key]

for key in obs_encoder_dict.keys():
    print(key)

torch.save(obs_encoder_dict, run_dir + '/obs_encoder_dict.pth')
