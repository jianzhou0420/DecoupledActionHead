import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import natsort


import os


def get_encoder_pth(run_dir):
    checkpoints_folder = run_dir + '/checkpoints'
    ckpt_files = natsort.natsorted([f for f in os.listdir(checkpoints_folder) if f.endswith('.ckpt')])

    last_epoch_name = ckpt_files[-1]
    last_epoch_path = os.path.join(checkpoints_folder, last_epoch_name)
    last_epoch_ckpt = torch.load(last_epoch_path, map_location='cpu')
    obs_encoder_dict = {}

    for key in last_epoch_ckpt['state_dict'].keys():
        if 'obs_encoder' in key:
            obs_encoder_dict[key] = last_epoch_ckpt['state_dict'][key]
    for key in obs_encoder_dict.keys():
        print(key)
    torch.save(obs_encoder_dict, f'encoder_{last_epoch_name}.pth')


if __name__ == "__main__":
    run_dir = "/media/jian/data/cached_from_sub_machine/runtime/12.54.49_Exp_Normal_1000__D"
    get_encoder_pth(run_dir)
