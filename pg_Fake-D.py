import zarr
from equi_diffpo.model.common.rotation_transformer import RotationTransformer
import pickle
import numpy as np
import h5py

import os
import torch
from natsort import natsorted
from codebase.z_utils.Rotation_torch import matrix_to_rotation_6d, euler2mat
from equi_diffpo.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from equi_diffpo.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

from equi_diffpo.dataset.robomimic_replay_image_dataset import _convert_actions
dataset_dir = "/media/jian/ssd4t/DP/first/data/robomimic/datasets"
all_datasets = natsorted(os.listdir(dataset_dir))

if 'ABC' in all_datasets:
    all_datasets.remove('ABC')  # Remove ABC dataset if it exists
statistic_dict = {}

actions_all = []

for dataset in all_datasets:
    dataset_path = os.path.join(dataset_dir, dataset, f"{dataset}_abs_traj_eePose.hdf5")

    print(f"Processing dataset: {dataset_path}")
    this_actions_all = []
    with h5py.File(dataset_path, 'r') as f:
        data = f['data']
        demo_names = natsorted(list(data.keys()))
        print(f"Number of demonstrations: {len(demo_names)}")
        length_counter = 0
        for demo_name in demo_names:
            this_actions_all.append(data[demo_name]['actions'][:])

    actions_all.append(this_actions_all)


root = zarr.group(zarr.MemoryStore)
root.require_group('data', overwrite=True)

pass

# 到这里，获得all_action，包含所有数据集的动作数据
