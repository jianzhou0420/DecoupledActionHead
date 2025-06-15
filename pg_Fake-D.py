import numpy as np
import h5py

import os

from natsort import natsorted


dataset_dir = "/media/jian/ssd4t/DP/first/data/robomimic/datasets"
all_datasets = natsorted(os.listdir(dataset_dir))

average_demo_length = {}

actions_all = []

for dataset in all_datasets:
    dataset_path = os.path.join(dataset_dir, dataset, f"{dataset}_abs_traj_eePose.hdf5")

    print(f"Processing dataset: {dataset_path}")
    with h5py.File(dataset_path, 'r') as f:
        data = f['data']
        demo_names = natsorted(list(data.keys()))
        print(f"Number of demonstrations: {len(demo_names)}")
        length_counter = 0
        for demo_name in demo_names:
            print(f"Processing demonstration: {demo_name}")
            actions_all.extend(data[demo_name]['actions'][:])

print(f"Total number of actions across all demonstrations: {len(actions_all)}")
actions_all = np.array(actions_all)
print(f"Shape of actions_all: {actions_all.shape}")
