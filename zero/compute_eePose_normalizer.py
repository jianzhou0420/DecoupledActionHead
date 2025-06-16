import pickle
import numpy as np
import h5py
import os
from natsort import natsorted
from equi_diffpo.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    array_to_stats
)

dataset_dir = "/media/jian/ssd4t/DP/first/data/robomimic/datasets"
all_datasets = natsorted(os.listdir(dataset_dir))


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

    actions_all.append(np.concatenate(this_actions_all, axis=0))

for i, dataset in enumerate(all_datasets):
    this_actions = actions_all[i]
    this_xyz = this_actions[:, 0:3]

    mean = np.mean(this_xyz, axis=0)
    std = np.std(this_xyz, axis=0)

    statistic_dict[dataset] = {
        'mean': mean,
        'std': std,
        'num_actions': this_actions.shape[0]
    }

all_action = np.concatenate(actions_all, axis=0)[:, 0:3]
mean_all = np.mean(all_action, axis=0)
std_all = np.std(all_action, axis=0)
statistic_dict['all'] = {
    'mean': mean_all,
    'std': std_all,
    'num_actions': all_action.shape[0]
}

# Print the final dictionary in a nice, readable format
print("Position statistics for all datasets:")
print("========================================")
print(f"{'Dataset':<20} | {'Mean Position (3D)':<30} | {'Std Deviation (3D)':<30} | {'Num Actions':<15}")
print(f"{'-'*20}-+-{'-'*30}-+-{'-'*30}-+-{'-'*15}")
for dataset, stats in statistic_dict.items():
    # Format mean and std lists to 5 decimal places for better readability
    formatted_mean = [f"{x:.5f}" for x in stats['mean']]
    formatted_std = [f"{x:.5f}" for x in stats['std']]
    print(f"{dataset:<20} | {str(formatted_mean):<30} | {str(formatted_std):<30} | {stats['num_actions']:<15}")


# JP的normalizer是无所谓的，stage1里面用就stage1里面用
# eePose的normalizer其实是xyz的normalizer（rot本身就是-1，1；详见dataset的class怎么写的）
# 所以，真正重要的normalizer就是action的xyz的normalizer。

stat = array_to_stats(all_action)

this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

with open('normalizer.pkl', 'wb') as f:
    pickle.dump(this_normalizer, f)

print(this_normalizer)
