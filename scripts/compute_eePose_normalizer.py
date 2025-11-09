# Import necessary libraries
import os
import pickle
import numpy as np
import h5py
import torch
import argparse
from natsort import natsorted
from jiandecouple.z_utils.JianRotationTorch import matrix_to_rotation_6d, euler2mat
from jiandecouple.model.common.rotation_transformer import RotationTransformer
from jiandecouple.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from jiandecouple.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
from jiandecouple.dataset.robomimic_replay_image_dataset import _convert_actions


def main(task_codes=None, position_only=True):
    """
    Processes specified Robomimic datasets and generates a normalizer.

    Args:
        task_codes (str, optional): A string of task codes to process (e.g., "ABC").
                                    If None, all datasets in the directory will be processed.
    """
    # Define the mapping from task codes to actual task names
    TASK_MAP = {
        "A": "stack_d1",
        "B": "square_d2",
        "C": "coffee_d2",
        "D": "threading_d2",
        "E": "stack_three_d1",
        "F": "hammer_cleanup_d1",
        "G": "three_piece_assembly_d2",
        "H": "mug_cleanup_d1",
        "I": "nut_assembly_d0",
        "J": "kitchen_d1",
        "K": "pick_place_d0",
        "L": "coffee_preparation_d1",
    }

    # Define the dataset directory
    dataset_dir = "data/robomimic/datasets"

    # Determine the list of datasets to process based on the input argument
    if task_codes:
        datasets_to_process = []
        for char in task_codes:
            if char in TASK_MAP:
                datasets_to_process.append(TASK_MAP[char])
            else:
                print(f"Warning: Task code '{char}' not found in TASK_MAP.")
        all_datasets = natsorted(datasets_to_process)
    else:
        # If no task codes are provided, process all datasets in the directory
        all_datasets = natsorted(os.listdir(dataset_dir))
        # Remove 'ABC' dataset if it exists, as specified in earlier logic

    # Dictionary to store statistics for each dataset
    statistic_dict = {}

    # List to store all actions data
    actions_all = []

    # Loop through each dataset
    for dataset in all_datasets:
        # Construct the full path to the HDF5 file
        dataset_path = os.path.join(dataset_dir, dataset, f"{dataset}_abs_traj_eePose.hdf5")

        # Check if the file exists before processing
        if not os.path.exists(dataset_path):
            print(f"Skipping dataset: {dataset_path} as it does not exist.")
            continue

        # Print a message indicating which dataset is being processed
        print(f"Processing dataset: {dataset_path}")

        # List to store actions for the current dataset
        this_actions_all = []

        # Read the HDF5 file
        try:
            with h5py.File(dataset_path, 'r') as f:
                data = f['data']
                demo_names = natsorted(list(data.keys()))
                print(f"Number of demonstrations: {len(demo_names)}")

                # Iterate through each demonstration and extract actions
                for demo_name in demo_names:
                    this_actions_all.append(data[demo_name]['actions'][:])

            # Concatenate actions for the current dataset and add to the main list
            actions_all.append(np.concatenate(this_actions_all, axis=0))
        except Exception as e:
            print(f"Error processing {dataset_path}: {e}")
            continue

    # Check if any datasets were successfully processed
    if not actions_all:
        print("No datasets were processed. Please check your data directory and the task codes you provided.")
    else:
        # Loop through the processed datasets to calculate position statistics
        for i, dataset in enumerate(all_datasets):
            # Skip if a dataset was not processed due to an error
            if i >= len(actions_all):
                continue

            this_actions = actions_all[i]
            # Extract the xyz position data (first 3 columns)
            this_xyz = this_actions[:, 0:3]

            # Calculate mean and standard deviation
            mean = np.mean(this_xyz, axis=0)
            std = np.std(this_xyz, axis=0)

            # Store the statistics in the dictionary
            statistic_dict[dataset] = {
                'mean': mean,
                'std': std,
                'num_actions': this_actions.shape[0]
            }

        # Concatenate all actions from all datasets for overall statistics
        all_action = np.concatenate(actions_all, axis=0)
        mean_all = np.mean(all_action, axis=0)
        std_all = np.std(all_action, axis=0)
        statistic_dict['all'] = {
            'mean': mean_all,
            'std': std_all,
            'num_actions': all_action.shape[0]
        }

        # Initialize RotationTransformer to convert rotation representations
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')

        # Convert the rotation part of the actions
        all_action = _convert_actions(all_action, True, rotation_transformer)

        # Print the final statistics table
        print("Position statistics for all processed datasets:")
        print("========================================")
        print(f"{'Dataset':<20} | {'Mean Position (3D)':<30} | {'Std Deviation (3D)':<30} | {'Num Actions':<15}")
        print(f"{'-'*20}-+-{'-'*30}-+-{'-'*30}-+-{'-'*15}")
        for dataset, stats in statistic_dict.items():
            # Format mean and std lists to 5 decimal places for readability
            formatted_mean = [f"{x:.5f}" for x in stats['mean']]
            formatted_std = [f"{x:.5f}" for x in stats['std']]
            print(f"{dataset:<20} | {str(formatted_mean):<30} | {str(formatted_std):<30} | {stats['num_actions']:<15}")

        # --- Create and save the normalizer ---

        stat = array_to_stats(all_action[:, :3]) if position_only else array_to_stats(all_action)
        this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)

        # Save the normalizer object to a pickle file
        if position_only:
            save_name = f'normalizer_Pos_{task_codes}.pkl'
        else:
            save_name = f'normalizer_Action_{task_codes}.pkl'

        with open(save_name, 'wb') as f:
            pickle.dump(this_normalizer, f)

        # Print the generated normalizer object
        print("\nGenerated Normalizer:")
        print(this_normalizer)


if __name__ == "__main__":
    # --- Command-line argument parser ---
    parser = argparse.ArgumentParser(description="Compute end-effector pose normalizer for specific or all tasks.")
    parser.add_argument("--task", type=str, default='A', help="A string of task codes to process (e.g., 'ABC')")
    parser.add_argument("--p", action='store_true', help="If set, only normalize position data.")
    args = parser.parse_args()

    # Call the main function with the provided task codes
    # tasks = natsorted(args.task) if args.task else None
    main(task_codes=args.task, position_only=args.p)
