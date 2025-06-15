import numpy as np
import h5py

import os

from natsort import natsorted


dataset_dir = "/media/jian/ssd4t/DP/first/data/robomimic/datasets_abs"
all_datasets = natsorted(os.listdir(dataset_dir))

average_demo_length = {}

for dataset in all_datasets:
    dataset_path = os.path.join(dataset_dir, dataset, f"{dataset}_abs.hdf5")

    print(f"Processing dataset: {dataset_path}")
    with h5py.File(dataset_path, 'r') as f:
        data = f['data']
        demo_names = natsorted(list(data.keys()))
        print(f"Number of demonstrations: {len(demo_names)}")
        length_counter = 0
        for demo_name in demo_names:
            demo = data[demo_name]
            if 'actions' in demo:
                actions = demo['actions']
                if isinstance(actions, np.ndarray):
                    length = actions.shape[0]
                else:
                    length = len(actions)
                length_counter += length
        average_length = length_counter / len(demo_names)
        average_demo_length[dataset] = average_length


print("Average demonstration lengths:")

# Sort the dictionary items by their average length (the value)
# item[1] refers to the value (avg_length) in each (key, value) pair
sorted_demo_lengths = sorted(average_demo_length.items(), key=lambda item: item[1])

# Determine the maximum length for the dataset names for consistent alignment
max_dataset_length = 0
for dataset, _ in sorted_demo_lengths:  # Iterate through the sorted list
    if len(dataset) > max_dataset_length:
        max_dataset_length = len(dataset)


name_2_alphabet = {
    "stack_d1": "A",
    "coffee_d2": "B",
    "three_piece_assembly_d2": "C",
    "stack_three_d1": "D",
    "square_d2": "E",
    "threading_d2": "F",
    "hammer_cleanup_d1": "G",
    "mug_cleanup_d1": "H",
    "kitchen_d1": "I",
    "nut_assembly_d0": "J",
    "pick_place_d0": "K",
    "coffee_preparation_d1": "L"
}


name_2_alphabet = {
    "stack_d1": "A",
    "coffee_d2": "B",
    "three_piece_assembly_d2": "C",
    "stack_three_d1": "D",
    "square_d2": "E",
    "threading_d2": "F",
    "hammer_cleanup_d1": "G",
    "mug_cleanup_d1": "H",
    "kitchen_d1": "I",
    "nut_assembly_d0": "J",
    "pick_place_d0": "K",
    "coffee_preparation_d1": "L"
}

print("Average demonstration lengths (sorted and mapped):")

# Sort the dictionary items by their average length
sorted_demo_lengths = sorted(average_demo_length.items(), key=lambda item: item[1])

# Determine the maximum length for the original dataset names for consistent alignment
max_dataset_name_length = 0
for dataset_name, _ in sorted_demo_lengths:
    if len(dataset_name) > max_dataset_name_length:
        max_dataset_name_length = len(dataset_name)

# Define column widths for neat printing
alphabet_col_width = 1  # For a single letter
spacing_after_alphabet = 3  # Spacing between letter and dataset name
dataset_name_col_width = max_dataset_name_length  # Dynamic width for dataset names
spacing_after_dataset_name = 3  # Spacing between dataset name and average length


# Print header
# Using f-strings to align column headers as well
print(f"{'':<{alphabet_col_width}} {'Dataset':<{dataset_name_col_width}} {'Avg. Length':>11}")
print(f"{'-'*alphabet_col_width} {'-'*dataset_name_col_width} {'-'*11}")


for dataset_name, avg_length in sorted_demo_lengths:
    # Get the corresponding alphabet letter
    alphabet = name_2_alphabet.get(dataset_name, "N/A")  # 'N/A' if not found

    # Print the alphabet, original dataset name, and average length in columns
    print(
        f"{alphabet:<{alphabet_col_width}} "  # Alphabet, left-aligned
        f"{' ': <{spacing_after_alphabet}}"  # Spacing
        f"{dataset_name:<{dataset_name_col_width}} "  # Dataset name, left-aligned
        f"{' ': <{spacing_after_dataset_name}}"  # Spacing
        f"{avg_length:>.2f} timesteps"  # Average length, right-aligned, 2 decimal places
    )
