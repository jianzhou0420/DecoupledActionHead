#!/bin/bash

# Define an array of dataset names
datasets=("stack_d1" "coffee_d2" "three_piece_assembly_d2" "stack_three_d1" "square_d2" "threading_d2" "hammer_cleanup_d1" "mug_cleanup_d1" "kitchen_d1" "nut_assembly_d0" "pick_place_d0" "coffee_preparation_d1")

# Loop through each dataset and run the python script
for dataset in "${datasets[@]}"; do
    echo "Processing dataset: $dataset"
    python equi_diffpo/scripts/robomimic_dataset_conversion.py -i "data/robomimic/datasets_abs/${dataset}/${dataset}.hdf5" -o "data/robomimic/datasets_abs/${dataset}/${dataset}_abs.hdf5" -n 18
    echo "Finished processing dataset: $dataset"
    echo "---------------------------------"
done

echo "All tasks completed."
