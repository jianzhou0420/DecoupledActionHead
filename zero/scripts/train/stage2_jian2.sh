#!/bin/bash

# ---
# Check for provided task names
# ---
if [ -z "$1" ]; then
    echo "Usage: $0 <task_letters>"
    echo "Example: $0 ABCDEFG"
    echo "Example: $0 ABE"
    exit 1
fi

# ---
# Define the mapping of single letters to descriptive task names
# ---
declare -A TASK_MAP
TASK_MAP["A"]="stack_d1"
TASK_MAP["B"]="coffee_d2"
TASK_MAP["C"]="three_piece_assembly_d2"
TASK_MAP["D"]="stack_three_d1"
TASK_MAP["E"]="square_d2"
TASK_MAP["F"]="threading_d2"
TASK_MAP["G"]="hammer_cleanup_d1"
TASK_MAP["H"]="mug_cleanup_d1"
TASK_MAP["I"]="kitchen_d1"
TASK_MAP["J"]="nut_assembly_d0"
TASK_MAP["K"]="pick_place_d0"
TASK_MAP["L"]="coffee_preparation_d1"
# Add more mappings as needed

# ---
# Get the input task letters
# ---
INPUT_TASK_LETTERS="$1"
echo "Received task letters: $INPUT_TASK_LETTERS"
echo "---"

# ---
# Iterate through each letter and run the corresponding task
# ---
for LETTER in $(echo "$INPUT_TASK_LETTERS" | sed -e 's/\(.\)/\1 /g'); do
    DESCRIPTIVE_TASK_NAME=${TASK_MAP["$LETTER"]}

    if [ -z "$DESCRIPTIVE_TASK_NAME" ]; then
        echo "Warning: No descriptive name found for task letter '$LETTER'. Skipping."
        continue # Skip to the next iteration if no mapping is found
    fi

    echo "Running trainer.py for task: '$LETTER' (Descriptive Name: $DESCRIPTIVE_TASK_NAME)"
    python trainer_pl_stage2.py --config-name=DP_DecoupleActhonHead_stage2 n_demo=1000 task_name="$DESCRIPTIVE_TASK_NAME" task.env_runner.n_envs=19
    echo "Finished trainer.py for task: '$LETTER'"
    echo "---"
done

echo "All specified tasks completed!"
