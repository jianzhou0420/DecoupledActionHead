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
TASK_MAP["B"]="square_d2"
TASK_MAP["C"]="coffee_d2"
TASK_MAP["D"]="threading_d2"
TASK_MAP["E"]="stack_three_d1"
TASK_MAP["F"]="hammer_cleanup_d1"
TASK_MAP["G"]="three_piece_assembly_d2"
TASK_MAP["H"]="mug_cleanup_d1"
TASK_MAP["I"]="nut_assembly_d0"
TASK_MAP["J"]="kitchen_d1"
TASK_MAP["K"]="pick_place_d0"
TASK_MAP["L"]="coffee_preparation_d1"
# Add more mappings as needed

# ---
# Get the input task letters
# ---
INPUT_TASK_LETTERS="$1"
echo "Received task letters: $INPUT_TASK_LETTERS"
echo "---"

date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')

# build your run_dir

# ---
# Iterate through each letter and run the corresponding task
# ---
for LETTER in $(echo "$INPUT_TASK_LETTERS" | sed -e 's/\(.\)/\1 /g'); do
    DESCRIPTIVE_TASK_NAME=${TASK_MAP["$LETTER"]}

    run_dir="data/outputs/${date_part}/${time_part}_${DESCRIPTIVE_TASK_NAME}_${LETTER}"

    ckpt_path="data/tmp/stage1_${LETTER}_1000_epoch\=049.ckpt"

    if [ -z "$DESCRIPTIVE_TASK_NAME" ]; then
        echo "Warning: No descriptive name found for task letter '$LETTER'. Skipping."
        continue # Skip to the next iteration if no mapping is found
    fi

    python trainer_pl_all.py \
        --config-name=DP_DecoupleActionHead_stage2 \
        n_demo=100 \
        task_alphabet=$LETTER \
        task.env_runner.n_envs=28 \
        training.val_every=1000 \
        logging.project="DecoupleActionHead_Stage2_Summary" \
        logging.grouping="A1000_A100_Stage2" \
        ckpt_path="$ckpt_path" \
        train_mode=stage2_rollout

done

echo "All specified tasks completed!"
