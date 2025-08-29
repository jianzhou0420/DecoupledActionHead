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
SEED="$2"
shift
shift
EXTRA_ARGS="$@"  # capture all remaining args


echo "Received task letters: $INPUT_TASK_LETTERS"
echo "Extra arguments: $EXTRA_ARGS"
echo "---"

date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
EXP_NAME="ICRA_DP_T_Stage1_Multi_seed${SEED}"
# build your run_dir

# ---
# Iterate through each letter and run the corresponding task
# ---

run_name="${EXP_NAME}__${INPUT_TASK_LETTERS}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"
num_tasks=${#INPUT_TASK_LETTERS}
n_demo_value=$(( num_tasks * 1000 ))

python trainer_pl_all.py \
    --config-name=ICRA_Decoupled_DP_T_stage1 \
    seed=${SEED} \
    \
    task_alphabet=$INPUT_TASK_LETTERS \
    train_mode=stage1 \
    n_demo=$n_demo_value \
    ckpt_path=$ckpt_path \
    \
    dataloader.num_workers=32 \
    dataloader.batch_size=256 \
    training.val_every=1 \
    training.checkpoint_every=1 \
    \
    run_dir="$run_dir" \
    run_name="${run_name}" \
    \
    logging.project="ICRA_Decoupled_Final_Experiments" \
    logging.group="${EXP_NAME}" \
    logging.name="${run_name}" \
    $EXTRA_ARGS && 
    rsync -avP ${run_dir}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name}/ &&
    rm -rf ${run_dir}
