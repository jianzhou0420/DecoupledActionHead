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
EXP_NAME="Exp_Multitask_OneStop"
# build your run_dir

# ---
# Iterate through each letter and run the corresponding task
# ---
run_name_stage1="${EXP_NAME}__${INPUT_TASK_LETTERS}_stage1"
run_name_stage2="${EXP_NAME}__${INPUT_TASK_LETTERS}_stage2"
run_dir_stage1="data/outputs/${date_part}/${time_part}_${run_name_stage1}"
run_dir_stage2="data/outputs/${date_part}/${time_part}_${run_name_stage2}"

ckpt_path=${run_dir}/checkpoints/last.ckpt

python trainer_pl_all.py \
    --config-name=DP_DecoupleActionHead_stage1 \
    n_demo=1000 \
    task_alphabet=$INPUT_TASK_LETTERS \
    dataloader.num_workers=16 \
    training.val_every=1000 \
    training.checkpoint_every=10 \
    logging.project="DecoupleActionHead_Stage1_Summary" \
    logging.group="${EXP_NAME}" \
    logging.name="${run_name_stage1}" \
    train_mode=stage1 \
    run_dir="$run_dir_stage1" \
    run_name="${run_name_stage1}" &&
    python trainer_pl_all.py \
        --config-name=DP_DecoupleActionHead_stage2 \
        n_demo=1000 \
        task_alphabet=$INPUT_TASK_LETTERS \
        dataloader.num_workers=16 \
        training.val_every=1000 \
        logging.project="DecoupleActionHead_Stage2_Summary" \
        logging.group="${EXP_NAME}" \
        logging.name="${run_name_stage2}" \
        ckpt_path="$ckpt_path" \
        train_mode=stage2 \
        run_dir="$run_dir_stage2" \
        run_name="${run_name_stage2}" &&
    rsync -avP ${run_dir_stage1}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name_stage1}/ &&
    rsync -avP ${run_dir_stage2}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name_stage2}/ &&
    rsync -avP ${run_dir_stage2}/ jian@10.12.65.130:/data/eval_candidates/${time_part}_${run_name_stage2}/ &&
    rm -rf ${run_dir_stage1} &&
    rm -rf ${run_dir_stage2} &&
    ssh jian@10.12.65.19 "touch /media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name_stage2}/ready.flag" &&
    ssh jian@10.12.65.130 "touch /data/eval_candidates/${time_part}_${run_name_stage2}/ready.flag"

echo "All specified tasks completed!"
