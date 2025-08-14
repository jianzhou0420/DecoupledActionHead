#!/bin/bash

# --------------------
# region 1. Input
if [ -z "$1" ]; then
    echo "Usage: $0 <task_letters>"
    echo "Example: $0 ABCDEFG"
    echo "Example: $0 ABE"
    exit 1
fi

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

INPUT_TASK_LETTERS="$1"
echo "Received task letters: $INPUT_TASK_LETTERS"
echo "---"

# --------------------
# region 2. Run

date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
EXP_NAME="Exp_DP_FILM_MLP_Normal_1000_SingleLayer"

# build your run_dir

# ---
# Iterate through each letter and run the corresponding task
# ---

for LETTER in $(echo "$INPUT_TASK_LETTERS" | sed -e 's/\(.\)/\1 /g'); do
    DESCRIPTIVE_TASK_NAME=${TASK_MAP["$LETTER"]}

    run_name="${EXP_NAME}__${LETTER}"
    run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

    ckpt_path=""

    python trainer_pl_all.py \
        --config-name=DP_DecoupleActionHead_MLP_stage2_film \
        \
        policy.n_layers=1\
        \
        task_alphabet=$LETTER \
        train_mode=normal_rollout \
        n_demo=1000 \
        ckpt_path=${ckpt_path} \
        \
        dataloader.num_workers=16 \
        training.val_every=1 \
        \
        run_dir="$run_dir" \
        run_name="${run_name}" \
        \
        logging.project="DecoupleActionHead_MLP" \
        logging.group="${EXP_NAME}" \
        logging.name="${run_name}" &&
        rsync -avP ${run_dir}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name}/ &&
        rm -rf ${run_dir}
done


