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
# region 2. Run SameEncoder_Normal_100

date_part=$(date +'%Y.%m.%d')
time_part=$(date +'%H.%M.%S')
EXP_NAME="Exp_SameEncoder_Normal_100"


# build your run_dir

LETTER=$INPUT_TASK_LETTERS

DESCRIPTIVE_TASK_NAME=${TASK_MAP["$LETTER"]}

run_name="${EXP_NAME}__${LETTER}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

ckpt_path=""

python trainer_pl_all_fix_encoder.py \
    --config-name=DP_DecoupleActionHead_stage2 \
    \
    encoder_path="data/encoder_Exp_Normal_1000__F_epoch=049.ckpt.pth" \
    \
    task_alphabet=$LETTER \
    train_mode=normal_rollout \
    n_demo=100 \
    ckpt_path=${ckpt_path} \
    \
    dataloader.num_workers=16 \
    training.val_every=1000 \
    \
    run_dir="$run_dir" \
    run_name="${run_name}" \
    \
    logging.project="DecoupleActionHead_Normal_Headless" \
    logging.group="${EXP_NAME}" \
    logging.name="${run_name}" &&
    rsync -avP ${run_dir}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name}/ &&
    rm -rf ${run_dir}

# --------------------
# region 2. Run


EXP_NAME="Exp_SameEncoder_Single100_16_16"

# build your run_dir

# ---
# Iterate through each letter and run the corresponding task
# ---


DESCRIPTIVE_TASK_NAME=${TASK_MAP["$LETTER"]}

run_name="${EXP_NAME}__${LETTER}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

ckpt_path="data/robomimic/Stage1/Exp_Single1000_16_16_49/Exp_Single1000_16_16_Stage1__${LETTER}_epoch\=049.ckpt"

python trainer_pl_all_fix_encoder.py \
    --config-name=DP_DecoupleActionHead_stage2 \
    \
    encoder_path="data/encoder_Exp_Normal_1000__F_epoch=049.ckpt.pth" \
    \
    task_alphabet=$LETTER \
    train_mode=stage2_rollout \
    n_demo=100 \
    ckpt_path=${ckpt_path} \
    \
    dataloader.num_workers=16 \
    training.val_every=1000 \
    \
    run_dir="$run_dir" \
    run_name="${run_name}" \
    \
    logging.project="DecoupleActionHead_Normal_Headless"  \
    logging.group="${EXP_NAME}" \
    logging.name="${run_name}" &&
    rsync -avP ${run_dir}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name}/ &&
    rm -rf ${run_dir}


# --------------------
# region 3. Run SameEncoder_Other1000_A100_16_16


EXP_NAME="Exp_SameEncoder_Other1000_A100_16_16"

# build your run_dir

# ---
# Iterate through each letter and run the corresponding task
# ---

DESCRIPTIVE_TASK_NAME=${TASK_MAP["$LETTER"]}

run_name="${EXP_NAME}__${LETTER}"
run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

ckpt_path="data/robomimic/Stage1/Exp_Single1000_16_16_49/Exp_Single1000_16_16_Stage1__B_epoch\=049.ckpt"

python trainer_pl_all_fix_encoder.py \
    --config-name=DP_DecoupleActionHead_stage2 \
    \
    encoder_path="data/encoder_Exp_Normal_1000__F_epoch=049.ckpt.pth" \
    \
    task_alphabet=$LETTER \
    train_mode=stage2_rollout \
    n_demo=1000 \
    ckpt_path=${ckpt_path} \
    \
    dataloader.num_workers=16 \
    training.val_every=1000 \
    \
    run_dir="$run_dir" \
    run_name="${run_name}" \
    \
    logging.project="DecoupleActionHead_Normal_Headless"  \
    logging.group="${EXP_NAME}" \
    logging.name="${run_name}" &&
    rsync -avP ${run_dir}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name}/ &&
    rm -rf ${run_dir}



