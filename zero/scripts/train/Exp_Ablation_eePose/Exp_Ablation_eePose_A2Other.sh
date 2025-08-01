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
EXP_NAME="Exp_Ablation_eePose_A2Other"
# build your run_dir

# ---
# Iterate through each letter and run the corresponding task
# ---
for LETTER in $(echo "$INPUT_TASK_LETTERS" | sed -e 's/\(.\)/\1 /g'); do
    DESCRIPTIVE_TASK_NAME=${TASK_MAP["$LETTER"]}

    run_name="${EXP_NAME}__${LETTER}"
    run_dir="data/outputs/${date_part}/${time_part}_${run_name}"

    ckpt_path="data/robomimic/Stage1/tmp/Exp_Ablation_eePose_OneStop__A_stage1_epoch\=049.ckpt"

    python trainer_pl_all.py \
        --config-name=DP_DecoupleActionHead_stage2 \
        \
        task_alphabet=A \
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
        logging.project="DecoupleActionHead_Stage2_Summary" \
        logging.group="${EXP_NAME}" \
        logging.name="${run_name}" &&
        bash zero/scripts/train/ArchieveAndEval.sh ${run_dir} ${time_part} ${run_name} &&
        rm -rf ${run_dir}
done

