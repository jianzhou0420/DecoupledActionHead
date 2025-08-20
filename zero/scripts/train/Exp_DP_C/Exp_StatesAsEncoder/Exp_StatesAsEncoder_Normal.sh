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
EXP_NAME="Exp_StatesAsEncoder_Normal"


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
        --config-name=DP_DecoupleActionHead_stage2_states \
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
        logging.project="DecoupleActionHead_test" \
        logging.group="${EXP_NAME}" \
        logging.name="${run_name}" &&
        rsync -avP ${run_dir}/ jian@10.12.65.19:/media/jian/data/cached_from_sub_machine/runtime/${time_part}_${run_name}/ &&
        rm -rf ${run_dir}
done

# config logic:
# 数据集类、训练参数类、文件管理类、logging类、

# 数据集类：
# - task_alphabet: 任务字母
# - train_mode: 训练模式（stage1）
# - n_demo: 演示数量
# \
# 训练参数类：
# - dataloader.num_workers: 数据加载器的工作线程数，用的zarr，loadfromdisk，所以可以多一点
# - training.val_every: 验证间隔，这里相当于不需要eval
# - training.checkpoint_every: 检查点保存间隔
# \
# 文件管理类：
# - run_dir: 运行目录，存储输出结果
# - run_name: 运行名称，便于区分不同任务
# \
# logging类：
# - logging.project: 日志项目名称
# - logging.group: 日志分组名称
# - logging.name: 日志名称，包含任务字母
