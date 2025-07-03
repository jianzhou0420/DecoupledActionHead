#!/bin/bash

# ——— 固定参数 ———
N_DEMOS=1000
TASK_LETTER="A"
TASK_NAME="stack_d1" # 任务 A 的描述名

echo "开始执行 Task $TASK_LETTER ($TASK_NAME) 在 indices 0,10,20,30,40 上共 5 次，n_demo=$N_DEMOS"
echo "---------------------------------------------------------"

for i in $(seq 9 10 29); do
    # 格式化两位数：00, 10, 20, 30, 40
    idx=$(printf "%03d" "$i")

    CKPT_PATH="data/robomimic/Stage1/2025.06.08_00.00.00_pretrain_JPee_stage1_stack_d1_1000/checkpoint_epoch\=$idx.ckpt"

    echo "[Run index $idx] using checkpoint: $CKPT_PATH"
    python trainer_pl_all.py --config-name=DP_DecoupleActionHead_stage2 \
        train_mode=stage2_rollout \
        n_demo=$N_DEMOS \
        task_alphabet=$TASK_LETTER \
        ckpt_path=$CKPT_PATH \
        logging.project=BestEpoch \
        logging.name=Stage2_${TASK_LETTER}_${N_DEMOS}_idx_${idx} \
        training.rollout_every=1 \
        training.checkpoint_every=10000

    echo "[Finished index $idx]"
    echo "---------------------------------------------------------"
done

echo "所有指定索引 (00,10,20,30,40) 的运行已完成！"
