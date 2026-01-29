#!/bin/bash

# Stage 2: 训练MLP投影层（LLM冻结，使用训练好的PQ codebook）

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 路径配置
DATA_PATH="/mnt/workspace/wangliang/alignx/AlignX_train_150k_unique_prompt_add.jsonl"
EMBEDDINGS_PATH="/mnt/workspace/wangliang/alignx/AlignX_train_150k_ugc_all_emb.pt"
LLM_PATH="/mnt/workspace/wangliang/model/qwen-7B-soft"
CODEBOOK_PATH="/mnt/workspace/wangliang/alignx/codebook_stage1/stage1_final/codebook_model.pt"
OUTPUT_DIR="/mnt/workspace/wangliang/alignx/codebook_stage2"

# 使用accelerate启动多GPU训练
accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    /mnt/workspace/wangliang/alignx/stage1/step2.py \
    --data_path ${DATA_PATH} \
    --embeddings_path ${EMBEDDINGS_PATH} \
    --llm_path ${LLM_PATH} \
    --codebook_path ${CODEBOOK_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --his_len 8 \
    --hidden_dim 3584 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_epochs 1 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --save_steps 1000 \
    --max_length 500
