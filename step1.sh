#!/bin/bash

# Stage 1: 只训练codebook（不涉及LLM）
# 使用VQ-VAE风格的loss，防止codebook塌缩

# 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 数据路径
EMBEDDINGS_PATH="/mnt/workspace/wangliang/alignx/AlignX_train_150k_ugc_all_emb.pt"
ENCODER_PATH="/mnt/workspace/wangliang/model/contriever"
OUTPUT_DIR="/mnt/workspace/wangliang/alignx/codebook_stage1_ablation"

# 使用accelerate启动多GPU训练
accelerate launch \
    --num_processes=8 \
    --mixed_precision=bf16 \
    /mnt/workspace/wangliang/alignx/stage1/step1.py \
    --embeddings_path ${EMBEDDINGS_PATH} \
    --encoder_path ${ENCODER_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --codebook_size 1000 \
    --batch_size_per_sample 8 \
    --codebook_weight 1.0 \
    --diversity_weight 0.15 \
    --usage_weight 1 \
    --use_balanced_kmeans_init \
    --init_sample_size 10000 \
    --kmeans_max_iters 100 \
    --batch_size 32 \
    --num_epochs 10 \
    --learning_rate 5e-4 \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --save_steps 1000 
