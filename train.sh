#!/bin/bash

base_dir="./output_dir"
mkdir -p ${base_dir}

# train with SD1.5 needs 4 GPUs
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=1 \
./train.py \
    --exp_name MaskCLIP_sd15 \
    --model_setting_name 'ViTL' \
    --model MaskCLIP \
    --world_size 1 \
    --batch_size 16 \
    --data_path "nebula/OpenSDI_train" \
    --epochs 50 \
    --lr 1e-4 \
    --image_size 512 \
    --if_resizing \
    --min_lr 0 \
    --weight_decay 0.05 \
    --edge_mask_width 7 \
    --if_predict_label \
    --if_not_amp \
    --test_data_path "nebula/OpenSDI_test" \
    --warmup_epochs 0 \
    --output_dir "./output_dir" \
    --log_dir "./output_dir" \
    --accum_iter 1 \
    --seed 42 \
    --test_period 1



