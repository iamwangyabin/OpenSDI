#!/bin/bash

base_dir="./output_dir"
mkdir -p ${base_dir}

torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=2 \
./test.py \
    --model MaskCLIP \
    --model_setting_name 'ViT' \
    --edge_mask_width 7 \
    --world_size 1 \
    --checkpoint_path "output_dir/Si.pth" \
    --test_batch_size 32 \
    --image_size 512 \
    --if_resizing \
    --output_dir "./log/" \
    --log_dir "./log/"
