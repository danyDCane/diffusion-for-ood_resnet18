#!/bin/bash

# 数据集选择：cifar10 或 cifar100
DATASET=${1:-cifar10}  # 默认使用 cifar10，可以通过命令行参数指定

# 根据数据集设置项目名称
if [ "$DATASET" == "cifar100" ]; then
    WANDB_PROJECT="diffusion-ood-cifar100"
    WANDB_NAME="diffusion-ood-c100"
    NUM_CLASSES=100
else
    WANDB_PROJECT="diffusion-ood-cifar10"
    WANDB_NAME="diffusion-ood-c10"
    NUM_CLASSES=10
fi

python3 train_cifar10.py \
--dataset $DATASET \
--num_classes $NUM_CLASSES \
--data_root ../datasets \
--batch_size 128 \
--num_workers 4 \
--epochs 200 \
--lr_backbone 0.1 \
--lr_diffusion 5e-5 \
--momentum 0.9 \
--weight_decay 5e-4 \
--lambda_diff 1.0 \
--diffusion_denoiser_channels 512 \
--num_diffusion_steps 1000 \
--device cuda \
--save_dir ./checkpoints \
--save_freq 10 \
--eval_freq 5 \
--use_wandb \
--wandb_project $WANDB_PROJECT \
--wandb_name $WANDB_NAME

