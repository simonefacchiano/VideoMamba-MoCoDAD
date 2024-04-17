#!/usr/bin/env bash


export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

JOB_NAME='videomamba_middle_f32_res224'
OUTPUT_DIR="$(dirname $0)/$JOB_NAME"
LOG_DIR="./logs/${JOB_NAME}"
PREFIX='/media/hdd/simone/train_fixed' #'/media/hdd/aleflabo/breakfast'
DATA_PATH='/media/hdd/simone/train_fixed' #'/media/hdd/aleflabo/breakfast'

PARTITION='video5'
GPUS=1
GPUS_PER_NODE=1
CPUS_PER_TASK=16

python run_class_finetuning.py \
    --model videomamba_middle \
    --data_path ${DATA_PATH} \
    --prefix ${PREFIX} \
    --data_set 'Breakfast' \
    --finetune /media/hdd/aleflabo/breakfast/pretrained/videomamba_m16_k400_mask_ft_f32_res224.pth \
    --split ',' \
    --nb_classes 10 \
    --log_dir ${OUTPUT_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size 8 \
    --num_sample 2 \
    --input_size 224 \
    --short_side_size 224 \
    --save_ckpt_freq 100 \
    --num_frames 32 \
    --orig_t_size 32 \
    --num_workers 12 \
    --warmup_epochs 5 \
    --tubelet_size 1 \
    --epochs 45 \
    --lr 2e-4 \
    --layer_decay 0.8 \
    --drop_path 0.4 \
    --opt adamw \
    --opt_betas 0.9 0.999 \
    --weight_decay 0.05 \
    --test_num_segment 4 \
    --test_num_crop 3 \
    --dist_eval \
    --test_best \
    --disable_eval_during_finetuning
