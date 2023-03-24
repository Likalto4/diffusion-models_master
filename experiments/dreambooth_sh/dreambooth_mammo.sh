#!/bin/bash

MODEL_NAME="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="/home/ricardo/master_thesis/diffusion-models_master/results/mammo_abs-64_promt-short"
INSTANCE_DATA_DIR="/home/ricardo/master_thesis/diffusion-models_master/data/images/breast10p_RGB"
INSTANCE_PROMPT="mammogram"
# HP
MAX_TRAIN_STEPS=2500
BATCH_SIZE=1
GRAD_ACC=8
VALIDATION_STEPS=500
NUM_WORKERS=4
LR=1e-6

WANDB_START_METHOD="thread"
WANDB_DISABLE_SERVICE=true
WANDB_CONSOLE="off"
# 
# --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \ # Face VAE

accelerate launch dreambooth_mammo.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=$BATCH_SIZE \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --set_grads_to_none \
  --gradient_accumulation_steps=$GRAD_ACC \
  --learning_rate=$LR \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --dataloader_num_workers=$NUM_WORKERS \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --num_validation_images=4 \
  --validation_steps=$VALIDATION_STEPS \
  --report_to="wandb" \
  --checkpointing_steps=200000 \
  --validation_prompt="$INSTANCE_PROMPT" \
  --enable_xformers_memory_efficient_attention \
  --push_to_hub \
  --gradient_checkpointing \