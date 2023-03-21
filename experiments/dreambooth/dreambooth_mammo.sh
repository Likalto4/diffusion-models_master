#!/bin/bash

MODEL_NAME="runwayml/stable-diffusion-v1-5"
OUTPUT_DIR="/home/ricardo/master_thesis/extra_materials/Diffusion_models_HF_course/results/mammo_bs8_ga-8_GS"
INSTANCE_DATA_DIR="/home/ricardo/master_thesis/extra_materials/Diffusion_models_HF_course/data/breast10p_RGB"
INSTANCE_PROMPT="a mammogram"
MAX_TRAIN_STEPS=640
# WANDB_START_METHOD="thread"
WANDB_DISABLE_SERVICE=true
# WANDB_CONSOLE="off"

# --pretrained_vae_name_or_path="stabilityai/sd-vae-ft-mse" \ # Face VAE

accelerate launch dreambooth_mammo.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --instance_data_dir=$INSTANCE_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --revision="fp16" \
  --instance_prompt="$INSTANCE_PROMPT" \
  --seed=1337 \
  --resolution=512 \
  --train_batch_size=8 \
  --train_text_encoder \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --set_grads_to_none \
  --gradient_accumulation_steps=8 --gradient_checkpointing \
  --learning_rate=1e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --dataloader_num_workers=4 \
  --max_train_steps=$MAX_TRAIN_STEPS \
  --num_validation_images=4 \
  --validation_steps=500 \
  --report_to="wandb" \
  --checkpointing_steps=200000 \
  --validation_prompt="$INSTANCE_PROMPT" \
  --enable_xformers_memory_efficient_attention