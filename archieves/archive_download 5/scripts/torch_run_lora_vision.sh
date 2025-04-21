#!/bin/bash

# You can use 2B instead of 7B
# MODEL_NAME="Qwen/Qwen2-VL-7B-Instruct"
MODEL_NAME="Qwen/Qwen2-VL-2B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-VL-7B-Instruct"

export PYTHONPATH=src:$PYTHONPATH

GLOBAL_BATCH_SIZE=8
BATCH_PER_DEVICE=1
NUM_DEVICES=1
GRAD_ACCUM_STEPS=4

# If you want to tune the `embed_token` with LoRA, You need to tune `lm_head` together
# You should freeze the the merger also, becuase the merger is included in the vision_tower.

torchrun --nproc_per_node=4 src/training/train.py \
    --use_liger True \
    --lora_enable True \
    --vision_lora True \
    --use_dora False \
    --lora_namespan_exclude "['lm_head', 'embed_tokens','experts']" \
    --lora_rank 4 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --num_lora_modules -1 \
    --model_id $MODEL_NAME \
    --data_path filtered_PangeaIns.json \
    --image_folder downloads/datasets--neulab--PangeaInstruct/snapshots/d0819917abe1cae38c008de0ca172f885f1f26a4 \
    --remove_unused_columns False \
    --freeze_vision_tower True \
    --freeze_llm True \
    --tune_merger False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 True \
    --gradient_checkpointing False \
    --output_dir output/lora_vision_test \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BATCH_PER_DEVICE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --image_min_pixels $((256 * 28 * 28)) \
    --image_max_pixels $((512 * 28 * 28)) \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --dataloader_num_workers 4 \
    --bits 4 \
    --optim adamw_8bit