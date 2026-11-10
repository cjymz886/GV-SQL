IMAGE_MAX_TOKEN_NUM=768 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=4,5,6,7 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=4 \
swift sft \
    --model "/data/models/Qwen3.8-27B" \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --vit_lr 1e-5 \
    --aligner_lr 5e-5 \
    --target_modules all-linear \
    --dataset "/data/cjy/mystudy/GV-SQL/data/train_data/sql_qwen_top30_vl_train.json" \
    --add_non_thinking_prefix true \
    --attn_impl flash_attn \
    --loss_scale ignore_empty_think \
    --torch_dtype bfloat16 \
    --load_from_cache_file false \
    --group_by_length true \
    --split_dataset_ratio 0.0 \
    --eval_strategy steps \
    --output_dir output \
    --save_steps 1179 \
    --eval_steps 1179 \
    --save_total_limit 1 \
    --logging_steps 10 \
    --num_train_epochs 1 \
    --max_length 25000 \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --deepspeed zero3 \
    --model_author swift \
    --model_name swift-robot

