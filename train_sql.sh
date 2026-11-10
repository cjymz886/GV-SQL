CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=8 \
swift sft \
    --model "/home/ubuntu/cjy/models/Qwen3.6-27B" \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --learning_rate 1e-4 \
    --target_modules all-linear \
    --dataset "/data/cjy/mystudy/GV-SQL/data/train_data/sql_qwen_top30_train.json" \
    --attn_impl flash_attn \
    --add_non_thinking_prefix true \
    --loss_scale ignore_empty_think \
    --group_by_length true \
    --torch_dtype bfloat16 \
    --load_from_cache_file false \
    --split_dataset_ratio 0.0 \
    --eval_strategy steps \
    --output_dir output \
    --save_steps 1179 \
    --eval_steps 1179 \
    --save_total_limit 2 \
    --logging_steps 100 \
    --num_train_epochs 1 \
    --max_length 22000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --warmup_ratio 0.05 \
    --save_only_model true \
    --deepspeed zero3 \
    --model_author swift \
    --model_name swift-robot


