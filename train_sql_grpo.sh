export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=7200
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo  # 使用本地回环
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_PORT=29501  # 更换端口，避免冲突
export OMP_NUM_THREADS=4  # 适当增加

IMAGE_MAX_TOKEN_NUM=768 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=5,6,7 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=3 \
swift rlhf \
    --rlhf_type grpo \
    --model "/data/cjy/mystudy/sql_recall_rank/output/sql_recall_top30/checkpoint-1178-merged" \
    --dataset "/data/cjy/mystudy/sql_recall_rank/data/train/sql_qwen_recall_top30_train.json" \
    --remove_unused_columns false \
    --load_from_cache_file false \
    --torch_dtype bfloat16 \
    --beta 0.001 \
    --epsilon 3e-4 \
    --epsilon_high 4e-4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 6 \
    --tuner_type lora \
    --lora_rank 32 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --external_plugins /data/cjy/mystudy/sql_recall_rank/reward.py \
    --reward_funcs sql_accuracy sql_format \
    --reward_weights 1.0 0.1 \
    --enable_thinking false \
    --use_vllm true \
    --vllm_mode server \
    --vllm_enable_lora true \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --vllm_server_timeout 360 \
    --max_length 25000 \
    --max_completion_length 512 \
    --save_steps 500 \
    --eval_steps 500 \
    --learning_rate 5e-7 \
    --save_total_limit 20 \
    --logging_steps 5 \
    --max_grad_norm 1.0 \
    --warmup_steps 0.05 \
    --dataloader_num_workers 6 \
    --resume_from_checkpoint "/data/cjy/mystudy/sql_recall_rank/output/checkpoint-1178-merged/v1-20260821-103925/checkpoint-2000" \
    --deepspeed zero2 \
    --log_completions true \
    --async_generate false

#    --offload_optimizer true \
#    --offload_model true \
#    --sleep_level 1 \
#    --steps_per_generation 8 \
#   --gradient_checkpointing true \
#