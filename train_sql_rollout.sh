export NCCL_P2P_DISABLE=1
export NCCL_TIMEOUT=10000
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo  # 使用本地回环
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=INFO
export MASTER_PORT=29502  # 更换端口，避免冲突
export OMP_NUM_THREADS=4  # 适当增加

IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
CUDA_VISIBLE_DEVICES=4 \
swift rollout \
    --model "/data/cjy/mystudy/sql_recall_rank/output/sql_repair3_vl_1024/checkpoint-786-merged" \
    --vllm_tensor_parallel_size 1 \
    --vllm_data_parallel_size 1 \
    --vllm_gpu_memory_utilization 0.9  \
    --vllm_max_model_len 32000 \
    --vllm_enable_prefix_caching true \
    --vllm_enable_lora true \
    --temperature 1.0 \
    --top_p=0.95 \
    --max_new_tokens 4096 \
    --vllm_max_lora_rank 32 \

