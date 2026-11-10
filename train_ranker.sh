CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
INFONCE_TEMPERATURE=0.1 \
NPROC_PER_NODE=8 \
swift sft \
    --model "/home/ubuntu/models/Qwen3-Reranker-8B" \
    --task_type generative_reranker \
    --loss_type listwise_reranker \
    --tuner_type full \
    --learning_rate 6e-6 \
    --target_modules all-linear \
    --dataset "/home/ubuntu/cjy/mystudy/sql_recall_rank/data/bird/train/train_sql_ranker_qwen_enrich_neg2.json" \
    --attn_impl flash_attn \
    --padding_free true \
    --torch_dtype bfloat16 \
    --load_from_cache_file true \
    --split_dataset_ratio 0.0 \
    --eval_strategy steps \
    --output_dir output \
    --save_steps 285 \
    --eval_steps 285 \
    --save_total_limit 2 \
    --logging_steps 100 \
    --num_train_epochs 1 \
    --max_length 8192 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --warmup_ratio 0.05 \
    --dataloader_drop_last true \
    --deepspeed zero3 \
    --save_only_model true


#9127
#--loss_type pointwise_reranker \