# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift export \
    --adapters output/sql_repair3_vl_1024/checkpoint-786 \
    --merge_lora true