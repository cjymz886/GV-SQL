"""
Perform inference in parallel, one model deployed on each GPU;
Requires that a single GPU's memory can load the model
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '768'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '12'


import json
import re
import torch
from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Global variable storing the model
model_cache = {}
model_base = '/data/models/Qwen3.8-27B'
adapter_dir = './output/v0-20260923-081134/checkpoint-1179'

input_file = './data/dev_data/sql_qwen_top30_vl_dev.json'
save_file = './data/dev_data/predict/sql_predict_top30_vl_dev_1179.json'
infer_mode = 'vl'


def get_engine(gpu_id):
    """Get or create the inference engine on the specified GPU (singleton pattern)"""
    global model_cache

    if gpu_id not in model_cache:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        print(f"GPU {gpu_id} loading model...")
        enable_thinking = False

        model, processor = get_model_processor(model_base,
                                               device_map=device,
                                               dtype=torch.bfloat16,
                                               )
        model = PeftModel.from_pretrained(model, adapter_dir)
        template = get_template(processor, enable_thinking=enable_thinking)
        engine = TransformersEngine(model, template=template)

        model_cache[gpu_id] = engine
        print(f"GPU {gpu_id} model loaded")

    return model_cache[gpu_id]

def inference_on_gpu(gpu_id, prompts_batch):
    """Process all requests assigned to the same GPU in batches"""
    engine = get_engine(gpu_id)

    # Build inference requests in batch
    infer_requests = []
    if infer_mode=='vl':
        for line in prompts_batch:
            infer_requests.append(InferRequest(messages=[{
                "role": "user",
                "content": line['messages'][0]['content'],
            }],images=line['images']))
    else:
        for line in prompts_batch:
            infer_requests.append(InferRequest(messages=[{
                "role": "user",
                "content": line['messages'][0]['content'],
            }]))

    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer(infer_requests, request_config=request_config)

    # Extract responses
    results = [resp.choices[0].message.content for resp in resp_list]
    return results

def extract_sql(text):
    p1 = re.compile("```(.*?)```")
    if p1.findall(text):
        sql = p1.findall(text)[0]
    else:
        sql = ""
        print(text)
    return sql

def parallel_inference(gpu_ids=[0,1,2,3]):
    """Run multiple inference tasks concurrently; gpu_ids must match the number of available GPUs"""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    prompt_list = data

    # Assign tasks by grouping them per GPU
    gpu_batches = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, prompt in enumerate(prompt_list):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        gpu_batches[gpu_id].append((idx, prompt))

    # Start processes using the spawn method
    with ProcessPoolExecutor(max_workers=len(gpu_ids),
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        futures = {}
        for gpu_id, batch in gpu_batches.items():
            if batch:  # If there are tasks assigned to this GPU
                prompts = [prompt for _, prompt in batch]
                idxs = [idx for idx, _ in batch]
                future = executor.submit(inference_on_gpu, gpu_id, prompts)
                futures[future] = idxs

        # Collect results
        results = [None] * len(prompt_list)
        for future in as_completed(futures):
            idxs = futures[future]
            batch_results = future.result()
            for idx, result in zip(idxs, batch_results):
                results[idx] = result

    D = []
    for i, pred_sql in enumerate(results):
        line = data[i]
        item = line['data']
        gold_sql = line['messages'][1]['content']
        gold_sql = extract_sql(gold_sql)
        pred_sql = extract_sql(pred_sql)
        D.append({'pred_sql': pred_sql, 'gold_sql': gold_sql, 'data': item})

    # 3142
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps(D, ensure_ascii=False, indent=2) + '\n')

    return results


if __name__ == "__main__":

    multiprocessing.set_start_method('spawn', force=True)
    results = parallel_inference()
    print(f"Processed {len(results)} requests")
