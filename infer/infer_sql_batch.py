import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
os.environ['IMAGE_MAX_TOKEN_NUM'] = '768'
os.environ['VIDEO_MAX_TOKEN_NUM'] = '128'
os.environ['FPS_MAX_FRAMES'] = '12'
import json
import re
import torch
import tqdm
from peft import PeftModel
from swift import get_model_processor, get_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig
import json

# from swift import InferEngine, InferRequest, InferStats, RequestConfig, TransformersEngine,

model_base = '/home/ubuntu/cjy/models/Qwen3.6-27B'
adapter_dir = 'output/sql_recall_top20_vl/checkpoint-1178'  #
enable_thinking = False
model, processor = get_model_processor(model_base,
                                       device_map='auto',
                                       dtype=torch.bfloat16,
                                       attn_impl='flash_attn',  #flash_attention_2,flash_attn
                                       # max_pixels=1572864,
                                       )
model = PeftModel.from_pretrained(model, adapter_dir)
template = get_template(processor, enable_thinking=enable_thinking)
engine = TransformersEngine(model, template=template, max_batch_size=3)



def extract_sql(text):
    p1 = re.compile("```(.*?)```")
    if p1.findall(text):
        sql = p1.findall(text)[0]
    else:
        sql = ""
    return sql


def get_batch_data(data,bs=2):
    infer_requests = []
    data_list = []
    for line in data:
        infer_requests.append(InferRequest(messages=[{
            "role": "user",
            "content": line['messages'][0]['content'],
        }], images=line['images']))
        data_list.append(line)

    if len(infer_requests)==bs:
        yield (infer_requests,data_list)

    if infer_requests:
        yield (infer_requests,data_list)


def generate_sql(infer_requests):
    request_config = RequestConfig(max_tokens=512, temperature=0)
    resp_list = engine.infer(infer_requests, request_config=request_config)
    res = []
    for line in resp_list:
        res.append(line.choices[0].message.content)
    return res


def predict_dev_data():
    with open('./data/bird/train/sql_qwen_recall_top20_vl_train.json','r',encoding='utf-8') as f:
        data = json.load(f)

    D = []
    for infer_requests,data_list in get_batch_data(data):

        res = generate_sql(infer_requests)
        assert len(res) == len(data_list)
        for i,pred_sql in enumerate(res):
            gold_sql =data_list[i]['messages'][1]['content']
            gold_sql = extract_sql(gold_sql)
            pred_sql = extract_sql(pred_sql)
            # print('gold_sql',gold_sql)
            # print('pred_sql',pred_sql)

            D.append({'pred_sql': pred_sql, 'gold_sql': gold_sql, 'data': data_list[i]['data']})

    #3142
    with open('./data/bird/train/predict/sql_qwen_train_recall_top20_vl_1178.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(D, ensure_ascii=False, indent=2) + '\n')




if __name__ == "__main__":
    predict_dev_data()





