# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import json
import tqdm


model_path = "/home/ubuntu/cjy/mystudy/sql_recall_rank/output/v0-20260727-092033/checkpoint-285" #1140,2280,3420


tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             dtype=torch.float16,
                                             attn_implementation="flash_attention_2").cuda().eval()

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192


def format_instruction(query, doc):
    instruction = 'Given sql query, retrieve relevant columns that answer the query'
    # instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                     query=query, doc=doc)
    return output


def process_inputs(pairs):

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs


@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores





def get_ranker_api(query,doc_list):
    pairs = [format_instruction(query, doc) for doc in doc_list]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)
    return scores

def get_rank_result(query,key_list,meaning_data,bs=8):

    doc_list = []
    for doc in key_list:
        if meaning_data:
            meaning = meaning_data.get(doc,'')
            if meaning:
                doc = doc +':' + meaning
        doc_list.append(doc)

    batch = []
    fres = []
    for doc in doc_list:
        batch.append(doc)
        if len(batch)==bs:
            res = get_ranker_api(query,batch)
            for i,sim in enumerate(res):
                key = batch[i]
                fres.append([key,sim])
            batch = []

    if batch:
        res = get_ranker_api(query, batch)
        for i, sim in enumerate(res):
            key = batch[i]
            fres.append([key,sim])
    return fres



def get_ranker_dev():

    with open('./data/bird/dev/dev_enrich_column.json','r',encoding='utf-8') as f:
        meaning_data = json.load(f)

    with open('./data/bird/dev/predict/sql_dev_recall_enrich_570.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    D = []
    for i in tqdm.tqdm(range(len(data))):
        line = data[i]
        query = line['question']
        evidence = line['evidence']
        query = query + ' ' + evidence
        recall_res = line['recall_res']
        table_list = [d[0] for d in recall_res ]
        rank_res = get_rank_result(query,table_list,meaning_data=meaning_data)
        rank_res = sorted(rank_res, key=lambda x: x[1], reverse=True)
        line['rank_res'] = rank_res
        D.append(line)

    with open('./data/bird/dev/predict/dev_sql_enrich_column_570_ranker_285.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(D, ensure_ascii=False, indent=2) + '\n')


if __name__=="__main__":
    get_ranker_dev()

