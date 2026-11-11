import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import tqdm
from sentence_transformers import SentenceTransformer
import numpy as np

model_base = "/home/ubuntu/cjy/models/Qwen3-Embedding-8B"
model_path = "/home/ubuntu/cjy/mystudy/sql_recall_rank/output/recall_model/checkpoint-570" #285,570,855

# tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
# model = AutoModel.from_pretrained(model_path,
#                                   attn_implementation="flash_attention_2",
#                                   dtype=torch.float16,
#                                   device_map="auto")


model = SentenceTransformer(
    model_path,
    device = torch.device('cuda'),
    # model_kwargs={"attn_implementation": "flash_attention_2"},
    processor_kwargs={"padding_side": "left"}
)


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'


def get_embed_api(query,doc_list):
    task = 'Given a query, retrieve the relevant tables and corresponding column names.'

    queries = [
        get_detailed_instruct(task, query)
    ]
    input_texts = queries + doc_list

    max_length = 1024

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    with torch.no_grad():
        outputs = model(**batch_dict)
        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        scores = (embeddings[:1] @ embeddings[1:].T)

    # print(scores.tolist())
    return scores.tolist()


def get_embed_api2(q_list):
    with torch.no_grad():
        query_embeddings = model.encode(q_list, prompt_name="query")
        norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        return query_embeddings / norms

def get_query_doc_sim(query,db_id,table_list,meaning_data,bs=8):

    doc_list = []
    for tab in table_list:
        for col in table_list[tab]:
            doc = db_id + '|' + tab +'|' + col
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
            res = get_embed_api(query,batch)
            for i,sim in enumerate(res[0]):
                key = batch[i]
                fres.append([key,sim])
            batch = []

    if batch:
        res = get_embed_api(query, batch)
        for i, sim in enumerate(res[0]):
            key = batch[i]
            fres.append([key,sim])

    return fres

def get_query_doc_sim2(db_map,meaning_data,bs=8):

    db_embed = {}
    for db_id in db_map:
        # if db_id != _id:
        #     continue
        table_list = db_map[db_id]
        doc_list = []
        key_list = []
        for tab in table_list:
            for col in table_list[tab]:
                doc = db_id + '|' + tab +'|' + col
                key_list.append(doc)
                if meaning_data:
                    meaning = meaning_data.get(doc,'')
                    if meaning:
                        doc = doc +':' + meaning
                doc_list.append(doc)

        batch = []
        embed_list = []
        for doc in doc_list:
            batch.append(doc)
            if len(batch)==bs:
                embed = get_embed_api2(batch)
                embed_list.extend(embed)
                batch = []
        if batch:
            embed = get_embed_api2(batch)
            embed_list.extend(embed)

        assert len(doc_list) == len(embed_list)
        all_embeddings = np.vstack(embed_list)
        db_embed[db_id] = {'doc_list':key_list,'doc_embed':all_embeddings}

    return db_embed






def get_embed_dev():
    db_map = {}
    with open('./data/dev/dev_tables.json','r',encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            db_id = item['db_id']
            tabel_names = item['table_names_original']
            column_names = item['column_names_original']

            col_map = {}
            for i,col in enumerate(column_names):
                k,n = col
                if k not in col_map:
                    col_map[k] = [n]
                else:
                    col_map[k].append(n)

            table_map = {}
            for i, tab in enumerate(tabel_names):
                table_map[tab] = col_map[i]
                table_map[tab].append('*')

            db_map[db_id] = table_map

    D = []

    with open ('./data/dev/dev_sql_final.json','r',encoding='utf-8') as f:
        data = json.load(f)
        for i in tqdm.tqdm(range(len(data))):
            item = data[i]
            query = item['question']
            db_id = item['db_id']
            tabel_list = db_map[db_id]
            sim_fres = get_query_doc_sim(query,db_id,tabel_list,None)
            # print(sim_fres)
            sim_fres = sorted(sim_fres,key=lambda x:x[1],reverse=True)
            # print(sim_fres)
            item['recall_res'] = sim_fres
            D.append(item)

    with open('./data/dev/dev_sql_final_embed.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')



def get_embed_meaning_dev():
    db_map = {}
    with open('./data/bird/dev/dev_tables.json','r',encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            db_id = item['db_id']
            tabel_names = item['table_names_original']
            column_names = item['column_names_original']
            col_map = {}
            for i,col in enumerate(column_names):
                k,n = col
                if k not in col_map:
                    col_map[k] = [n]
                else:
                    col_map[k].append(n)

            table_map = {}
            for i, tab in enumerate(tabel_names):
                table_map[tab] = col_map[i]
                table_map[tab].append('*')

            db_map[db_id] = table_map

    with open('./data/bird/dev/dev_enrich_column.json','r',encoding='utf-8') as f:
        meaning_data = json.load(f)

    D = []
    with open ('./data/bird/dev/sql_recall_filter.json','r',encoding='utf-8') as f:
        data = json.load(f)
        for i in tqdm.tqdm(range(len(data))):
            item = data[i]
            query = item['question']
            db_id = item['db_id']
            tabel_list = db_map[db_id]
            sim_fres = get_query_doc_sim2(query,db_id,tabel_list,meaning_data)
            print(sim_fres)
            sim_fres = sorted(sim_fres,key=lambda x:x[1],reverse=True)
            print(sim_fres)
            item['recall_res'] = sim_fres
            D.append(item)
            break

    with open('./data/bird/dev/sql_dev_recall_enrich_143.jsonl', 'w', encoding='utf-8') as f:
        f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')


def get_embed_meaning_dev2():
    db_map = {}
    with open('./data/bird/dev/dev_tables.json','r',encoding='utf-8') as f:
        data = json.load(f)
        for item in data:
            db_id = item['db_id']
            tabel_names = item['table_names_original']
            column_names = item['column_names_original']
            col_map = {}
            for i,col in enumerate(column_names):
                k,n = col
                if k not in col_map:
                    col_map[k] = [n]
                else:
                    col_map[k].append(n)

            table_map = {}
            for i, tab in enumerate(tabel_names):
                table_map[tab] = col_map[i]
                table_map[tab].append('*')

            db_map[db_id] = table_map

    with open('./data/bird/dev/dev_enrich_column.json','r',encoding='utf-8') as f:
        meaning_data = json.load(f)

    db_embed = get_query_doc_sim2(db_map, meaning_data)

    D = []
    with open ('./data/bird/dev/sql_recall_filter.json','r',encoding='utf-8') as f:
        data = json.load(f)
        for i in tqdm.tqdm(range(len(data))):
            item = data[i]
            query = item['question']
            db_id = item['db_id']
            evidence = item['evidence']

            # db_embed = get_query_doc_sim2(db_id, db_map, meaning_data)
            evid_embed = get_embed_api2([evidence])
            query_embed = get_embed_api2([query])
            doc_embed = db_embed[db_id]['doc_embed']
            doc_list = db_embed[db_id]['doc_list']

            similarities = np.dot(query_embed, doc_embed.T).flatten()
            similarities2 = np.dot(evid_embed, doc_embed.T).flatten()

            doc_similarity_map = {}
            for i,key in enumerate(doc_list):
                doc_similarity_map[key] = similarities[i].item()

            for i,key in enumerate(doc_list):
                score = similarities2[i].item()
                if doc_similarity_map[key] < score:
                    doc_similarity_map[key] = score

            # print(doc_similarity_map)
            sim_fres = sorted(doc_similarity_map.items(), key=lambda x: x[1], reverse=True)
            # print(sim_fres)
            item['recall_res'] = sim_fres
            D.append(item)

    with open('./data/bird/dev/predict/sql_dev_recall_enrich_570.json', 'w', encoding='utf-8') as f:
        f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')


if __name__=="__main__":
    # get_embed_dev()
    # get_embed_meaning_dev()
    get_embed_meaning_dev2()