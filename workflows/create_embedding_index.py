
"""
Build vector indexes for later retrieval:
1) Encode the question (query) of the dataset, used for later few-shot sample recall
2) Encode the enhanced schema columns of the database, used for later schema retrieval
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import json
import numpy as np
import faiss
import tqdm
import pickle
from sentence_transformers import SentenceTransformer
import logging

from utils.my_logger import setup_logging

setup_logging(log_level=logging.INFO, log_file='gv_sql.log')
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device=torch.device('cuda:0')

def build_query_index():
	"""Use the open-source Qwen3-Embedding-8B as the encoding model
	"""
	model_base = "/data/models/Qwen3-Embedding-8B"
	input_file = "./data/BIRD/train/train.json"  #BIRD's original training data file
	index_file = "./data/BIRD/database/query_index.faiss"
	metadata_file = "./data/BIRD/database/query_metadata.pkl"

	def get_emebd(queries):
		with torch.no_grad():
			query_embeddings = model.encode(queries, prompt_name="query")
			norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
			return query_embeddings / norms


	model = SentenceTransformer(
		model_base,
		device=device,
		model_kwargs={"attn_implementation": "flash_attention_2"},
		processor_kwargs={"padding_side": "left"}
	)

	with open(input_file,'r',encoding='utf-8') as f:
		data = json.load(f)

	index = faiss.IndexFlatL2(4096)
	query_list = []
	for i in tqdm.tqdm(range(len(data))):
		item = data[i]
		query = item['question']
		embed = get_emebd([query])
		index.add(embed.astype(np.float32))
		query_list.append(query)

	metadata = {
		'sentences': query_list,
		'dimension': 4096,
		'num_vectors': len(query_list)
	}

	faiss.write_index(index,index_file )

	with open(metadata_file, "wb") as f:
		pickle.dump(metadata, f)

	logger.info(f"Completed building the query vector index for the training data, a total of {len(query_list)} entries")

def build_column_index(mode):
	"""Encode the enhanced database schema columns, using the model fine-tuned during recall training"""

	model_base = "./models/recall_model/checkpoint-570"
	input_file = f"./data/BIRD/{mode}/{mode}_enhanced_column.json"
	index_file = f"./data/BIRD/database/{mode}_column_index.faiss"
	metadata_file = f"./data/BIRD/database/{mode}_column_metadata.pkl"

	def get_emebd(docs):
		with torch.no_grad():
			query_embeddings = model.encode(docs,prompt_name='document')
			norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
			return query_embeddings / norms

	model = SentenceTransformer(
		model_base,
		device=device,
		model_kwargs={"attn_implementation": "flash_attention_2"},
		processor_kwargs={"padding_side": "left"}
	)

	with open(input_file, 'r',encoding='utf-8') as f:
		data = json.load(f)
	logger.info(f"load {len(data)} enhanced columns")
	index = faiss.IndexFlatL2(4096)
	doc_list = []
	for key in data :
		doc = key +':' + data[key]
		embed = get_emebd([doc])
		index.add(embed.astype(np.float32))
		doc_list.append(key)

	metadata = {
		'sentences': doc_list,
		'dimension': 4096,
		'num_vectors': len(doc_list)
	}

	faiss.write_index(index, index_file)

	with open(metadata_file, "wb") as f:
		pickle.dump(metadata, f)

	logger.info(f"Completed building the vector index for the enhanced database columns, a total of {len(doc_list)} columns")

if __name__=="__main__":
	build_query_index()
	# build_column_index("train")


