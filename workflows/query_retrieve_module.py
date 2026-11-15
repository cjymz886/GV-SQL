"""
This module retrieves similar queries (questions) from the training set, 
and uses the similar query data as few-shot training data.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import torch
import  faiss
from sentence_transformers import SentenceTransformer
import numpy as np


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda:0')



class RetrieveModule():
	def __init__(self,num_samples=5,threshold=0.88):
		self.model_base= "/data/models/Qwen3-Embedding-8B"
		self.query_index_path = "./data/BIRD/database/query_index.faiss"
		self.query_metadata_path = "./data/BIRD/database/query_metadata.pkl"
		self.load_index()
		self.load_model()
		self.topk = num_samples  #number of few-shot examples
		self.threshold = threshold
		self.is_empty = True #The retrieve results can be empty. If not allowed to be empty,the top 1 result will be returned.


	def load_model(self):
		self.embed_model = SentenceTransformer(
			self.model_base,
			device=device,
			model_kwargs={"attn_implementation": "flash_attention_2"},
			processor_kwargs={"padding_side": "left"}
		)

	def get_emebd(self,queries):
		with torch.no_grad():
			query_embeddings = self.embed_model.encode(queries, prompt_name="query")
			norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
			return query_embeddings / norms

	def load_index(self):
		self.query_index = faiss.read_index(self.query_index_path)
		with open(self.query_metadata_path, "rb") as f:
			query_metadata = pickle.load(f)

		self.query_list = query_metadata['sentences']
		self.query_map = {}
		for i in range(len(self.query_list)):
			q = self.query_list[i]
			self.query_map[q] = i

	def get_similarity_query(self,query):

		q_idx = self.query_map.get(query,-1)

		if q_idx!=-1:
			vector = self.query_index.reconstruct(q_idx)
		else:
			vector = self.get_emebd([query])

		if vector.ndim == 1:
			vector = vector.reshape(1, -1)

		scores, indices = self.query_index.search(vector.astype(np.float32), self.topk)
		res = []
		for idx, score in zip(indices[0], scores[0]):
			if idx == -1 or idx >= len(self.query_map):
				continue
			if idx == q_idx:
				continue
			if score > self.threshold:
				res.append(self.query_list[idx])
		if not res:
			if not self.is_empty:
				res.append(self.query_list[indices[0][0]])
		return res



if __name__=="__main__":
	rm = RetrieveModule()
	get_query_sim = rm.get_similarity_query
	q = "State the most popular movie? When was it released and who is the director for the movie?"
	res =get_query_sim(q)
	print(res)