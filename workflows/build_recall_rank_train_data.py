"Use the original data of the dataset to build the training and evaluation data for embedding and reranker"

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import random
import tqdm
import argparse
import logging
from utils.my_logger import setup_logging


setup_logging(log_level=logging.INFO, log_file='gv_sql.log')
logger = logging.getLogger(__name__)


def get_recall_neg_samples(db_id,sent_list,meaning_data,easy_num=6,medium_num=1,hard_num=1):
	"""Build the negative sampling strategy for the recall training data"""
	neg_list = []
	table_list = []
	for key in sent_list:
		kl = key.split('|')
		assert len(kl)==3
		tab = kl[1]
		if tab not in table_list:
			table_list.append(tab)

	key_list = list(meaning_data.keys())
	random.shuffle(key_list)

	# easy samples
	for key in key_list:
		kl = key.split('|')
		assert len(kl)==3
		if db_id != kl[0]:
			col_meaning = meaning_data[key].replace('#', '')
			text = key + ':' + col_meaning
			neg_list.append(text)
		if len(neg_list)==easy_num:
			break

	# medium samples
	for key in key_list:
		kl = key.split('|')
		assert len(kl) == 3
		if db_id == kl[0] and key not in sent_list and kl[1] not in table_list:
			col_meaning = meaning_data[key].replace('#', '')
			text = key + ':' + col_meaning
			neg_list.append(text)
		if len(neg_list) == easy_num + medium_num:
			break

	# hard samples
	for key in key_list:
		kl = key.split('|')
		assert len(kl) == 3
		if db_id == kl[0] and key not in sent_list and kl[1]  in table_list:
			col_meaning = meaning_data[key].replace('#', '')
			text = key + ':' + col_meaning
			neg_list.append(text)
		if len(neg_list) == easy_num + medium_num + hard_num:
			break

	random.shuffle(neg_list)
	return neg_list


def get_ranker_neg_samples(db_id,sent_list,meaning_data,easy_num=0,medium_num=6,hard_num=2):
	"""Build the negative sampling strategy for the reranker training data"""
	neg_list = []
	table_list = []
	for key in sent_list:
		kl = key.split('|')
		assert len(kl)==3
		tab = kl[1]
		if tab not in table_list:
			table_list.append(tab)

	key_list = list(meaning_data.keys())
	random.shuffle(key_list)

	# easy samples
	if easy_num > 0:
		for key in key_list:
			kl = key.split('|')
			assert len(kl) == 3
			if db_id != kl[0] and key not in sent_list:
				col_meaning = meaning_data[key]
				text = key + ':' + col_meaning
				neg_list.append(text)
			if len(neg_list) == easy_num:
				break

	# medium samples
	for key in key_list:
		kl = key.split('|')
		assert len(kl) == 3
		if db_id == kl[0] and key not in sent_list and kl[1] not in table_list:
			col_meaning = meaning_data[key]
			text = key + ':' + col_meaning
			neg_list.append(text)
		if len(neg_list) == easy_num + easy_num:
			break

	# hard samples
	for key in key_list:
		kl = key.split('|')
		assert len(kl) == 3
		if db_id == kl[0] and key not in sent_list and kl[1]  in table_list:
			col_meaning = meaning_data[key]
			text = key + ':' + col_meaning
			neg_list.append(text)
		if len(neg_list) == easy_num + easy_num + hard_num:
			break
	random.shuffle(neg_list)
	return neg_list


def build_recall_data(args):
	"""Build the training and evaluation data for the recall model.
	enhanced_column_file comes from the processing result of preprocessing/generate_enhanced_schema.py
	sql_recall_filter_file comes from preprocessing/use_llm_processing_data.py/identify_column_sql() function, which produces the sql_recall field information; then it is
	compared with the processing result of preprocessing/collect_ddl_schema.py to filter out the tables/columns appearing in the SQL statement that do not exist in enhanced_column, producing the sql_recall_filter field information;
	because in the BRID dataset the SQL statements may contain errors or inconsistencies: the tables and columns appearing in the statement may be inconsistent with those in the database.
	"""


	enhanced_column_file = f"{args.input_path}/{args.mode}/{args.mode}_enhanced_column.json"
	sql_recall_filter_file = f"{args.input_path}/{args.mode}/{args.mode}_sql_recall_filter.json"  
	output_file1 = f"{args.input_path}/{args.mode}/sql_recall_qwen_noenhanced_{args.mode}.json"
	output_file2 = f"{args.input_path}/{args.mode}/sql_recall_qwen_enhanced_{args.mode}.json"

	with open(enhanced_column_file, 'r', encoding='utf-8') as f:
		meaning_data = json.load(f)
	logger.info(f"load enhanced column: {len(meaning_data)} items")

	with open(sql_recall_filter_file, 'r', encoding='utf-8') as f:
		data = json.load(f)

	D1 = []
	D2 = []
	for i, line in enumerate(data):
		db_id = line['db_id']
		query = line['question']
		evidence = line['evidence']
		sql = line['SQL']
		sql_recall_filter = line['sql_recall_filter']

		flag = True
		sent_list1, sent_list2 = [], []
		for tab in sql_recall_filter:
			for col in sql_recall_filter[tab]:
				key = db_id + '|' + tab + '|' + col
				sent_list1.append(key)
				if key in meaning_data:
					col_meaning = meaning_data[key]
					text = key + ':' + col_meaning
					sent_list2.append(text)
				else:
					if '*' in key:
						text = key + ':' + "all the columns of this table"
						sent_list2.append(text)
					else:
						text = key + ':' + "one of columns of this table"
						sent_list2.append(text)
						logger.info(f"{key} not in {args.mode}_enhanced_column.json ")

		if sent_list2:
			#use the question and the enhanced column to build the training pair
			tmp = {}
			tmp['messages'] = [{'role': 'user', 'content': query}]
			pos_list = []
			for sen in sent_list2:
				pos_list.append([{'role': 'user', 'content': sen}])

			#negative sample sampling
			neg_sen_list = get_recall_neg_samples(db_id,sent_list1,meaning_data,6,1,1)
			neg_list = []
			for sen in neg_sen_list:
				neg_list.append([{'role': 'user', 'content': sen}])

			tmp['positive_messages'] = pos_list
			tmp['negative_messages'] = neg_list
			D2.append(tmp)

			# use the question and the original column to build the training pair, with the same negative samples as above, just without the enhanced content
			tmp = {}
			tmp['messages'] = [{'role': 'user', 'content': query}]
			pos_list = []
			for sen in sent_list1:
				pos_list.append([{'role': 'user', 'content': sen}])

			neg_list = []
			for key in neg_sen_list:
				sen = key.split(':')[0]
				neg_list.append([{'role': 'user', 'content': sen}])

			tmp['positive_messages'] = pos_list
			tmp['negative_messages'] = neg_list
			D1.append(tmp)



			##use the evidence and the enriched column to build the training pair
			tmp = {}
			tmp['messages'] = [{'role': 'user', 'content': evidence}]
			pos_list = []
			for sen in sent_list2:
				pos_list.append([{'role': 'user', 'content': sen}])

			neg_sen_list = get_recall_neg_samples(db_id, sent_list1,meaning_data,6,1,1)
			neg_list = []
			for sen in neg_sen_list:
				neg_list.append([{'role': 'user', 'content': sen}])

			tmp['positive_messages'] = pos_list
			tmp['negative_messages'] = neg_list
			D2.append(tmp)

			#use the evidence and the original column to build the training pair, with the same negative samples as above, just without the enhanced content
			tmp = {}
			tmp['messages'] = [{'role': 'user', 'content': evidence}]
			pos_list = []
			for sen in sent_list1:
				pos_list.append([{'role': 'user', 'content': sen}])

			neg_list = []
			for key in neg_sen_list:
				sen = key.split(':')[0]
				neg_list.append([{'role': 'user', 'content': sen}])

			tmp['positive_messages'] = pos_list
			tmp['negative_messages'] = neg_list
			D1.append(tmp)


	with open(output_file1, 'w', encoding='utf-8') as f:
		f.write(json.dumps(D1, ensure_ascii=False, indent=2) + '\n')

	with open(output_file2, 'w', encoding='utf-8') as f:
		f.write(json.dumps(D2, ensure_ascii=False, indent=2) + '\n')

	logger.info(f"Built {len(D1)} recall training records in original column form, stored in {output_file1}")
	logger.info(f"Built {len(D2)} recall training records in enhanced column form, stored in {output_file2}")



def build_reranker_data(args):
	"""Build the training and evaluation data for the reranker model.
	enhanced_column_file comes from the processing result of preprocessing/generate_enhanced_schema.py
	sql_recall_filter_file comes from preprocessing/use_llm_processing_data.py/identify_column_sql() function, which produces the sql_recall field information; then it is
	compared with the processing result of preprocessing/collect_ddl_schema.py to filter out the tables/columns appearing in the SQL statement that do not exist in enhanced_column, producing the sql_recall_filter field information;
	because in the BRID dataset the SQL statements may contain errors or inconsistencies: the tables and columns appearing in the statement may be inconsistent with those in the database.
	"""

	enhanced_column_file = f"{args.input_path}/{args.mode}/{args.mode}_enhanced_column.json"
	sql_recall_filter_file = f"{args.input_path}/{args.mode}/{args.mode}_sql_recall_filter.json"  # from preprocessing/use_llm_
	output_file1 = f"{args.input_path}/{args.mode}/sql_reranker_qwen_noenhanced_{args.mode}.json"
	output_file2 = f"{args.input_path}/{args.mode}/sql_reranker_qwen_enhanced_{args.mode}.json"

	with open(enhanced_column_file, 'r', encoding='utf-8') as f:
		meaning_data = json.load(f)
	logger.info(f"load enhanced column: {len(meaning_data)} items")


	with open(sql_recall_filter_file, 'r', encoding='utf-8') as f:
		data = json.load(f)

	D1 = []
	D2 = []
	k = 0
	for i, line in enumerate(data):
		db_id = line['db_id']
		query = line['question']
		sql = line['SQL']
		evidence = line['evidence']
		query = query + ' ' +evidence  # concatenate question and evidence as the overall query information
		sql_recall_filter = line['sql_recall_filter']

		sent_list1, sent_list2 = [], []
		for tab in sql_recall_filter:
			for col in sql_recall_filter[tab]:
				key = db_id + '|' + tab + '|' + col
				sent_list1.append(key)
				if key in meaning_data:
					col_meaning = meaning_data[key].replace('#', '')
					text = key + ':' + col_meaning
					sent_list2.append(text)
				else:
					if '*' in key:
						text = key + ':' + "all the columns of this table"
						sent_list2.append(text)
					else:
						sent_list2.append(text)

		if sent_list2:
			tmp = {}
			tmp['messages'] = [{'role': 'system', 'content': 'Given sql query, retrieve relevant columns that answer the query'},
								{'role': 'assistant', 'content': query}]
			pos_list = []
			for sen in sent_list2:
				pos_list.append([{'role': 'assistant', 'content': sen}])

			neg_sen_list = get_ranker_neg_samples(db_id,sent_list1,meaning_data,0,6,2)
			neg_list = []
			for sen in neg_sen_list:
				neg_list.append([{'role': 'user', 'content': sen}])

			tmp['positive_messages'] = pos_list
			tmp['negative_messages'] = neg_list
			D2.append(tmp)

			tmp = {}
			tmp['messages'] = [{'role': 'assistant', 'content': query}]
			pos_list = []
			for sen in sent_list1:
				pos_list.append([{'role': 'assistant', 'content': sen}])

			neg_list = []
			for key in neg_sen_list:
				sen = key.split(':')[0]
				neg_list.append([{'role': 'assistant', 'content': sen}])

			tmp['positive_messages'] = pos_list
			tmp['negative_messages'] = neg_list
			D1.append(tmp)




	with open(output_file1, 'w',encoding='utf-8') as f:
		f.write(json.dumps(D1, ensure_ascii=False, indent=2) + '\n')

	with open(output_file2, 'w',encoding='utf-8') as f:
		f.write(json.dumps(D2, ensure_ascii=False, indent=2) + '\n')

	logger.info(f"Built {len(D1)} reranker training records in original column form, stored in {output_file1}")
	logger.info(f"Built {len(D2)} reranker training records in enhanced column form, stored in {output_file2}")



def parser():
	parser = argparse.ArgumentParser("")
	parser.add_argument('--input_path', type=str, default="/home/gz11778/windows-F/study/GV-SQL/data/BIRD")
	parser.add_argument('--mode', type=str, default="dev")
	args = parser.parse_args()
	return args


if __name__=="__main__":

	args = parser()
	build_recall_data(args)
	build_reranker_data(args)
