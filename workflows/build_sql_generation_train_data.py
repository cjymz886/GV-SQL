"""
Build SQL generation training data.
Use schema_retrieve_module for schema retrieval.
Use query_retrieve_module to generate few-shot samples.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
import tqdm
import logging
import argparse

from workflows.schema_retrieve_module import *
from workflows.query_retrieve_module import *
from config.prompt_template import sql_generation_prompt,sql_generation_with_vl_prompt
from utils.my_logger import setup_logging

setup_logging(log_level=logging.INFO, log_file='gv_sql.log')
logger = logging.getLogger(__name__)


def convert_few_shot(query_map,res):
	"Get few-shot results and convert them to the format required by the prompt"
	content = ''
	for q in res:
		q_item = query_map.get(q,{})
		if q_item:
			sql = q_item['sql']
			evidence = q_item['evidence']
			content += 'question: ' +q +'\n'
			content += 'evidence: ' + evidence + '\n'
			content += 'SQL: ' + sql + '\n\n'
	return content


def get_gv_image(db_id, image_path):
	"Retrieve the GV visual image information by database id"
	file_list = []
	rel_list = []
	for file in os.listdir(image_path):
		file_path = os.path.join(image_path, file)
		if '_part' in file:
			file_name = file.split('_part')[0]
			if file_name == db_id :
				file_list.append(file_path)
		else:
			file_name = file.replace('.png', '').strip()
			if file_name == db_id:
				rel_list.append(file_path)
	if not file_list or not rel_list:
		logger.error(f"There is a problem with the GV image of {db_id}!")
	return file_list + rel_list


def build_genration_data(args):
	"Build SQL generation training data"

	script_dir = os.path.dirname(os.path.abspath(__file__))

	# Load the schema retrieval module
	srm = SQLRecllModule(mode=args.mode, topk=args.topk)
	get_retrieve_schema = srm.get_retrieve_result

	# Load the query (question) retrieval module
	rm = RetrieveModule()
	get_query_sim = rm.get_similarity_query

	input_file = f"./data/BIRD/{args.mode}/{args.mode}.json"

	# Load the dataset
	with open(input_file,'r',encoding='utf-8') as f:
		data = json.load(f)
	logger.info(f"Loaded the {args.mode} dataset, {len(data)} samples in total")


	# Used for few-shot retrieval
	query_map = {}
	for item in data:
		query = item['question']
		evidence = item['evidence']
		sql = item['SQL']
		query_map[query]= {'sql':sql,'evidence':evidence}

	D = []
	for i in tqdm.tqdm(range(len(data))):
		item = data[i]
		db_id = item['db_id']
		query = item['question']
		evidence = item['evidence'].strip()
		sql = item['SQL']
		output = "```" + sql + "```"
		dialect = "sqlite"

		schema = get_retrieve_schema(item)
		if len(schema) <10:
			raise ValueError(f"The schema of the {db_id} database has a problem")

		few_shot = get_query_sim(query)
		few_shot = convert_few_shot(query_map,few_shot)


		if args.use_vl:
			image_path =  os.path.normpath(os.path.join(script_dir,'..', f"GV_images/{args.mode}"))
			image_file = get_gv_image(db_id,image_path)

			input_image = ''.join(['<image>'] * len(image_file))
			prompt = sql_generation_with_vl_prompt.format(
				input_image=input_image,
				Dialect=dialect,
				Database=db_id,
				Schema=schema,
				Examples=few_shot,
				question=query,
				evidence=evidence
			)
		else:
			prompt =sql_generation_prompt.format(
								  Dialect=dialect,
								  Database=db_id,
								  Schema=schema,
								  Examples=few_shot,
								  question=query,
								  evidence=evidence
								  )

		tmp = []
		tmp.append({"role": "user", "content": prompt})
		tmp.append({"role": "assistant", "content": output})

		if args.use_vl:
			D.append({'messages': tmp, 'data': item, 'images': image_file})
		else:
			D.append({'messages':tmp,'data':item})

	random.shuffle(D)

	if args.use_vl:
		prefix = f"top{args.topk}_vl"
	else:
		prefix = f"top{args.topk}"
	output_file = f"./data/{args.mode}_data/sql_qwen_{prefix}_{args.mode}.json"

	with open(output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D, ensure_ascii=False, indent=2) + '\n')

	logger.info(f"Generated {len(D)} samples and saved them to {output_file}")


def parser():
	parser = argparse.ArgumentParser("")
	parser.add_argument('--mode', type=str, default="train")
	parser.add_argument('--use_vl', type=bool, default=True)
	parser.add_argument('--topk', type=int, default=30)
	args = parser.parse_args()
	return args


if __name__=="__main__":
	args = parser()
	build_genration_data(args)
