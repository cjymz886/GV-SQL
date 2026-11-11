"""
Mainly includes:
1) Extract the keywords of the question from the data;
2) Identify the tables and columns contained in the SQL of the data;
3) Generate the corresponding thinking process for the SQL generation and refinement, used for model training.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from json_repair import repair_json
import tqdm
import argparse
import logging
from PIL import Image
import base64
import io
import time

from utils.my_logger import setup_logging
from utils.llm_api import openai_llm_api
from config.prompt_template import (identify_column_in_SQL_prompt,
									question_keywords_extract_prompt,
									sql_think_output_extract_prompt,
									sql_think_output_extract_with_vl_prompt,
									sql_refinement_think_output_extract_with_vl_prompt)


setup_logging()
logger = logging.getLogger(__name__)



def identify_column_sql(input_file,output_file):
	"""Use LLM to identify the tables and their corresponding columns contained in the SQL statement, for schema recall and ranking training and evaluation"""

	with open(input_file,'r',encoding='utf-8') as f:
		data = json.load(f)

	logger.info(f"Loaded {len(data)} records that need to be processed")

	D = []
	for i in tqdm.tqdm(range(len(data))):
		line = data[i]
		sql = line['SQL']
		prompt = identify_column_in_SQL_prompt.format(sql_input=sql)
		# logger.info(f"prompt:\n {prompt}")
		retry_num = 0
		while True:
			output = openai_llm_api(prompt=prompt,max_tokens=1024)
			if output and len(output)>2:
				output = repair_json(output, return_objects=True)
				line['sql_recall'] = output
				D.append(line)
				break
			else:
				retry_num +=1
			if retry_num >=3:
				line['sql_recall'] = ''
				D.append(line)
				logger.error(f"[{line}] Failed to identify columns in SQL after 3 retries")
				break
			time.sleep(1)

	success_num = len([line for line in D if line['column_recall']])
	logger.info(f"have finished {success_num} items to identify columns in SQL and {len(data)-success_num} items failed")
	with open(output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')



def extract_question_keywords(input_file,outputfile):
	"Use LLM to identify the keywords contained in the question, for subsequent retrieval use"

	def clearn_output(text):
		if ':' in text:
			text = text.split(':')[1]
		text = text.replace('Keywords','').replace('keywords','')
		out = []
		for kw in text.split('|'):
			if kw and kw not in out:
				out.append(kw)
		return out


	with open(input_file,'r',encoding='utf-8') as f:
		data = json.load(f)
	logger.info(f"Loaded {len(data)} records that need to be processed")

	D = []
	for i in tqdm.tqdm(range(len(data))):
		line = data[i]
		query = line['question']
		evidence = line['evidence']
		input = 'Question: ' + query + '\n' + 'Evidence: '+ evidence
		prompt = question_keywords_extract_prompt.format(input_paras=input)
		retry_num = 0
		while True:
			output = openai_llm_api(prompt=prompt,max_tokens=1024)
			if output and len(output)>2:
				output = clearn_output(output)
				line['keywords'] = output
				D.append(line)
				break
			else:
				retry_num +=1
			if retry_num >=3:
				line['keywords'] = []
				D.append(line)
				logger.error(f"[{line}] Failed to extract keywords in question after 3 retries")
				break
			time.sleep(1)

	success_num = len([line for line in D if line['keywords']])
	logger.info(f"have finished {success_num} items to extract question keywords and {len(data)-success_num} items failed")
	with open(output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')



def generate_think_output(input_file,output_file):
	"""Use LLM to generate the corresponding thinking process for the SQL in the original dataset"""

	p1 = re.compile(r"## Database Schema:(.*?)This markdown schema describe")
	with open(input_file,'r',encoding='utf-8') as f:
		data = json.load(f)
	logger.info(f"Loaded {len(data)} records that need to be processed")

	D = []
	for i in tqdm.tqdm(range(len(data))):
		line = data[i]
		think_prompt = line['messages'][0]['content']
		think_prompt = think_prompt.replace('\n','&&')
		schema = p1.findall(think_prompt)[0].strip()
		schema = schema.replace('&&','\n').strip()
		item = line['data']
		db_id = item['db_id']
		query = item['question']
		evidence = item['evidence']
		sql = item['SQL']
		prompt = sql_think_output_extract_prompt.format(schema=schema,
														question=query,
														evidence=evidence,
														sql=sql)
		# logger.info(f"prompt:\n {prompt}")
		retry_num = 0
		while True:
			output = openai_llm_api(prompt=prompt,max_tokens=2048)
			if output and len(output)>2:
				line['think_output'] = output
				D.append(line)
				break
			else:
				retry_num +=1
			if retry_num >=3:
				line['think_output'] = ''
				D.append(line)
				logger.error(f"[{line}] Failed to generate think output after 3 retries")
				break
			time.sleep(1)

	success_num = len([line for line in D if line['think_output']])
	logger.info(f"have finished {success_num} item to generate think output and {len(data)-success_num} item failed")
	with open(output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')


def encode_image(image_path, max_pixels=2097152):
	"Preprocess the input image"
	with Image.open(image_path) as img:
		width, height = img.size
		current_pixels = width * height

		if current_pixels > max_pixels:
			scale = (max_pixels / current_pixels) ** 0.5
			new_width = int(width * scale)
			new_height = int(height * scale)
			img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

		img_byte_arr = io.BytesIO()
		img_format = img.format if img.format else 'PNG'
		img.save(img_byte_arr, format=img_format, quality=95)
		img_byte_arr = img_byte_arr.getvalue()
	return base64.b64encode(img_byte_arr).decode('utf-8')


def generate_think_output_with_vl(input_file,output_file):
	"Based on adding visual information, use MLLM to generate the corresponding thinking process for the SQL in the original dataset"

	p1 = re.compile(r"## Database Schema:(.*?)The input image is a ddl schema")

	with open(input_file,'r',encoding='utf-8') as f:
		data = json.load(f)
	logger.info(f"Loaded {len(data)} records that need to be processed")

	D = []
	for i in tqdm.tqdm(range(len(data[:5]))):
		line = data[i]
		think_prompt = line['messages'][0]['content']
		think_prompt = think_prompt.replace('\n','&&')
		schema = p1.findall(think_prompt)[0].strip()
		schema = schema.replace('&&','\n').strip()
		# print(schema)
		item = line['data']
		db_id = item['db_id']
		query = item['question']
		evidence = item['evidence']
		sql = item['SQL']

		prompt = sql_think_output_extract_with_vl_prompt.format(schema=schema,
																	question=query,
																	evidence=evidence,
																	sql=sql)
		# logger.info(f"prompt:\n {prompt}")

		input_list = []
		for image in  line['images']:
			image_path = './GV_images/dev/' + image.split('/')[-1]
			imgbase64 = encode_image(image_path)
			input_list.append({'type':'image_url',
							   'image_url': {'url': f'data:image/png;base64,{imgbase64}'}})

		input_list.append({'type': 'text', 'text': prompt})
		msg = [{ 'role': 'user','content':input_list}]

		retry_num = 0
		while True:
			output = openai_llm_api(prompt=None,msg=msg,max_tokens=2048)
			if output and len(output)>2:
				line['think_output'] = output
				# logger.info(f"output:\n {output}")
				D.append(line)
				break
			else:
				retry_num +=1
			if retry_num >=3:
				line['think_output'] = ''
				D.append(line)
				logger.error(f"[{line}] Failed to generate think output after 3 retries")
				break
			time.sleep(1)

	success_num = len([line for line in D if line['think_output']])
	logger.info(f"have finished {success_num} item to generate think output and {len(data)-success_num} item failed")
	with open(output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')


def generate_refinement_think_output_with_vl(input_file,output_file):
	"Based on adding images, generate the corresponding thinking process for the sql refinement data"

	p1 = re.compile(r"## Database Schema:(.*?)The input images and this markdown schema")
	p2 = re.compile(r"SQL:(.*?)Error clue:")
	p3 = re.compile(r"Error clue:(.*?)## Instructions:")

	with open(input_file,'r',encoding='utf-8') as f:
		data = json.load(f)
	logger.info(f"Loaded {len(data)} records that need to be processed")

	D = []
	for i in tqdm.tqdm(range(len(data[:10]))):
		line = data[i]
		org_prompt = line['messages'][0]['content']
		org_prompt = org_prompt.replace('\n','&&')
		schema = p1.findall(org_prompt)[0].strip()
		schema = schema.replace('&&','\n').strip()

		pred_sql = p2.findall(org_prompt)[0].strip()
		pred_sql = pred_sql.replace('&&', '\n').strip()

		error_msg = p3.findall(org_prompt)[0].strip()
		error_msg = error_msg.replace('&&', '\n').strip()


		item = line['data']
		db_id = item['db_id']
		query = item['question']
		evidence = item['evidence']
		sql = item['SQL']
		prompt = sql_refinement_think_output_extract_with_vl_prompt.format(schema=schema,
																		question=query,
																		evidence=evidence,
																		gold_sql=sql,
																		pred_sql=pred_sql,
																		ERROR = error_msg
																			)
		logger.info(f"prompt:\n {prompt}")

		input_list = []
		for image in  line['images']:
			image_path = './GV_images/train/' + image.split('/')[-1]
			imgbase64 = encode_image(image_path)
			input_list.append({'type':'image_url',
							   'image_url': {'url': f'data:image/png;base64,{imgbase64}'}})

		input_list.append({'type': 'text', 'text': prompt})
		msg = [{ 'role': 'user','content':input_list}]

		retry_num = 0
		while True:
			output = openai_llm_api(prompt=None,msg=msg,max_tokens=2048)
			if output and len(output)>2:
				line['think_output'] = output
				logger.info(f"output:\n {output}")
				D.append(line)
				break
			else:
				retry_num +=1
			if retry_num >=3:
				line['think_output'] = ''
				D.append(line)
				logger.error(f"[{line}] Failed to generate think output after 3 retries")
				break
			time.sleep(1)

	success_num = len([line for line in D if line['think_output']])
	logger.info(f"have finished {success_num} item to generate think output and {len(data)-success_num} item failed")
	with open(output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')





if __name__ == '__main__':

	#########################identify_column_sql##################
	input_file = "/home/gz11778/windows-F/open_data/text2sql/BIRD/dev/dev_20240627/dev.json"
	output_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev/dev_sql_recall.json"
	identify_column_sql(input_file,output_file)
	#########################identify_column_sql##################


	#########################extract question keywords#############
	# input_file = "/home/gz11778/windows-F/open_data/text2sql/BIRD/dev/dev_20240627/dev.json"
	# output_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev/dev_sql_keywords.json"
	# extract_question_keywords(input_file,output_file)
	#########################extract question keywords#############


	#########################generate think output##################
	# input_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev/sql_qwen_recall_top30_dev.json"
	# output_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev/sql_qwen_recall_top30_think.devjson"
	# generate_think_output(input_file,output_file)
	#########################generate think output##################


	###################generate think output with images#############
	# input_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev/sql_qwen_recall_top30_vl_dev.json"
	# output_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev/sql_qwen_recall_top30_vl_think_dev.json"
	# generate_think_output_with_vl(input_file,output_file)
	###################generate think output with images#############

	##############generate refinement think output with images#######
	# input_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/train/sql_qwen_top60_refinement_vl_train_sample.json"
	# output_file = "/home/gz11778/windows-F/study/GV-SQL/data/BIRD/train/sql_qwen_top60_refinement_vl_think_train_sample.json"
	# generate_refinement_think_output_with_vl(input_file,output_file)
	##############generate_refinement_think_output_with_vl#######
