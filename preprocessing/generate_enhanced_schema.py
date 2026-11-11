"""
1. Mainly expand the description of each column via LLM to enhance its semantic expression.
2. This program targets the open-source SQL BIRD dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import argparse
import logging
import tqdm
import time
import csv
from utils.my_logger import setup_logging
from utils.llm_api import openai_llm_api
from config.prompt_template import schema_enhanced_prompt


setup_logging(log_level=logging.INFO, log_file='gv_sql.log')
logger = logging.getLogger(__name__)


def get_prompts_each_db(db_root_path,table_json,mode):
	"""First obtain the enhancement instruction for each column in a database, then request the LLM api"""

	#################################Main processing#######################
	prompt_dict = {}
	for i in tqdm.tqdm(range(len(table_json))):
		table_info = table_json[i]
		db_id = table_info['db_id']

		#Connect to the database via sqlite3, used later to execute query sql
		db_path = os.path.join(db_root_path,f'{mode}_databases', db_id, f'{db_id}.sqlite')
		conn = sqlite3.connect(db_path)
		cursor = conn.cursor()

		#Read the table and column information in the database, which is provided in advance for this dataset; alternatively, it can be fetched by querying the data directly
		csv_dir = os.path.join(db_root_path,f'{mode}_databases',db_id,'database_description')
		otn_list = table_info['table_names_original']  #the original table names in the database
		tn_list = table_info['table_names']            #the table aliases

		for i, otn in enumerate(otn_list):
			csv_path = os.path.join(os.path.join(csv_dir, f"{otn}.csv"))
			csv_dict = csv.DictReader(open(csv_path, newline='', encoding="latin1"))
			alias_otn = tn_list[i]
			for row in csv_dict:

				#Read the column information from the csv
				org_column_name = row['original_column_name'].strip()
				alias_column_name = row['column_name']
				column_description = row['column_description'].strip()
				column_type = row['data_format'].strip()
				value_description = str(row['value_description']).strip()

				if org_column_name in ['',' ']:
				   continue
				key = f'{db_id}|{otn}|{org_column_name}'

				#If the column description and value description equal the name itself, set them to empty to avoid redundant semantics
				column_description = None if (column_description in ['',' '] or column_description == org_column_name or column_description == alias_column_name) else column_description
				value_description = None if (value_description in ['',' '] or value_description == org_column_name or value_description == alias_column_name) else value_description
				column_type = None if column_type in ['', ' '] else column_type
				alias_otn = alias_otn if (alias_otn!= otn and alias_otn not in ['',' ']) else None

				#Collect the various meta information of the column
				input_paras = f"database_id='{db_id}', table_name='{otn}', column_name='{org_column_name}', column_type='{column_type}'"
				if alias_otn:
					input_paras += f", table_alias_name='{alias_otn}'"
				if alias_column_name:
					input_paras += f", column_alias_name='{alias_column_name}'"
				if column_description:
					input_paras += f", column_description='{column_description}'"
				if value_description:
					input_paras += f", value_description='{value_description}'"

				#Get the specific values of the column; if there are fewer than 10 distinct values, show all of them; otherwise sample 5 to display
				if column_type in  ['text', 'date', 'datetime','integer','real']:
					sql = f'''SELECT DISTINCT "{org_column_name}" FROM `{otn}` where "{org_column_name}" IS NOT NULL ORDER BY RANDOM()'''
					cursor.execute(sql)
					values = cursor.fetchall()
					if len(values) > 0:
						if len(values) <= 10:
							all_possible_values = [v[0] for v in values]
							example_values = None
						else:
							all_possible_values = None
							example_values = [v[0] for v in values[:5]]

					if example_values:
						input_paras += f', example_values = {example_values}'
					if all_possible_values:
						input_paras += f', all_possible_values = {all_possible_values}'

				prompt_dict[key] = schema_enhanced_prompt.format(input_paras = input_paras)
				# with open(output_file,'a',encoding='utf-8') as f:
				# 	f.write(json.dumps(tmp,ensure_ascii=False)+'\n')
		logger.info(f"Obtained a total of {len(prompt_dict)} column enhancement instructions")

	return prompt_dict


def parser():
	parser = argparse.ArgumentParser("")
	parser.add_argument('--db_root_path', type=str, default="/home/gz11778/windows-F/open_data/text2sql/BIRD/dev/dev_20240627")
	parser.add_argument('--mode', type=str, default='dev')
	parser.add_argument('--output_path', type=str, default="/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev")
	args = parser.parse_args()
	return args



def main(args):

	output_enrich_column_file = os.path.join(args.output_path,f'{args.mode}_enhanced_column.jsonl')

	#Read all database information in the dataset and generate the corresponding column enhancement instruction data
	logger.info("#########################step1: executing the column enhancement instruction generation task############")
	table_json_path = os.path.join(args.db_root_path, f'{args.mode}_tables.json')
	table_json = json.load(open(table_json_path, 'r'))
	prompt_dict = get_prompts_each_db(args.db_root_path,table_json,args.mode)


	logger.info("#########################step2: calling the LLM to perform the enhancement generation task############")
	output_dict = {}
	data = list(prompt_dict.items())
	for i in tqdm.tqdm(range(len(data))):
		key,prompt = data[i]
		retry_num = 0
		while True:
			output = openai_llm_api(prompt,max_tokens=1024)
			if output and len(output)>10:
				output_dict[key] = output
				break
			else:
				retry_num += 1
			if retry_num >= 3:
				output_dict[key] = ''
				logger.error(f"[{key}] Failed to enrich after 3 retries")
				break
			time.sleep(1)
		
	success_num = len([ r for r in output_dict if output_dict[r]])
	logger.info(f"have finished {success_num} columns to enrich by llm, and {len(prompt_dict)-success_num} columns failed")
	with open(output_enrich_column_file, 'a',encoding='utf-8') as f:
		f.write(json.dumps(output_dict,ensure_ascii=False,indent=2)+'\n')



if __name__ == '__main__':
	opt = parser()
	main(opt)
