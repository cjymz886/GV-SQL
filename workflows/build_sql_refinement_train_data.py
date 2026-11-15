"""
1) Build training data for sql refinement;
2) Idea: select some models trained in the sql generation stage, predict on the training data to get predicted_sql, use the database engine to execute and analyze predict_sql, and collect some error information to build sql repair training data;
3) On top of this, GV visual data can be added, and think-mode training data can also be generated accordingly (using the preprocessing.use_llm_processing_data.generate_refinement_think_output_with_vl function);
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
from func_timeout import func_timeout, FunctionTimedOut
import re
import tqdm
import logging


from config.prompt_template import sql_refinement_prompt
from utils.my_logger import setup_logging

setup_logging(log_level=logging.INFO, log_file='gv_sql.log')
logger = logging.getLogger(__name__)


def execute_sql(predicted_sql,gold_sql,db_path):
	"execute the SQL query"

	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()
	cursor.execute(predicted_sql)
	predicted_res = cursor.fetchall()
	cursor.execute(gold_sql)
	gold_res = cursor.fetchall()
	status = 0
	#If gold_sql is known in advance, it can be used, but in reality it is unknown;
	if set(predicted_res) == set(gold_res):
		status = 1
	return status,predicted_res

def execute_analyse_model(predicted_sql,gold_sql,db_path, meta_time_out):
	"Analyze and classify the execution results"
	msg = ''
	status = 0
	try:
		status,pred_res = func_timeout(meta_time_out, execute_sql,
								  args=(predicted_sql, gold_sql, db_path))
		if len(pred_res) <= 0 and status==0:
			msg = f"The execution result is empty."
			status = 0
	except KeyboardInterrupt:
		sys.exit(0)
	except FunctionTimedOut:
		msg = "The SQL execution has timed out . The SQL may be incorrect , or too complex , or the join table query may be unreasonable ."
		status = 0
	except Exception as e:
		msg = f"An error occurred during the execution of the SQL: {e}"
		status = 0

	if status ==0 and len(msg)<=0 and len(pred_res)>0:
		if len(pred_res) >=10:
			pred_res = pred_res[:5]
		msg = f"Although the SQL executed without any issues, the result :{pred_res} , is incorrect ."
	if status ==1:
		msg = "The execution result is correct."
	logger.info(f"status:{status},msg:{msg}")
	return msg


def build_sql_refinement_data(pred_file_list,input_file,output_file,db_root_path):
	"""
	pred_file_list: the results of prediction on the training dataset; the predicted sql is put into the pred_sql field
	input_file: the training data of sql generation, mainly using its retrieved schema information
	db_root_path: the train-type database path of the dataset, used to access the database files for sql execution queries
	output_file: the output file
	"""
	p1 = re.compile(r"## Database Schema:(.*?)This markdown schema describe")

	sql_map = {}
	for file in pred_file_list:
		with open(file,'r',encoding='utf-8') as f:
			data = json.load(f)
			for item in data:
				pred_sql = item['pred_sql']
				gold_sql = item['gold_sql']
				if gold_sql not in sql_map:
					sql_map[gold_sql] = [pred_sql]
				else:
					if pred_sql not in sql_map[gold_sql]:
						sql_map[gold_sql].append(pred_sql)

	logger.info(f"The number of data entries with prediction errors in the training data is: {len(sql_map)}")

	with open(input_file,'r',encoding='utf-8') as f:
		train_data = json.load(f)

	D = []
	for i in tqdm.tqdm(range(len(train_data))):
		line = train_data[i]
		dialect = 'sqlite'
		org_prompt = line['messages'][0]['content']
		# logger.info(f"org_prompt: {org_prompt}")
		output = line['messages'][1]['content']
		org_prompt = org_prompt.replace('\n', '&&')
		schema = p1.findall(org_prompt)[0].strip()
		schema = schema.replace('&&', '\n').strip()
		schema = '\n'.join(schema.split('\n')[1:])

		db_id = line['data']['db_id']
		query = line['data']['question']
		evidence = line['data']['evidence'].strip()
		gold_sql = line['data']['SQL']

		pred_list = sql_map.get(gold_sql,[])

		for pred_sql in pred_list:
			db_path = db_root_path + db_id + '/' + db_id + '.sqlite'
			msg = execute_analyse_model(pred_sql,gold_sql,db_path,meta_time_out=30)
			if msg !="The execution result is correct." :
				prompt =sql_refinement_prompt.format(
										  Dialect=dialect,
										  Database=db_id,
										  Schema=schema,
										  question=query,
										  evidence=evidence,
										  SQL=pred_sql,
										  ERROR=msg
										  )
				# logger.info(f"prompt: {prompt}")
				tmp = []
				tmp.append({"role": "user", "content": prompt})
				tmp.append({"role": "assistant", "content": output})
				D.append({'messages': tmp, 'data': line['data']})

	with open(output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D, ensure_ascii=False, indent=2) + '\n')





if __name__=="__main__":

	pred_file_list = ['./data/BIRD/train/predict/sql_qwen_train_recall_top50_3142.json',
					  './data/BIRD/train/predict/sql_qwen_train_lora_10000_3142.json',
					  './data/BIRD/train/predict/sql_qwen_train_recall_top20_vl_1178.json',
					  './data/BIRD/train/predict/sql_qwen_train_recall_top30_vl_1178.json'
					  ]

	input_file = "./data/BIRD/train/sql_qwen_recall_top60_train.json"
	db_root_path = "/home/gz11778/windows-F/open_data/text2sql/BIRD/train/train_databases/"
	output_file =  "./data/BIRD/train/sql_qwen_top60_refinement_train.json"


	build_sql_refinement_data(pred_file_list,input_file,output_file,db_root_path)





