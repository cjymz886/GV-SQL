"""
Obtain the DDL schema of the database along with its statistical information for subsequent use.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sqlite3
import argparse
import logging
from utils.my_logger import setup_logging


setup_logging(log_level=logging.INFO, log_file='gv_sql.log')
logger = logging.getLogger(__name__)


def get_table_schema(db_id, db_path, num_samples):
	conn = sqlite3.connect(db_path)
	cursor = conn.cursor()

	# 1. get all tables
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
	tables = cursor.fetchall()
	logger.info('Database [{}] has tables: {}'.format(db_path.split('/')[-1], tables))

	tab_dict = {}
	for (table_name,) in tables:
		# query sqlite_master to get schema
		cursor.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'")
		create_sql = cursor.fetchone()[0]
		if len(create_sql) > 10:
			tab_dict[table_name] = {'schema': create_sql}
		else:
			logger.warning('have errors: {}-{}'.format(db_id, db_path))
			continue

		# get column information
		cursor.execute(f"PRAGMA table_info(`{table_name}`);")
		columns_info = cursor.fetchall()

		column_names = []
		column_types = {}
		primary_keys = []

		for col_info in columns_info:
			col_id, col_name, col_type, not_null, default_val, is_pk = col_info
			column_names.append(col_name)
			column_types[col_name] = col_type
			if is_pk == 1:  # judge primary keys
				primary_keys.append(col_name)

		# get forigen keys
		cursor.execute(f"PRAGMA foreign_key_list(`{table_name}`);")
		foreign_keys = cursor.fetchall()

		foreign_key_info = []
		for fk in foreign_keys:
			# fk: (id, seq, table, from, to, on_update, on_delete, match)
			if len(fk) >= 5:
				fk_id, seq, ref_table, from_col, to_col = fk[:5]
				foreign_key_info.append({
					'from': f"{table_name}.{from_col}",
					'to': f"{ref_table}.{to_col}",
					'constraint': f"{table_name}.{from_col}={ref_table}.{to_col}"
				})

		# get samples
		col_dict = {}
		for column_name in column_names:
			query = f"""
			SELECT `{column_name}` 
			FROM (
				SELECT DISTINCT `{column_name}` 
				FROM `{table_name}` 
				WHERE `{column_name}` IS NOT NULL and `{column_name}` != ''
			) AS unique_values
			LIMIT {num_samples};
			"""
			cursor.execute(query)
			values = cursor.fetchall()
			values = [value[0] for value in values]

			for idx in range(len(values)):
				if isinstance(values[idx], str):
					values[idx] = values[idx][:48]

			if len(values) > 0:
				col_dict[column_name] = {
					'sample_values': values,
					'data_type': column_types.get(column_name, 'UNKNOWN')
				}

		# update 
		tab_dict[table_name].update({
			'column_value': col_dict,
			'primary_keys': primary_keys,
			'foreign_keys': foreign_key_info,
			'columns': column_names,
			'column_types': column_types
		})

	conn.close()
	return tab_dict

def parser():
	parser = argparse.ArgumentParser("")
	parser.add_argument('--input_file', type=str, default="/home/gz11778/windows-F/open_data/text2sql/BIRD/dev/dev_20240627/dev_tables.json")
	parser.add_argument('--db_root_path', type=str, default='/home/gz11778/windows-F/open_data/text2sql/BIRD/dev/dev_20240627/dev_databases')
	parser.add_argument('--output_file', type=str, default="/home/gz11778/windows-F/study/GV-SQL/data/BIRD/dev/dev_ddl_schema.json")
	parser.add_argument('--num_samples', type=int, default=5)
	args = parser.parse_args()
	return args


def main(args):
	with open(args.input_file,'r',encoding='utf-8') as f:
		data = json.load(f)

	logger.info("have {} Database".format(len(data)))
	D = {}
	for i,line in enumerate(data,1):
		db_id = line['db_id']
		db_path = os.path.join(args.db_root_path,db_id,f'{db_id}.sqlite')
		logger.info("******************processing {} Database [{}]********".format(str(i),db_id))
		res = get_table_schema(db_id,db_path, args.num_samples)
		D[db_id] = res
		break
	
	logger.info("have finished all Databases and save in {}".format(args.output_file))
	with open(args.output_file,'w',encoding='utf-8') as f:
		f.write(json.dumps(D,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
	args = parser()
	main(args)