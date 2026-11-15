import re
from typing import List
import sqlite3
from swift.rewards import ORM, orms


class SQLAccuracy(ORM):

    @staticmethod
    def extract_answer(text: str) -> str:
        p1 = re.compile("```(.*?)```")
        if p1.findall(text):
            sql = p1.findall(text)[0]
        else:
            sql = ""
        return sql

    @staticmethod
    def execute_sql(predicted_sql, ground_truth, db_id):
        db_path = f'/data/cjy/mystudy/sql_recall_rank/data/BIRD/train/train_databases/{db_id}/{db_id}.sqlite'
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(predicted_sql)
        predicted_res = cursor.fetchall()
        cursor.execute(ground_truth)
        ground_truth_res = cursor.fetchall()
        res = 0
        if set(predicted_res) == set(ground_truth_res):
            res = 1
        return res


    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        data = kwargs['data']
        # print(data)
        for completion, item in zip(completions, data):
            db_id = item['db_id']
            gt_sql = item['SQL']
            pred_sql = self.extract_answer(completion)
            correct = 0
            if gt_sql and pred_sql:
                try:
                    correct = self.execute_sql(gt_sql,pred_sql,db_id)
                except Exception as e:
                    correct = 0
            rewards.append(1.0 if correct==1 else 0.0)
        return rewards


class SQLFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            has_answer = bool(re.search(r"```(.*?)```", completion))
            rewards.append(1.0 if has_answer else 0.0)
        return rewards


orms['sql_accuracy'] = SQLAccuracy
orms['sql_format'] = SQLFormat