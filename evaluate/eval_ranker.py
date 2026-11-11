import json




def eval_recall(input_file,topk):

    def judge(db_id, gold, pred):
        t_num =0
        c_num = 0
        pred_list = [p[0].split(':')[0] for p in pred]
        for tab in gold:
            for col in gold[tab]:
                key = db_id + '|' + tab + '|' + col
                t_num +=1
                if key in pred_list:
                    c_num+=1
        return t_num,c_num

    with open(input_file,'r',encoding='utf-8') as f:
        data = json.load(f)

        total_num = 0
        correct_num = 0
        for line in data :
            db_id = line['db_id']
            sql_recall = line['sql_recall_filter']
            recall_res = line['rank_res']
            t_num,c_num = judge(db_id,sql_recall,recall_res[:topk])
            total_num += t_num
            correct_num += c_num

        return round(correct_num/total_num,4)



if __name__=="__main__":

    input_file = './data/bird/dev/predict/dev_sql_enrich_column_570_ranker_285.json'
    top_10 = eval_recall(input_file, 10)
    top_20 = eval_recall(input_file,20)
    top_30 = eval_recall(input_file,30)
    top_40 = eval_recall(input_file, 40)
    top_50 = eval_recall(input_file, 50)
    print('top_10 recall:',top_10)
    print('top_20 recall:', top_20)
    print('top_30 recall:', top_30)
    print('top_40 recall:', top_40)
    print('top_50 recall:', top_50)

