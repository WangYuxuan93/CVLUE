import json
from tqdm import tqdm

result_file = 'VQA_result.json'

with open(result_file, 'r', encoding='utf-8') as f:
    string = f.read()
    results = json.loads(string)


gold_file = 'VQA_test_gold.json'

with open(gold_file, 'r', encoding='utf-8') as f:
    string = f.read()
    gold_data = json.loads(string)

gold_ans_map = {}
check_question = {}
for item in gold_data:
    gold_ans_map[item['question_id']] = item['answer']
    check_question[item['question_id']] = 0

num_ = 0
correct_num = 0
for item in tqdm(results):
    question_id = item['question_id']
    check_question[question_id] = 1
    pre_ = item['answer']
    gold_ = gold_ans_map[question_id]

    if str(pre_) == str(gold_):
        correct_num += 1
    num_ += 1

for question_id, flag in check_question.items():
    if flag != 1:
        print(question_id, ' not exists!')

print('Acc:', correct_num / num_)
