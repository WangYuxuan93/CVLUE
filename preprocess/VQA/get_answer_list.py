import json

# 依据测试集与验证集，获取答案列表

val_path = 'VQA_val.json'
test_path = 'VQA_test.json'
answer_path = 'answer_list.json'


answer_list = []
with open(val_path, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)

for line in raw_data:
    answer = line['answer']
    if answer not in answer_list:
        answer_list.append(answer)

with open(test_path, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)

for line in raw_data:
    answer = line['answer']
    if answer not in answer_list:
        answer_list.append(answer)

with open(answer_path, 'w', encoding='utf-8') as f:
    json.dump(answer_list, f, ensure_ascii=False, indent=2)