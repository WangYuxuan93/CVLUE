import json

# 将训练集、验证集和测试集中，问题和答案转化为 id 的表示

raw_train_path = 'new_train_ch.json'
new_train_path = 'last_train_ch.json'
raw_test_path = 'new_test_ch.json'
new_test_path = 'last_test_ch.json'
raw_val_path = 'new_val_ch.json'
new_val_path = 'last_val_ch.json'

question_list_path = 'vd_question.json'
answer_list_path = 'vd_answer.json'

with open(question_list_path, 'r', encoding='utf-8') as f:
    string = f.read()
    question_list = json.loads(string)

new_question_list = {v:k for k,v in question_list.items()}

with open(answer_list_path, 'r', encoding='utf-8') as f:
    string = f.read()
    answer_list = json.loads(string)

new_answer_list = {v:k for k,v in answer_list.items()}

with open(raw_train_path, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_train = json.loads(string)
new_train = []
for line in raw_train:
    new_line = {}
    new_line['image_id'] = line['image_id']
    new_line['image'] = line['image']
    new_line['caption'] = line['caption']
    new_line['dialog_id'] = line['dialog_id']
    new_line['question'] = new_question_list[line['question']]
    new_line['answer_options'] = []
    for raw_answer in line['answer_options']:
        new_line['answer_options'].append(new_answer_list[raw_answer])
    new_line['answer'] = new_answer_list[line['answer']]
    new_train.append(new_line)
with open(new_train_path, 'w', encoding='utf-8') as f:
    json.dump(new_train, f, ensure_ascii=False, indent=2)

with open(raw_test_path, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_test = json.loads(string)
new_test = []
for line in raw_test:
    new_line = {}
    new_line['image_id'] = line['image_id']
    new_line['image'] = line['image']
    new_line['caption'] = line['caption']
    new_line['dialog_id'] = line['dialog_id']
    new_line['question'] = new_question_list[line['question']]
    new_line['answer_options'] = []
    for raw_answer in line['answer_options']:
        new_line['answer_options'].append(new_answer_list[raw_answer])
    new_line['answer'] = new_answer_list[line['answer']]
    new_test.append(new_line)
with open(new_test_path, 'w', encoding='utf-8') as f:
    json.dump(new_test, f, ensure_ascii=False, indent=2)

with open(raw_val_path, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_val = json.loads(string)
new_val = []
for line in raw_val:
    new_line = {}
    new_line['image_id'] = line['image_id']
    new_line['image'] = line['image']
    new_line['caption'] = line['caption']
    new_line['dialog_id'] = line['dialog_id']
    new_line['question'] = new_question_list[line['question']]
    new_line['answer_options'] = []
    for raw_answer in line['answer_options']:
        new_line['answer_options'].append(new_answer_list[raw_answer])
    new_line['answer'] = new_answer_list[line['answer']]
    new_val.append(new_line)
with open(new_val_path, 'w', encoding='utf-8') as f:
    json.dump(new_val, f, ensure_ascii=False, indent=2)


