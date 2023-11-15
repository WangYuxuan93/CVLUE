import json
import numpy as np
import copy
import random
from tqdm import tqdm

# 答案选项构造
# 根据问题相似度、答案出现频率和随机选取的方法构建测试集与验证集的答案候选列表

train_source_path = 'VD_train.json'

# val_source_path = 'VD_val.json'
# souce_file = 'VD_test.json'
# result_file = 'new_VD_test.json'

val_source_path = 'VD_test.json'
souce_file = 'VD_val.json'
result_file = 'new_VD_val.json'

answer_list_file = 'vd_answer.json'
answer_cnt_file = 'vd_answer_cnt.json'

question_hid_file = 'vd_question_hid.json'

with open(question_hid_file, 'r', encoding='utf-8') as f:
    string = f.read()
    question_hidden_states = json.loads(string)
hid_matrix = np.mat(question_hidden_states)
sim_matrix = np.dot(hid_matrix, hid_matrix.T)
sim_list = sim_matrix.tolist()

print('sim_matrix_size: ', sim_matrix.shape)

with open(answer_list_file, 'r', encoding='utf-8') as f:
    string = f.read()
    answer_list = json.loads(string)

answer2id = {v:k for k,v in answer_list.items()}

with open(answer_cnt_file, 'r', encoding='utf-8') as f:
    string = f.read()
    answer_cnt = json.loads(string)
answer_cnt_tuple = sorted(answer_cnt.items(), key=lambda x:x[1], reverse=True)

top_30_answer = []
for i in range(30):
    top_30_answer.append(int(answer2id[answer_cnt_tuple[i][0]]))


with open(souce_file, 'r', encoding='utf-8') as f:
    string = f.read()
    source = json.loads(string)

with open(train_source_path, 'r', encoding='utf-8') as f:
    string = f.read()
    train_source = json.loads(string)

with open(val_source_path, 'r', encoding='utf-8') as f:
    string = f.read()
    val_source = json.loads(string)

question2dialog = {}
for line in source:
    question = line['question']
    image_id = line['image_id']
    answer = line['answer']
    if question not in question2dialog:
        question2dialog[question] = {}
        question2dialog[question]['answer'] = []
        question2dialog[question]['image_id'] = []
    question2dialog[question]['image_id'].append(image_id)
    question2dialog[question]['answer'].append(answer)

for line in train_source:
    question = line['question']
    image_id = line['image_id']
    answer = line['answer']
    if question not in question2dialog:
        question2dialog[question] = {}
        question2dialog[question]['answer'] = []
        question2dialog[question]['image_id'] = []
    question2dialog[question]['image_id'].append(image_id)
    question2dialog[question]['answer'].append(answer)

for line in val_source:
    question = line['question']
    image_id = line['image_id']
    answer = line['answer']
    if question not in question2dialog:
        question2dialog[question] = {}
        question2dialog[question]['answer'] = []
        question2dialog[question]['image_id'] = []
    question2dialog[question]['image_id'].append(image_id)
    question2dialog[question]['answer'].append(answer)

result = []
for i in tqdm(range(len(source))):
    line = source[i]
    temp_line = {}
    image_id = line['image_id']
    temp_line['image_id'] = image_id
    temp_line['image'] = line['image']
    temp_line['caption'] = line['caption']
    temp_line['dialog_id'] = line['dialog_id']
    question = line['question']
    temp_line['question'] = question
    temp_line['answer_options'] = []
    temp_line['answer_options'].append(line['answer'])
    temp_line['answer'] = line['answer']
    temp_sim = copy.deepcopy(sim_list[question])
    temp_sim.sort(reverse=True)

    temp_sim_A_num = 0
    for sim in temp_sim:
        temp_sim_A_num += 1
        sim_index = sim_list[question].index(sim)
        # print('sim_index: ', sim_index)
        if sim_index == question:
            continue
        if image_id in question2dialog[sim_index]['image_id']:
            continue
        new_answer = random.choice(question2dialog[sim_index]['answer'])
        # print('new_answer: ', new_answer)
        temp_line['answer_options'].append(int(new_answer))
    
        if temp_sim_A_num >= 50:
            break
    
    temp_line['answer_options'] += top_30_answer
    temp_line['answer_options'] = list(set(temp_line['answer_options']))
    # print('len(temp_line[answer_options]): ', len(temp_line['answer_options']))
    j = 0
    tg_len = 100 - len(temp_line['answer_options'])
    # print('tg_len: ', tg_len)
    while j < tg_len:
        new_index = random.randint(0, len(answer_list)-1)
        if new_index not in temp_line['answer_options']:
            temp_line['answer_options'].append(new_index)
            j += 1

    if len(temp_line['answer_options']) != 100:
        print('len(temp_line[answer_options]): ', len(temp_line['answer_options']))

    assert len(temp_line['answer_options']) == 100

    result.append(temp_line)


with open(result_file, 'w',encoding='utf-8') as f:
    json.dump(result, f,ensure_ascii=False,indent=2)