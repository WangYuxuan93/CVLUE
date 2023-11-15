import json

# 统计 VD 数据集中全部的问题与答案，并将其转化为 id 的表示
# 同时，统计答案出现的频数。发你后续答案列表的生成

raw_path = 'VD.json'

question_list_file = 'vd_question.json'
answer_list_file = 'vd_answer.json'
answer_count_file = 'vd_answer_cnt.json'
new_path = 'new_VD.json'


with open(raw_path, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)

temp_id = 0
question_list = {}
answer_list = {}
answer_count = {}
answer2id = {}
question2id = {}
temp_question_id = 0
temp_answer_id = 0
new_data = []
temp_image_id = 0

for line in raw_data:
    if len(line['data']) != 0:
        dialogs = line['data'][0]['dialog']
        image = line['data'][0]['image']
        caption = line['data'][0]['caption']
        new_lines = []

        for i in dialogs.keys():
            new_line = {}
            new_line['image_id'] = temp_image_id
            new_line['image'] = image
            new_line['caption'] = caption
            new_line['dialog_id'] = int(i) - 1

            question = dialogs[i]['question']
            answer = dialogs[i]['answer']

            # print('q',question)
            # print('a',answer)
            # print(new_line['dialog_id'])
            # input()

            if question not in question2id:
                question2id[question] = temp_question_id
                question_list[temp_question_id] = question
                temp_question_id += 1

            new_question = question2id[question]

            if answer not in answer2id:
                answer2id[answer] = temp_answer_id
                answer_list[temp_answer_id] = answer
                answer_count[answer] = 0
                temp_answer_id += 1

            answer_count[answer] += 1
            new_answer = answer2id[answer]

            new_line['question'] = new_question
            new_line['answer_options'] = []
            new_line['answer'] = new_answer
            new_lines.append(new_line)

        temp_image_id += 1
        new_data.append(new_lines)

with open(question_list_file, 'w', encoding='utf-8') as f:
    json.dump(question_list, f, ensure_ascii=False, indent=2)
with open(answer_list_file, 'w', encoding='utf-8') as f:
    json.dump(answer_list, f, ensure_ascii=False, indent=2)
with open(answer_count_file, 'w', encoding='utf-8') as f:
    json.dump(answer_count, f, ensure_ascii=False, indent=2)
with open(new_path, 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)



