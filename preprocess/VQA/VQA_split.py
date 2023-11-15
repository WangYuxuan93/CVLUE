import json
import random

# 训练集、验证集和测试集切分

raw_file = 'VQA.json'
train_file = 'VQA_train.json'
val_file = 'VQA_val.json'
test_file = 'VQA_test.json'

with open(raw_file, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)

new_data = {}
new_data_index = {}

for line in raw_data:
    image = line['image']

    index = image[2:].find('/')
    class_2 = image[2:2 + index]

    if class_2 not in new_data:
        new_data[class_2] = {}
        new_data_index[class_2] = []

    if image not in new_data_index[class_2]:
        new_data_index[class_2].append(image)

    if image not in new_data[class_2]:
        new_data[class_2][image] = []
    new_line = {}
    new_line['image'] = image
    new_line['question_id'] = line['question_id']
    new_line['question'] = line['question']
    new_line['answer'] = line['answer']
    new_data[class_2][image].append(new_line)

train = []
val = []
test = []
_num = {}

for _class in new_data_index:
    image_set = new_data_index[_class]
    line_set = new_data[_class]
    random.shuffle(image_set)
    image_set_len = len(image_set)
    _num[_class] = image_set_len
    for i in range(image_set_len):
        if i < image_set_len * 0.6-1:
            train += line_set[image_set[i]]
        elif i < image_set_len * 0.7-1:
            val += line_set[image_set[i]]
        else:
            test += line_set[image_set[i]]

with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val, f, ensure_ascii=False, indent=2)
with open(test_file, 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print('train_len',len(train))
print('test_len',len(test))
print('val_len',len(val))
print('label_num',len(_num))
print('sum_num',_num)
