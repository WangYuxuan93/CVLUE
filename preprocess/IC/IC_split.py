import json
import random

raw_file = 'IC.json'
train_file = 'IC_train.json'
val_file = 'IC_val.json'
test_file = 'IC_test.json'


with open(raw_file, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)

new_data = {}

for line in raw_data:
    image = line['image']
    caption = line['caption']
    class_1 = image[0]
    index = image[2:].find('/')
    class_2 = image[2:2+index]
    if class_2 not in new_data:
        new_data[class_2] = []

    new_line = {}
    new_line['image'] = image
    new_line['caption'] = caption
    new_data[class_2].append(new_line)

train = []
val = []
test = []
_num = {}

for _class in new_data:
    lines = new_data[_class]
    random.shuffle(lines)
    lines_len = len(lines)
    _num[_class] = lines_len
    for i in range(lines_len):
        if i < lines_len * 0.6:
            temp_image = lines[i]['image']
            temp_caps = lines[i]['caption']
            for cap in temp_caps:
                temp_line = {}
                temp_line['image'] = temp_image
                temp_line['caption'] = cap
                train.append(temp_line)
        elif i < lines_len * 0.7:
            temp_image = lines[i]['image']
            temp_caps = lines[i]['caption']
            for cap in temp_caps:
                temp_line = {}
                temp_line['image'] = temp_image
                temp_line['caption'] = cap
                val.append(temp_line)
        else:
            temp_image = lines[i]['image']
            temp_caps = lines[i]['caption']
            for cap in temp_caps:
                temp_line = {}
                temp_line['image'] = temp_image
                temp_line['caption'] = cap
                test.append(temp_line)

random.shuffle(train)
with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val, f, ensure_ascii=False, indent=2)
with open(test_file, 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print('all_num',_num)
print('label_len',len(_num))
print('_sum',sum(_num.values()))

print('train_len',len(train))
print('test_len',len(test))
print('val_len',len(val))