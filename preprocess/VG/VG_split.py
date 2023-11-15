import json
import random

# 训练集、验证集和测试集的切分

file_1 = 'VG.json.json'
file_2 = 'VG2.json.json'

new_data = []

with open(file_1, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    new_data = raw_data

with open(file_2, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    new_data += raw_data

img_data = {}
img_list = []
for line in new_data:
    ref_id = line['ref_id']
    image = line['image']
    text = line['text']
    if text == '' or text == None:
        continue
    if ref_id == 0:
        img_data[image] = []
        img_list.append(image)
    if image not in img_data:
        continue
    img_data[image].append(line)

random.shuffle(img_list)

# for img,groudings in img_data.items():
#     for i in range(len(groudings)):
#         # print(img_data[img][i]['text'])
#         # input()
#         img_data[img][i]['text'] = img_data[img][i]['text'].strip()

train = []
val = []
test = []


train_file = 'VG_train.json'
val_file = 'VG_val.json'
test_file = 'VG_test.json'

for i in range(len(img_list)):
    if i < len(img_list) * 0.6:
        train += img_data[img_list[i]]
    elif i < len(img_list) * 0.7:
        val += img_data[img_list[i]]
    else:
        test += img_data[img_list[i]]


with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val, f, ensure_ascii=False, indent=2)
with open(test_file, 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print('train_len',len(train))
print('test_len',len(test))
print('val_len',len(val))
