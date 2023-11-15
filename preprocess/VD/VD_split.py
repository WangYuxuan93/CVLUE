import json
import random

# 切分训练、验证和测试集

raw_path = 'new_VD.json'

train_file = 'VD_train.json'
val_file = 'VD_val.json'
test_file = 'VD_test.json'

with open(raw_path, 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)

random.shuffle(raw_data)

train = []
val = []
test = []

for i in range(len(raw_data)):
    if i < len(raw_data) * 0.6:
        train += raw_data[i]
    elif i < len(raw_data) * 0.7:
        val += raw_data[i]
    else:
        test += raw_data[i]


with open(train_file, 'w', encoding='utf-8') as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open(val_file, 'w', encoding='utf-8') as f:
    json.dump(val, f, ensure_ascii=False, indent=2)
with open(test_file, 'w', encoding='utf-8') as f:
    json.dump(test, f, ensure_ascii=False, indent=2)

print('train_len',len(train))
print('test_len',len(test))
print('val_len',len(val))