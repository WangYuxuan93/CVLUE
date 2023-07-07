
import json
import random

result = []

refs = []

with open('test_VQA.json', 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    for line in raw_data:
        refs.append(line)

for i in range(len(refs)):
    ref = refs[i]
    new_line = ref
    result.append(new_line)

with open("test_VQA_pre.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)