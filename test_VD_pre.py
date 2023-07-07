
import json
import random

result = []

refs = []

with open('test_VD.json', 'r', encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    for line in raw_data:
        refs.append(line)

for i in range(len(refs)):
    ref = refs[i]
    new_line = {}
    new_line['image'] = ref['image']
    new_line['caption'] = ref['caption']
    new_line['dialog'] = {}
    for k,j in ref['dialog'].items():
        question = j['question']
        answer = j['answer']
        random.shuffle(answer)
        new_line['dialog'][k] = {}
        new_line['dialog'][k]['question'] = question
        new_line['dialog'][k]['answer'] = answer

    result.append(new_line)

with open("test_VD_pre.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)