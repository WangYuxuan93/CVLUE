
import json

result = []

for i in range(6):
    new_line = {}
    new_line['id'] = i
    new_line['text'] = ''
    new_line['picture'] = []
    for j in range(5):
        new_line['picture'].append(i * 5 + j)
    new_line['correct'] = i * 5 + 2
    result.append(new_line)

with open("test_IR.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)