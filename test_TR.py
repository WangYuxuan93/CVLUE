
import json

result = []

for i in range(6):
    new_line = {}
    new_line['id'] = i
    new_line['picture'] = i
    new_line['text'] = []
    for j in range(5):
        new_line['text'].append('')
    new_line['correct'] = ''
    result.append(new_line)

with open("test_TR.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)