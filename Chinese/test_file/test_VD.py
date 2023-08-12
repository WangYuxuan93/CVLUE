import json

result = []

for i in range(3):
    new_line = {}
    new_line['image'] = i
    new_line['caption'] = ''
    new_line['dialog'] = {}
    for j in range(3):
        new_line['dialog'][j] = {}
        new_line['dialog'][j]['question'] = ''
        new_line['dialog'][j]['answer'] = []
        for k in range(5):
            new_line['dialog'][j]['answer'].append('')
        new_line['dialog'][j]['correct'] = ''

    result.append(new_line)

with open("test_VD.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)