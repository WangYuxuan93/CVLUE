import json

result = []

for i in range(6):
    new_line = {}
    new_line['id'] = i
    new_line['pre'] = ''
    result.append(new_line)

with open("test_IC_pre.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)