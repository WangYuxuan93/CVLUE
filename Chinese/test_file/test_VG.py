import json

result = []

for i in range(6):
    new_line = {}
    new_line['id'] = i
    new_line['ref'] = {}
    for j in range(1):
        new_line['ref'][j] = {'x1':0.0,'x2':0.0,'y1':0.0,'y2':0.0}
    result.append(new_line)

with open("test_VG.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)