import json

result = []

for i in range(6):
    new_line = {}
    new_line['question_id'] = i
    new_line['question'] = ''
    new_line['image'] = i
    new_line['answer'] = ''
    result.append(new_line)

with open("test_VQA.json",'w',encoding='utf-8') as f:
    json.dump(result,f,ensure_ascii=False,indent=1)