import json
from tqdm import tqdm

result_file = 'VD_result_new.json'

with open(result_file, 'r', encoding='utf-8') as f:
    string = f.read()
    results = json.loads(string)


gold_file = 'VD_test_gold_new.json'

with open(gold_file, 'r', encoding='utf-8') as f:
    string = f.read()
    gold_data = json.loads(string)

gold_ans_map = gold_data
check_image = {}
for image_id in gold_data.keys():
    check_image[image_id] = [0] * 10

num_ = 0
n_1 = 0
n_5 = 0
n_10 = 0

for item in tqdm(results):
    image_id = item['image_id']
    pred = item['dialog'][-1]['answer_sort']
    tmp_dialog_id = len(item['dialog']) - 1
    gold = gold_ans_map[str(image_id)][tmp_dialog_id]['answer']

    check_image[str(image_id)][tmp_dialog_id] = 1

    pred_rank = pred.index(int(gold))

    if pred_rank < 1:
        n_1 += 1
    if pred_rank < 5:
        n_5 += 1
    if pred_rank < 10:
        n_10 += 1
    
    num_ += 1

for image_id, flags in check_image.items():
    for i,flag in enumerate(flags):
        if flag != 1:
            print('Dialog ', i,' in ', image_id, ' not exists!')

r_1 = n_1 / num_
r_5 = n_5 / num_
r_10 = n_10 / num_

print('R_1:', r_1)
print('R_5:', r_5)
print('R_10:', r_10)

