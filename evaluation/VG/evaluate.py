import json
from tqdm import tqdm

result_file = 'VG_result.json'
gold_file = 'VG_test_gold.json'

with open(result_file, 'r', encoding='utf-8') as f:
    string = f.read()
    results = json.loads(string)

with open(gold_file, 'r', encoding='utf-8') as f:
    string = f.read()
    gold = json.loads(string)

ref_id_map = {}
check_ref_id = {}
for item in gold:
    ref_id_map[item['ref_id']] = item
    check_ref_id[item['ref_id']] = 0

def computeIoU(box1, box2):

    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0] + box1[2] - 1, box2[0] + box2[2] - 1)
    inter_y2 = min(box1[1] + box1[3] - 1, box2[1] + box2[3] - 1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)
    else:
        inter = 0
    union = box1[2] * box1[3] + box2[2] * box2[3] - inter
    return float(inter) / union

IoU_sum = 0
num_ = 0
correct_num_ = 0



for item in tqdm(results):
    ref_id = item['ref_id']
    pred = item['bbox']
    gold = ref_id_map[ref_id]['bbox']
    check_ref_id[ref_id] = 1

    IoU_tmp = computeIoU(gold, pred)
    IoU_sum += IoU_tmp

    num_ += 1
    if IoU_tmp > 0.5:
        correct_num_ += 1

for ref_id, flag in check_ref_id.items():
    if flag != 1:
        print(ref_id, ' not exists!')

avg_IoU = IoU_sum / num_
acc = correct_num_ / num_
print('Avg_IoU:', avg_IoU)
print('Acc:', acc)

