
import json
from tqdm import tqdm

def compute_VG(ref_list,pre_list):

    assert len(ref_list) == len(pre_list)

    scores = []
    correct = 0
    total = len(ref_list)

    for ref,pre in zip(ref_list,pre_list):

        gold_bbox = ref['bbox']
        pred_bbox = pre['bbox']

        x1 = max(gold_bbox[0], pred_bbox[0])
        y1 = max(gold_bbox[1], pred_bbox[1])
        x2 = min(gold_bbox[0] + gold_bbox[2] - 1, pred_bbox[0] + pred_bbox[2] - 1)
        y2 = min(gold_bbox[1] + gold_bbox[3] - 1, pred_bbox[1] + pred_bbox[3] - 1)

        if x1 < x2 and y1 < y2:
            inter = (x2 - x1 + 1) * (y2 - y1 + 1)
        else:
            inter = 0

        union = gold_bbox[2] * gold_bbox[3] + pred_bbox[2] * pred_bbox[3] - inter

        score = float(inter) / union

        if score > 0.5:
            correct += 1

        scores.append(score)
    # print(scores)
    return scores,float(correct) / float(total)

if __name__ == '__main__':
    ref_path = 'VG_result.json'
    pre_path = 'VG_test.json'

    ref = []
    pre = []

    with open(ref_path, 'r', encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            ref.append(line)

    with open(pre_path, 'r', encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            pre.append(line)

    IOU,result = compute_VG(ref,pre)
    print('IOU:',sum(IOU) / len(IOU))
    print('VG_score:',result)