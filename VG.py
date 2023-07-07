
import json
from tqdm import tqdm

def compute_VG(ref_list,pre_list):

    assert len(ref_list) == len(pre_list)

    scores = []
    correct = 0
    total = 0

    for ref,pre in zip(ref_list,pre_list):

        for _ref,_pre in zip(ref['ref'].values(),pre['pre'].values()):

            total += 1
            x = []
            y = []
            x.append(_ref['x1'])
            x.append(_ref['x2'])
            y.append(_ref['y1'])
            y.append(_ref['y2'])
            x.append(_pre['x1'])
            x.append(_pre['x2'])
            y.append(_pre['y1'])
            y.append(_pre['y2'])

            x.sort()
            y.sort()

            pre_x = abs(_pre['x1'] - _pre['x2'])
            pre_y = abs(_pre['y1'] - _pre['y2'])
            ref_x = abs(_ref['x1'] - _ref['x2'])
            ref_y = abs(_ref['y1'] - _ref['y2'])

            cor_x = x[2] - x[1]
            cor_y = y[2] - y[1]

            pre_s = pre_x * pre_y
            ref_s = ref_x * ref_y
            cor_s = cor_x * cor_y

            s1 = pre_s + ref_s - cor_s
            score = float(ref_s) / float(s1)
            if  score > 0.5:
                correct += 1

            scores.append(score)
    # print(scores)
    return scores,float(correct) / float(total)

if __name__ == '__main__':
    ref_path = 'test_VG.json'
    pre_path = 'test_VG_pre.json'

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