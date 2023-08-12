
import json
from tqdm import tqdm

def compute_VQA(pres,refs):

    assert len(pres) == len(refs)

    correct = 0
    total = len(pres)

    for i in range(total):
        pre = pres[i]['answer']
        ref = refs[i]['answer']

        if pre == ref:
            correct += 1

    return float(correct) / float(total)

if __name__ == '__main__':
    ref_path = 'VQA_test.json'
    pre_path = 'VQA_result.json'

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

    result = compute_VQA(pre,ref)

    print('VQA_score:',result)