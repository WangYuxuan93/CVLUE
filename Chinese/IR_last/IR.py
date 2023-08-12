
import json
from tqdm import tqdm

def R_n(pres,refs,n):

    assert len(pres) == len(refs)

    total = len(pres)
    correct = 0

    for i in range(total):

        gold_cap = refs[i]['caption']
        pre_caps = pres[i]['caption']

        pre_index = pre_caps.index(gold_cap)

        if pre_index < n:
            correct += 1

    return float(correct) / float(total)

if __name__ == '__main__':
    ref_path = 'IR_test.json'
    pre_path = 'IR_result.json'

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

    R_1 = R_n(pre,ref,1)
    R_2 = R_n(pre,ref,2)
    R_5 = R_n(pre,ref,5)

    print('R_1:',R_1,'  R_2:', R_2, '   R_5:',R_5)