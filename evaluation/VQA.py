
import json
from tqdm import tqdm

def compute_VQA(pres,refs):

    assert len(pres) == len(refs)

    correct = 0
    total = len(pres)

    for pre_line in pres:
        for ref_line in refs:
            if pre_line['question_id'] == ref_line['question_id']:

                if pre_line['answer'] == ref_line['answer']:
                    correct += 1

    return float(correct) / float(total)

if __name__ == '__main__':
    ref_path = 'VQA_test.json'
    pre_path = 'result_ch_clip.json'

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