
import json
from tqdm import tqdm
import numpy as np

def R_n(pres,refs,n):
    assert len(pres) == len(refs)

    correct = 0
    total = 0

    for i in range(len(pres)):
        pre = pres[i]['dialog']
        ref = refs[i]['dialog']
        assert len(pre) == len(ref)

        for j in pre.keys():
            pre_answer = pre[j]['answer']
            correct_answer = ref[j]['correct']
            pre_rank = pre_answer.index(correct_answer)
            total += 1

            if pre_rank < n:
                correct += 1

    return float(correct) / float(total)

def mean_rank(pres,refs):
    assert len(pres) == len(refs)

    total_rank = 0
    total = 0
    for i in range(len(pres)):
        pre = pres[i]['dialog']
        ref = refs[i]['dialog']
        assert len(pre) == len(ref)

        for j in pre.keys():
            pre_answer = pre[j]['answer']
            correct_answer = ref[j]['correct']
            pre_rank = pre_answer.index(correct_answer)
            total += 1
            total_rank += pre_rank + 1

    return float(total_rank) / float(total)

def MRR(pres,refs):
    assert len(pres) == len(refs)

    total_mrr = 0.
    total = 0
    for i in range(len(pres)):
        pre = pres[i]['dialog']
        ref = refs[i]['dialog']
        assert len(pre) == len(ref)

        for j in pre.keys():
            pre_answer = pre[j]['answer']
            correct_answer = ref[j]['correct']
            pre_rank = pre_answer.index(correct_answer)
            total += 1
            total_mrr += 1. / float(pre_rank + 1)

    return total_mrr / float(total)

def NDCG(pres,refs):
    assert len(pres) == len(refs)

    total_ndcg = 0.
    total = 0
    for i in range(len(pres)):
        pre = pres[i]['dialog']
        ref = refs[i]['dialog']
        assert len(pre) == len(ref)

        for j in pre.keys():
            pre_answer = pre[j]['answer']
            correct_answer = ref[j]['correct']
            pre_rank = pre_answer.index(correct_answer)
            total += 1
            total_ndcg += 1. / np.log2(pre_rank + 2)

    return total_ndcg / float(total)


if __name__ == '__main__':
    ref_path = 'test_VD.json'
    pre_path = 'test_VD_pre.json'

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

    R_1 = R_n(pre, ref, 1)
    R_2 = R_n(pre, ref, 2)
    R_5 = R_n(pre, ref, 5)
    print('R_!:',R_1,'  R_2:', R_2, '   R_5:',R_5)

    mean_rank = mean_rank(pre,ref)
    print('Mean_rank:',mean_rank)

    mrr = MRR(pre,ref)
    print('MRR:',mrr)

    ncdg = NDCG(pre,ref)
    print('NCDG:',ncdg)