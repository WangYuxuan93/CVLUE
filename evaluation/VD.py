
import json
from tqdm import tqdm
import numpy as np

def R_n(pres,refs,n):
    # assert len(pres) == len(refs)

    correct = 0
    total = 0

    for line in pres:
        for new_line in refs:
            if line['image'] == new_line['image'] and line['dialog_id'] == new_line['dialog_id']:
                total += 1
                pred_rank = line['answer']
                gold = new_line['answer']

                rank = pred_rank.index(gold)

                if rank < n:
                    correct += 1

    return float(correct) / float(total)

def mean_rank(pres,refs):
    # assert len(pres) == len(refs)

    total_rank = 0
    total = 0
    for line in pres:
        for new_line in refs:
            if line['image'] == new_line['image'] and line['dialog_id'] == new_line['dialog_id']:
                total += 1
                pred_rank = line['answer']
                gold = new_line['answer']

                rank = pred_rank.index(gold)

                total_rank += rank
                total += 1
    print('total',total)
    return float(total_rank) / float(total) + 1.

def MRR(pres,refs):
    # assert len(pres) == len(refs)

    total_mrr = 0.
    total = 0
    for line in pres:
        for new_line in refs:
            if line['image'] == new_line['image'] and line['dialog_id'] == new_line['dialog_id']:
                total += 1
                pred_rank = line['answer']
                gold = new_line['answer']

                rank = pred_rank.index(gold)

                total_mrr += 1. / (rank + 1.)
                total += 1

    return total_mrr / float(total)

def NDCG(pres,refs):
    # assert len(pres) == len(refs)

    total_ndcg = 0.
    total = 0
    for i in range(len(pres)):
        pred_rank = pres[i]['answer']
        gold = refs[i]['answer']

        rank = pred_rank.index(gold)

        total_ndcg += 1. / np.log2(rank + 2)
        total += 1

    return total_ndcg / float(total)


if __name__ == '__main__':
    ref_path = 'new_VD_test.json'
    pre_path = 'result_ch_blip.json'

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

    print(len(pre))
    print(len(ref))

    R_1 = R_n(pre, ref, 1)
    R_2 = R_n(pre, ref, 2)
    R_5 = R_n(pre, ref, 5)
    R_10 = R_n(pre, ref, 10)

    print('R_1:',R_1,'  R_2:', R_2, '   R_5:',R_5,'R_10:',R_10)

    mean_rank = mean_rank(pre,ref)
    print('Mean_rank:',mean_rank)

    mrr = MRR(pre,ref)
    print('MRR:',mrr)

    # ncdg = NDCG(pre,ref)
    # print('NCDG:',ncdg)