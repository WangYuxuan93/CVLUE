
import json
from collections import defaultdict
from tqdm import tqdm

def precook(s, n = 4):

    words = s.replace(" ","")
    counts = defaultdict(int)

    for k in range(1,n+1):
        for i in range(len(words)-k+1):
            ngram = words[i:i+k]
            counts[ngram] += 1

    return counts

def cook_refs(ref, n=4): 

    counts = precook(ref, n)

    return len(ref),counts

def cook_test(test, refparam, n=4):

    reflen,refmaxcounts = refparam[0],refparam[1]
    counts = precook(test, n)

    result = {}

    result["guess"] = [max(0,reflen-k+1) for k in range(1,n+1)]

    result['correct'] = [0]*n
    for (ngram, count) in counts.items():
        result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)

    return result

def compute_Rouge_N(ref_list,pre_list,rouge_n):

    assert len(ref_list) == len(pre_list)

    small = 1e-9
    tiny = 1e-15

    totalcomps = {'guess':[0]*rouge_n, 'correct':[0]*rouge_n}

    for refs,pre in zip(ref_list,pre_list):
        for ref in refs:
            new_ref = cook_refs(ref,rouge_n)
            new_pre = cook_test(pre,new_ref,rouge_n)

            for key in ['guess','correct']:
                for k in range(rouge_n):
                    totalcomps[key][k] += new_pre[key][k]

    rouges = []

    for k in range(rouge_n):
        rouge = 1.

        rouge *= float(totalcomps['correct'][k] + tiny) \
                / (totalcomps['guess'][k] + small)
        rouges.append(rouge)


    return rouges


if __name__ == '__main__':
    ref_path = 'test_IC.json'
    pre_path = 'test_IC_pre.json'
    ref = []
    pre = []
    rouge_n = 4

    with open(ref_path,'r',encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            ref.append(line['ref'])
    with open(pre_path,'r',encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            pre.append(line['pre'])

    result = compute_Rouge_N(ref,pre,rouge_n)

    rouge_list = []

    for i in range(1,rouge_n + 1):
        rouge = 'Rouge_'
        rouge_index = rouge + str(i)
        rouge_list.append(rouge_index)

    mat = "{:>20}\t" * (rouge_n)

    print(mat.format(*rouge_list))

    new_rouge = []

    for i in result:
        new_rouge.append(i)

    print(mat.format(*new_rouge))


    # print(result)

    # print(ref)
    # print(pre)
    # print(precook(pre[0],4))