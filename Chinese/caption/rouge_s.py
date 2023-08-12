
import json
from collections import defaultdict
from tqdm import tqdm

def precook(s, n = 4):

    words = s.replace(" ","")
    counts = defaultdict(int)

    for k in range(1,n+1):
        for i in range(len(words)-k):
            word_1 = words[i]
            word_2 = words[i+k]
            ngram = word_1 + word_2
            counts[ngram] += 1

    return (sum(counts.values()), counts)


def cook_test(test, ref, n=4):

    reflen, refmaxcounts = precook(ref, n)
    testlen, counts = precook(test, n)

    result = {}

    result["testlen"] = testlen

    result["reflen"] = reflen

    result['correct'] = 0
    for (ngram, count) in counts.items():
        result["correct"] += min(refmaxcounts.get(ngram,0), count)

    return result

def compute_Rouge_S(ref_list,pre_list,rouge_s_n,beta):

    assert len(ref_list) == len(pre_list)

    scores = []

    for refs,pre in zip(ref_list,pre_list):
        max_r = 0.
        max_p = 0.
        for ref in refs:
            result = cook_test(pre,ref,rouge_s_n)


            r = float(result['correct']) / float(result['reflen'])
            if r > max_r:
                max_r = r

            p = float(result['correct']) / float(result['testlen'])
            if p > max_p:
                max_p = p

        # print(max_r,max_p)
        if (max_p != 0 and max_r != 0):

            score = ((1 + beta ** 2) * max_p * max_r) / float(max_r + beta ** 2 * max_p)
        else:
            score = 0.0

        scores.append(score)

    return scores


if __name__ == '__main__':
    ref_path = 'test_IC.json'
    pre_path = 'test_IC_pre.json'
    rouge_s_n = 4
    beta = 1.2
    ref = []
    pre = []

    with open(ref_path, 'r', encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            ref.append(line['ref'])
    with open(pre_path, 'r', encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        for line in tqdm(raw_data):
            pre.append(line['pre'])

    result = compute_Rouge_S(ref, pre, rouge_s_n, beta)
    print(result)
    rouge_s = sum(result) / len(result)
    print('Rouge_S:', rouge_s)