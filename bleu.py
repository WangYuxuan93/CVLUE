
import math
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

    return (len(words), counts)

def cook_refs(refs, eff=None, n=4): 

    reflen = []
    maxcounts = {}
    for ref in refs:
        rl, counts = precook(ref, n)
        reflen.append(rl)
        for (ngram,count) in counts.items():
            maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    if eff == "shortest":
        reflen = min(reflen)
    elif eff == "average":
        reflen = float(sum(reflen))/len(reflen)

    return (reflen, maxcounts)

def cook_test(test, refparam, eff=None, n=4):

    reflen, refmaxcounts = refparam[0], refparam[1]
    testlen, counts = precook(test, n)

    result = {}

    if eff == "closest":
        result["reflen"] = min((abs(l-testlen), l) for l in reflen)[1]
    else: ## i.e., "average" or "shortest" or None
        result["reflen"] = reflen

    result["testlen"] = testlen

    result["guess"] = [max(0,testlen-k+1) for k in range(1,n+1)]

    result['correct'] = [0]*n
    for (ngram, count) in counts.items():
        result["correct"][len(ngram)-1] += min(refmaxcounts.get(ngram,0), count)

    return result

def compute_BLEU(ref_list,pre_list,bleu_n,eff):

    assert len(ref_list) == len(pre_list)

    refs = []
    pres = []
    small = 1e-9
    tiny = 1e-15
    _testlen = 0
    _reflen = 0
    totalcomps = {'testlen':0, 'reflen':0, 'guess':[0]*bleu_n, 'correct':[0]*bleu_n}

    bleu_list = [[] for _ in range(bleu_n)]

    for ref,pre in zip(ref_list,pre_list):
        new_ref = cook_refs(ref,eff,bleu_n)
        new_pre = cook_test(pre,new_ref,eff,bleu_n)
        refs.append(new_ref)
        pres.append(new_pre)
        testlen = new_pre['testlen']
        reflen = new_pre['reflen']
        _testlen += testlen
        _reflen += reflen

        for key in ['guess','correct']:
            for k in range(bleu_n):
                totalcomps[key][k] += new_pre[key][k]

        bleu = 1.
        for k in range(bleu_n):

            bleu *= (float(new_pre['correct'][k]) + tiny) \
                    /(float(new_pre['guess'][k]) + small) 
            bleu_list[k].append(bleu ** (1./(k+1)))

        ratio = (testlen + tiny) / (reflen + small)
        if ratio < 1:
            for k in range(bleu_n):
                bleu_list[k][-1] *= math.exp(1 - 1/ratio)


    totalcomps['reflen'] = _reflen
    totalcomps['testlen'] = _testlen

    bleus = []
    bleu = 1.
    for k in range(bleu_n):
        bleu *= float(totalcomps['correct'][k] + tiny) \
                / (totalcomps['guess'][k] + small)
        bleus.append(bleu ** (1./(k+1)))
    ratio = (_testlen + tiny) / (_reflen + small) ## N.B.: avoid zero division

    if ratio < 1:
        for k in range(bleu_n):
            bleus[k] *= math.exp(1 - 1/ratio)

    return bleus, bleu_list


if __name__ == '__main__':
    ref_path = 'test_IC.json'
    pre_path = 'test_IC_pre.json'
    ref = []
    pre = []
    bleu_n = 4
    eff_ref = 'shortest' # ref_length,shortest/average/closest

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

    result = compute_BLEU(ref,pre,bleu_n,eff_ref)

    bleu_list = []
    bleu_list.append('index')

    for i in range(1,bleu_n + 1):
        bleu = 'BLEU'
        bleu_index = bleu + str(i)
        bleu_list.append(bleu_index)

    mat = "{:>20}\t" * (bleu_n + 1)

    print(mat.format(*bleu_list))

    new_bleu = []
    new_bleu.append('total')

    for i in result[0]:
        new_bleu.append(i)

    print(mat.format(*new_bleu))

    for i in range(1,len(ref) +1):
        new_bleu_list = []
        new_bleu_list.append(i)
        for k in range(bleu_n):
            new_bleu_list.append(result[1][k][i-1])

        print(mat.format(*new_bleu_list))

    # print(result)

    # print(ref)
    # print(pre)
    # print(precook(pre[0],4))