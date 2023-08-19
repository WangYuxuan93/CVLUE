
import json
from tqdm import tqdm

def my_lcs(string, sub):

    if(len(string)< len(sub)):
        sub, string = string, sub

    lengths = [[0 for i in range(0,len(sub)+1)] for j in range(0,len(string)+1)]

    for j in range(1,len(sub)+1):
        for i in range(1,len(string)+1):
            if(string[i-1] == sub[j-1]):
                lengths[i][j] = lengths[i-1][j-1] + 1
            else:
                lengths[i][j] = max(lengths[i-1][j] , lengths[i][j-1])

    return lengths[len(string)][len(sub)]

def compute_Rouge_L(ref_list,pre_list,beta):

    scores = []

    for refs,pre in zip(ref_list,pre_list):
        max_r = 0.
        max_len = 0

        for ref in refs:
            temp_len = my_lcs(ref,pre)
            r = float(temp_len) / float(len(ref))
            if r > max_r:
                max_r = r
            if temp_len > max_len:
                max_len = temp_len
        
        if len(pre) == 0:
            max_p = 0
        else:
            max_p = float(max_len) / float(len(pre))

        if(max_p != 0 and max_r != 0):

            score = ((1 + beta ** 2) * max_p * max_r)/float(max_r + beta ** 2 * max_p)
        else:
            score = 0.0

        scores.append(score)

    return scores



if __name__ == '__main__':
    ref_path = 'test_IC.json'
    pre_path = 'test_IC_pre.json'
    beta = 1.2
    ref = []
    pre = []

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

    result = compute_Rouge_L(ref,pre,beta)
    print(result)
    rouge_l = sum(result) / len(result)
    print('Rouge_L:',rouge_l)
    