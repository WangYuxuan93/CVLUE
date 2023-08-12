import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np

def precook(s, n):

    words = s.replace(" ","")
    counts = defaultdict(int)

    for i in range(len(words)-n+1):
        ngram = words[i:i+n]
        counts[ngram] += 1

    return (len(words), counts)

def get_dict(refs,cider_n):
    sum_dict = [0] * cider_n
    dict = defaultdict(int)
    for ref in refs:
        for i in range(1,1+cider_n):
            for j in range(len(ref)-i+1):
                ngram = ref[j:j+i]
                sum_dict[i-1] += 1
                dict[ngram] += 1

    return sum_dict,dict

def compute_cider(ref_list,pre_list,cider_n):

    assert len(ref_list) == len(pre_list)

    ciders = []
    pic_num = len(ref_list)
    dic = []

    for i in range(pic_num):
        dic.append([])
        refs = ref_list[i]
        _,new_dic = get_dict(refs,cider_n)
        dic[i].append(new_dic)


    for pre,refs in zip(pre_list,ref_list):
        cider_ = []
        for k in range(1,1+cider_n):
            cider = 0.
            pre_len,pre_counts = precook(pre,k)
            pre_len -= k - 1
            s_pre = {}
            for pre_ngram,pre_freq in pre_counts.items():
                pre_tf = float(pre_freq) / float(pre_len)
                pre_sum = 1
                for j in range(pic_num):
                    if dic[j][0][pre_ngram] > 0:
                        pre_sum += 1
                pre_idf = np.log2(float(pic_num + 1) / float(pre_sum))
                s_pre[pre_ngram] = pre_tf * pre_idf

            pre_l = 0.
            for freq in s_pre.values():
                pre_l += pow(freq,2)
            pre_l = pow(pre_l,0.5)

            for ref in refs:
                ref_len,ref_counts = precook(ref,k)
                ref_len -= k - 1
                s_ref = {}
                for ref_ngram,ref_freq in ref_counts.items():
                    ref_tf = float(ref_freq) / float(ref_len)
                    ref_sum = 1
                    for j in range(pic_num):
                        if dic[j][0][ref_ngram] > 0:
                            ref_sum += 1
                    ref_idf = np.log2(float(pic_num + 1) / float(ref_sum))
                    s_ref[ref_ngram] = ref_tf * ref_idf

                ref_l = 0.
                for freq in s_ref.values():
                    ref_l += pow(freq, 2)
                ref_l = pow(ref_l, 0.5)

                _sum = 0.
                for ngram ,freq in s_pre.items():
                    if ngram in s_ref.keys():
                        _sum += freq * s_ref[ngram]

                temp = _sum / (pre_l * ref_l)
                cider += temp

            cider = cider / float(len(refs))
            cider_.append(cider)

        ciders.append(cider_)
    return ciders

    # for refs,pre in zip(ref_list,pre_list):
    #     cider = []
    #     for k in range(1,cider_n + 1):
    #         pre_len,pre_counts = precook(pre,k)
    #         pre_len -= k - 1
    #
    #         s_pre = {}
    #         for ngrams,freq in pre_counts.items():
    #             pre_tf = float(freq) / float(pre_len)
    #             pre_sum = 0
    #             for ref in refs:
    #                 ref_len, ref_counts = precook(ref, k)
    #                 ref_len -= k - 1
    #
    #                 if ref_counts[ngrams] > 0:
    #                     pre_sum += 1
    #
    #                 for ref_ngrams, ref_freq in ref_counts.items():
    #                     ref_tf = float(ref_freq) / float(ref_len)


if __name__ == '__main__':
    ref_path = 'test_IC.json'
    pre_path = 'test_IC_pre.json'
    cider_n = 4
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

    results = compute_cider(ref, pre, cider_n)
    new_results = []
    for result in results:
        new_result = sum(result) / len(result)
        new_results.append(new_result)
    # print(new_results)
    print('Cider:',sum(new_results) / len(new_results))

    # ciders = []
    #
    # for result in results:
    #     new_result = sum(result) / len(result)
    #     ciders.append(new_result)

    # print('Cider:',sum(ciders)/len(ciders))

    # print(get_dict(ref,4))