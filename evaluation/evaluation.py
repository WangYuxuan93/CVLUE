from bleu import compute_BLEU
from cider import compute_cider
from rouge_l import compute_Rouge_L
from rouge_n import compute_Rouge_N
from rouge_s import compute_Rouge_S
from VG import compute_VG
from VD import R_n as VD_R_n
from VD import mean_rank,MRR,NDCG
from VQA import compute_VQA
from IR import R_n as IR_R_n
from TR import R_n as TR_R_n

import json
from tqdm import tqdm
import argparse

def load_file(ref_path,pre_path):

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

    return ref,pre

def new_load_file(ref_path,pre_path):

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

    return ref,pre

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluation')
    parser.add_argument('--task', type=str, default='IC',help='Task')
    # parser.add_argument('--metric', type=str,default='Rouge', help='Metric')
    parser.add_argument('--n_gram', type=int,default=4, help='N-gram')
    parser.add_argument('--ref_path', type=str, default='test_IC.json',help='Task')
    parser.add_argument('--pre_path', type=str, default='test_IC_pre.json',help='Task')
    args = parser.parse_args()

    if args.task == 'IC':
    # Bleu
        print('Bleu:')
        eff_ref = 'shortest'
        ref , pre = load_file(args.ref_path,args.pre_path)
        bleu_n = args.n_gram
        result = compute_BLEU(ref, pre, bleu_n, eff_ref)
        bleu_list = []
        bleu_list.append('index')
        for i in range(1, bleu_n + 1):
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

        for i in range(1, len(ref) + 1):
            new_bleu_list = []
            new_bleu_list.append(i)
            for k in range(bleu_n):
                new_bleu_list.append(result[1][k][i - 1])
            print(mat.format(*new_bleu_list))

    # Cider
        print('Cider:')
        ref, pre = load_file(args.ref_path, args.pre_path)
        cider_n = args.n_gram
        results = compute_cider(ref, pre, cider_n)
        new_results = []
        for result in results:
            new_result = sum(result) / len(result)
            new_results.append(new_result)

        print('Cider:', sum(new_results) / len(new_results))

    # Rouge
        print('Rouge:')
        # rouge_l
        ref, pre = load_file(args.ref_path, args.pre_path)
        beta = 1.2
        result = compute_Rouge_L(ref, pre, beta)
        rouge_l = sum(result) / len(result)
        print('Rouge_L:', rouge_l)

        # rouge_n
        rouge_n = args.n_gram
        result = compute_Rouge_N(ref, pre, rouge_n)
        rouge_list = []
        print('Rouge_N:')
        for i in range(1, rouge_n + 1):
            rouge = 'Rouge_'
            rouge_index = rouge + str(i)
            rouge_list.append(rouge_index)
        mat = "{:>20}\t" * (rouge_n)
        print(mat.format(*rouge_list))

        new_rouge = []
        for i in result:
            new_rouge.append(i)
        print(mat.format(*new_rouge))

        # rouge_s
        rouge_s_n = args.n_gram
        result = compute_Rouge_S(ref, pre, rouge_s_n, beta)
        rouge_s = sum(result) / len(result)
        print('Rouge_S:', rouge_s)

    if args.task == 'VG':
        ref, pre = new_load_file(args.ref_path, args.pre_path)
        IOU, result = compute_VG(ref, pre)
        print('IOU:', sum(IOU) / len(IOU))
        print('VG_score:', result)

    if args.task == 'VD':
        print("VD:")
        ref, pre = new_load_file(args.ref_path, args.pre_path)
        R_1 = VD_R_n(pre, ref, 1)
        R_2 = VD_R_n(pre, ref, 2)
        R_5 = VD_R_n(pre, ref, 5)
        print('R_1:', R_1, '  R_2:', R_2, '   R_5:', R_5)

        mean_rank = mean_rank(pre, ref)
        print('Mean_rank:', mean_rank)

        mrr = MRR(pre, ref)
        print('MRR:', mrr)

        ncdg = NDCG(pre, ref)
        print('NCDG:', ncdg)

    if args.task == 'VQA':
        ref, pre = new_load_file(args.ref_path, args.pre_path)
        result = compute_VQA(pre, ref)
        print('VQA_score:', result)

    if args.task == 'IR':
        print("IR:")
        ref, pre = new_load_file(args.ref_path, args.pre_path)
        R_1 = IR_R_n(pre, ref, 1)
        R_2 = IR_R_n(pre, ref, 2)
        R_5 = IR_R_n(pre, ref, 5)
        print('R_1:', R_1, '  R_2:', R_2, '   R_5:', R_5)

    if args.task == 'TR':
        print("TR:")
        ref, pre = new_load_file(args.ref_path, args.pre_path)
        R_1 = TR_R_n(pre, ref, 1)
        R_2 = TR_R_n(pre, ref, 2)
        R_5 = TR_R_n(pre, ref, 5)
        print('R_1:', R_1, '  R_2:', R_2, '   R_5:', R_5)