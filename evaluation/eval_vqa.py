import json, os, argparse
import re
from string import punctuation

def merge_vqa(gold_file, pred_file, tmp_dir="tmp/"):
    with open(pred_file, "r", encoding="utf-8") as fp:
        preds = json.load(fp)

    with open(gold_file, "r", encoding="utf-8") as fg:
        data = json.load(fg)
    #print (len(data))
    #print (len(preds))

    pred_name = os.path.basename(pred_file)
    output_file = os.path.join(tmp_dir, "merged_"+pred_name)
    output = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for gold, pred in zip(data, preds):
            assert gold["question_id"] == pred["question_id"]
            gold["pred"] = pred["answer"]
            output.append(gold)
        json.dump(output, fo, indent=2, ensure_ascii=False)
    return output_file

def extract_numbers(s):
    return re.findall(r'\d+', s)

def is_english_string(string):
    return bool(re.match(r'^[a-zA-Z\s]+$', string))

def eval_vqa_json(input_path, strict_match=False):
    n_tot, n_match = 0, 0
    n_loose_match = 0
    n_en_ans = 0
    negatives = ["否","不是","没有","不一样","不一致","不相同", "不"]
    #with open(args.input_path, "r", encoding="utf-8") as f:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            if is_english_string(item["pred"]):
                n_en_ans += 1
            match_flag = False
            lmatch_flag = False
            n_tot += 1
            if item["pred"] is None: continue
            ans_numbers = extract_numbers(item["pred"])
            pred = item["pred"].lower().strip().strip(punctuation).strip("。")
            if pred.endswith("？"): continue
            if pred.endswith("吗"): continue
            #if item["pred"].lower().strip().strip(punctuation).strip("。") == item["answer"].lower().strip().strip(punctuation).strip("。"):
            if pred == item["answer"].lower().strip().strip(punctuation).strip("。"):
                match_flag = True
                #n_match += 1
            elif item["answer"] in ans_numbers:
                match_flag = True
                #n_match += 1
                #print (item, ans_numbers)
            if not strict_match:
                if item["answer"] in ["是","是的","有","有的","一样","一致","相同"] and pred in ["是","是的","有","有的","一样","一致","相同"]:
                    match_flag = True
                    #print (item)
                elif item["answer"] in ["否","不是","没有","不一样","不一致","不相同"] and pred in ["否","不是","没有","不一样","不一致","不相同"]:
                    match_flag = True
                elif item["answer"] in ["是","是的","有","有的","一样","一致","相同"] and item["answer"] in pred:
                    #print (item)
                    has_negative = False
                    for part in negatives:
                        if part in pred:
                            has_negative = True
                    if not has_negative:
                        match_flag = True
                        #print (item)
                elif item["answer"] in ["否","不是","没有","不一样","不一致","不相同"] and item["answer"] in pred:
                    #print (item)
                    match_flag = True
            if match_flag:
                n_match += 1
            if item["answer"].lower().strip().strip(".") in item["pred"].lower().strip().strip("."):
                n_loose_match += 1

    #print ("English/Total Answer={}/{}, {:.2f}".format(n_en_ans,n_tot, float(n_en_ans)*100 / n_tot))
            
    acc = float(n_match) / n_tot
    print ("Match/Total={}/{}, Acc={:.2f}".format(n_match, n_tot, acc*100))
    
    #lacc = float(n_loose_match) / n_tot
    #print ("Loose Match/Total={}/{}, Acc={:.2f}".format(n_loose_match, n_tot, lacc*100))

    return n_match, n_tot, acc #, n_loose_match, lacc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    #parser.add_argument("--input_path", required=True, help="path to input json file.")
    parser.add_argument("--gold_path", required=True, help="path to gold json file.")
    parser.add_argument("--pred_path", required=True, help="path to pred json file.")
    parser.add_argument("--strict_match", action="store_true", help="whether to match strictly")
    args = parser.parse_args()
    merged_file = merge_vqa(args.gold_path, args.pred_path)
    eval_vqa_json(merged_file, strict_match=args.strict_match)
