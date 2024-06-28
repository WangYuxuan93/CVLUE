import json, os, argparse

def merge_vd(gold_path, pred_path, tmp_dir="tmp/"):
    preds = []
    with open(pred_path, "r", encoding="utf-8") as fp:
        preds = json.load(fp)

    with open(gold_path, "r", encoding="utf-8") as fg:
        data = json.load(fg)["data"]
    pred_name = os.path.basename(pred_path)
    output_file = os.path.join(tmp_dir, "merged_"+pred_name)
    output = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for gold, pred in zip(data, preds):
            assert gold["image"] == pred["image"]
            assert gold["dialog_id"] == pred["dialog_id"]
            gold["pred"] = pred["answer_sort"]
            output.append(gold)
        json.dump(output, fo, indent=2, ensure_ascii=False)
    return output_file

def eval_vd_json(input_path):
    n_tot, n_top1, n_top5, n_top10 = 0, 0, 0, 0
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            n_tot += 1
            topk_ids = item["pred"][:10]
            ans_id = item["answer"]
            if ans_id == topk_ids[0]:
                n_top1 += 1
            if ans_id in topk_ids[:5]:
                n_top5 += 1
            if ans_id in topk_ids[:10]:
                n_top10 += 1

    r1 = float(n_top1) / n_tot
    r5 = float(n_top5) / n_tot
    r10 = float(n_top10) / n_tot
    print ("Top1/Top5/Top10/Total={}/{}/{}/{}".format(n_top1, n_top5, n_top10, n_tot))
    print ("R@1={},R@5={},R@10={}".format(r1, r5, r10))
    return n_tot, r1, r5, r10

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    #parser.add_argument("--input_path", required=True, help="path to input json file.")
    parser.add_argument("--gold_path", required=True, help="path to gold json file.")
    parser.add_argument("--pred_path", required=True, help="path to pred json file.")
    args = parser.parse_args()
    merged_file = merge_vd(args.gold_path, args.pred_path)
    eval_vd_json(merged_file)
