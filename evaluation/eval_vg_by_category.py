import json, os, argparse
import jsonlines
from eval_vg import eval_vg_json, merge_vg

parser = argparse.ArgumentParser(description="Demo")
#parser.add_argument("--input_path", help="path to input json file.")
parser.add_argument("--gold_path", required=True, help="path to gold json file.")
parser.add_argument("--pred_path", required=True, help="path to pred json file.")
parser.add_argument("--output_path", default="vg_score_by_category.json", help="path to output score json file.")
parser.add_argument("--tmp_dir", default="tmp/", help="path to input json file.")
args = parser.parse_args()

def get_category(path):
    path_splits = path.strip().split("/")
    cat = path_splits[-2]
    return cat

categories = ['1-panda', '2-cow', '3-fish', '4-dog', '5-horse', '6-chicken', '7-mouse', '8-bird', '9-human', 
              '10-cat', '11-hot_pot', '12-rice', '13-dumpling', '14-noodles', '15-baozi', '16-milk_tea', '17-coke', '18-milk', '19-tea', 
              '20-porridge', '21-alcohol', '22-hanfu', '23-tangzhuang', '24-chi_pao', '25-suit', '26-t_shirt', '27-willow', '28-ginkgo', '29-sycamore', 
              '30-birch', '31-pine', '32-chrysanthemum', '33-peony', '34-orchid', '35-lotus', '36-lily', '37-lychee', '38-hawthorn', '39-apple', 
              '40-cantaloupe', '41-longan', '42-xiaobaicai', '43-potato', '44-dabaicai', '45-carrot', '46-cauliflower', '47-hoe', '48-plow', '49-harrow', 
              '50-sickle', '51-staff', '52-spoon', '53-bowl', '54-cutting_board', '55-chopsticks', '56-wok', '57-fan', '58-chinese_cleaver', '59-spatula', 
              '60-tv', '61-table', '62-chair', '63-refrigerator', '64-stove', '65-ping_pong', '66-basketball', '67-swimming', '68-football', '69-running', 
              '70-lion_dance', '71-dragon_boat', '72-national_flag', '73-mooncake', '74-couplet', '75-lantern', '76-pencil', '77-blackboard', '78-brush_pen', '79-chalk', 
              '80-ballpen', '81-scissors', '82-guzheng', '83-erhu', '84-suona', '85-drums', '86-pipa', '87-calligraphy', '88-shadowplay', '89-papercutting', 
              '90-bingmayong', '91-tripod', '92-ceramic']

merged_file = merge_vg(args.gold_path, args.pred_path)

result_by_cat = {cat : [] for cat in categories}

with open(merged_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        img_path = item["image"]
        cat = get_category(img_path)
        result_by_cat[cat].append(item)

cat_num = {cat: len(res) for cat, res in result_by_cat.items()}
#print (cat_num)
file_name = os.path.basename(merged_file)
if not os.path.exists(args.tmp_dir):
    os.makedirs(args.tmp_dir)
scores = {}
print ("###################\nEvaluating the whole file")
n_total, avg_iou = eval_vg_json(merged_file)
scores["overall"] = {
        "n_tot": n_total,
        "iou": avg_iou, 
    }
num_cnt = 0
for cat in result_by_cat:
    result = result_by_cat[cat]
    if len(result) == 0: continue
    output_file = os.path.join(args.tmp_dir, file_name.split(".")[0]+"_"+cat+".json")
    with open(output_file, "w", encoding="utf-8") as fo:
        json.dump(result, fo, indent=2, ensure_ascii=False)
    print ("###################\nEvaluating category: {} ({} examples)".format(cat, len(result)))
    n_tot, avg_iou = eval_vg_json(output_file)
    scores[cat] = {
        "n_tot": n_tot,
        "iou": avg_iou, 
    }
    num_cnt += n_tot
assert num_cnt == n_total
with open(args.output_path, "w") as fs:
    json.dump(scores, fs, indent=2)
