import json, os, argparse
import jsonlines
from eval_itr import eval_itr_json
from collections import Counter

parser = argparse.ArgumentParser(description="Demo")
parser.add_argument("--input_gold_path", default="new_pred/test_gold/IC/IC_test_gold.json", help="path to input json file.")
parser.add_argument("--input_i2t_path", help="path to input json file.")
parser.add_argument("--input_t2i_path", help="path to input json file.")
parser.add_argument("--output_i2t_path", default="vd_score_by_category.json", help="path to output score json file.")
parser.add_argument("--output_t2i_path", default="vd_score_by_category.json", help="path to output score json file.")
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

i2t_result_by_cat = {cat : {} for cat in categories}
with open(args.input_i2t_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    for img_path in data:
        cat = get_category(img_path)
        i2t_result_by_cat[cat][img_path] = data[img_path]

i2t_cat_num = {cat: len(res) for cat, res in i2t_result_by_cat.items()}
#print (i2t_cat_num)
i2t_file_name = os.path.basename(args.input_i2t_path)

t2i_result_by_cat = {cat : {} for cat in categories}
with open(args.input_t2i_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    for text in data:
        img_paths = data[text]
        cat_list = [get_category(img_path) for img_path in img_paths]
        cat = Counter(cat_list).most_common()[0][0]
        t2i_result_by_cat[cat][text] = data[text]

t2i_cat_num = {cat: len(res) for cat, res in t2i_result_by_cat.items()}
#print (t2i_cat_num)
t2i_file_name = os.path.basename(args.input_t2i_path)

if not os.path.exists(args.tmp_dir):
    os.makedirs(args.tmp_dir)
scores_i2t, scores_t2i = {}, {}
print ("###################\nEvaluating the whole file")
overall_i2t_scores, overall_t2i_scores = eval_itr_json(args.input_gold_path, args.input_i2t_path, args.input_t2i_path)
scores_i2t["overall"] = overall_i2t_scores
scores_t2i["overall"] = overall_t2i_scores

#print (scores)

num_cnt = 0
for cat in i2t_result_by_cat:
    i2t_result = i2t_result_by_cat[cat]
    if len(i2t_result) == 0: continue
    i2t_output_file = os.path.join(args.tmp_dir, i2t_file_name.split(".")[0]+"_"+cat+".json")
    with open(i2t_output_file, "w", encoding="utf-8") as fo:
        json.dump(i2t_result, fo, indent=2, ensure_ascii=False)
    t2i_result = t2i_result_by_cat[cat]
    if len(t2i_result) == 0: continue
    t2i_output_file = os.path.join(args.tmp_dir, t2i_file_name.split(".")[0]+"_"+cat+".json")
    with open(t2i_output_file, "w", encoding="utf-8") as fo:
        json.dump(t2i_result, fo, indent=2, ensure_ascii=False)
    print ("###################\nEvaluating category: {} ({}/{} i2t/t2i examples)".format(cat, len(i2t_result), len(t2i_result)))
    i2t_scores, t2i_scores = eval_itr_json(args.input_gold_path, i2t_output_file, t2i_output_file)
    scores_i2t[cat] = i2t_scores
    scores_t2i[cat] = t2i_scores
    #num_cnt += n_tot
#assert num_cnt == n_total
output_dir = os.path.dirname(args.output_i2t_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_dir = os.path.dirname(args.output_t2i_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(args.output_i2t_path, "w") as fs:
    json.dump(scores_i2t, fs, indent=2)
with open(args.output_t2i_path, "w") as fs:
    json.dump(scores_t2i, fs, indent=2)
