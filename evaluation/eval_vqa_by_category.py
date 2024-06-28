import json, os, argparse
import jsonlines
from eval_vqa import eval_vqa_json, merge_vqa

parser = argparse.ArgumentParser(description="Demo")
#parser.add_argument("--input_path", help="path to input json file.")
parser.add_argument("--gold_path", required=True, help="path to gold json file.")
parser.add_argument("--pred_path", required=True, help="path to pred json file.")
parser.add_argument("--output_path", default="vqa_score_by_category.json", help="path to output score json file.")
parser.add_argument("--tmp_dir", default="tmp/", help="path to input json file.")
parser.add_argument("--zh_key", action="store_true", help="filter text with chinese key words")
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


zh_key_map= {'1-panda': ['熊猫'], '2-cow': ['牛'], '3-fish': ['鱼'], '4-dog': ['狗'], '5-horse': ['马'], '6-chicken': ['鸡'], '7-mouse': ['鼠'],
             '8-bird': ['鸟'], '9-human': ['人'], '10-cat': ['猫'], '11-hot_pot': ['火锅'], '12-rice': ['米'], '13-dumpling': ['饺子'], '14-noodles': ['面'],
             '15-baozi': ['包子'], '16-milk_tea': ['奶茶'], '17-coke': ['可乐'], '18-milk': ['牛奶'], '19-tea': ['茶'], '20-porridge': ['粥'],
             '21-alcohol': ['酒'], '22-hanfu': ['汉服'], '23-tangzhuang': ['唐装'], '24-chi_pao': ['旗袍'], '25-suit': ['西装'], '26-t_shirt': ['T恤'],
             '27-willow': ['柳'], '28-ginkgo': ['银杏'], '29-sycamore': ['梧桐'], '30-birch': ['白桦'], '31-pine': ['松'], '32-chrysanthemum': ['菊'],
             '33-peony': ['牡丹'], '34-orchid': ['兰'], '35-lotus': ['莲'], '36-lily': ['百合'], '37-lychee': ['荔枝'], '38-hawthorn': ['山楂'],
             '39-apple': ['苹果'], '40-cantaloupe': ['哈密瓜'], '41-longan': ['龙眼'], '42-xiaobaicai': ['小白菜'], '43-potato': ['马铃薯','土豆'],
             '44-dabaicai': ['大白菜'], '45-carrot': ['胡萝卜'], '46-cauliflower': ['花椰菜','花菜'], '47-hoe': ['锄'], '48-plow': ['犁'], '49-harrow': ['耙'],
             '50-sickle': ['镰刀'], '51-staff': ['担杖','担'], '52-spoon': ['勺'], '53-bowl': ['碗'], '54-cutting_board': ['砧板'], '55-chopsticks': ['筷子'],
             '56-wok': ['锅'], '57-fan': ['扇子'], '58-chinese_cleaver': ['菜刀'], '59-spatula': ['铲'], '60-tv': ['电视'], '61-table': ['桌'],
             '62-chair': ['椅'], '63-refrigerator': ['冰箱'], '64-stove': ['灶台'], '65-ping_pong': ['乒乓'], '66-basketball': ['篮球'],
             '67-swimming': ['游泳'], '68-football': ['足球'], '69-running': ['跑'], '70-lion_dance': ['舞狮'], '71-dragon_boat': ['龙舟'],
             '72-national_flag': ['国旗'], '73-mooncake': ['月饼'], '74-couplet': ['联'], '75-lantern': ['灯'], '76-pencil': ['铅笔'],
             '77-blackboard': ['黑板'], '78-brush_pen': ['毛笔'], '79-chalk': ['粉笔'], '80-ballpen': ['原子笔'], '81-scissors': ['剪刀'],
             '82-guzheng': ['古筝'], '83-erhu': ['二胡'], '84-suona': ['唢呐'], '85-drums': ['鼓'], '86-pipa': ['琵琶'], '87-calligraphy': ['书法'],
             '88-shadowplay': ['皮影'], '89-papercutting': ['剪纸'], '90-bingmayong': ['兵马俑','俑'], '91-tripod': ['鼎'], '92-ceramic': ['陶','瓷']}

merged_file = merge_vqa(args.gold_path, args.pred_path)

result_by_cat = {cat : [] for cat in categories}
filtered_output = []
with open(merged_file, "r", encoding="utf-8") as f:
    data = json.load(f)
    for item in data:
        img_path = item["image"]
        cat = get_category(img_path)
        if args.zh_key:
            zh_keys = zh_key_map[cat]
            has_zh_key = False
            for key_word in zh_keys:
                if key_word in item["question"]:
                    has_zh_key = True
            if not has_zh_key: continue
        
        result_by_cat[cat].append(item)
        filtered_output.append(item)

cat_num = {cat: len(res) for cat, res in result_by_cat.items()}
#print (cat_num)
file_name = os.path.basename(merged_file)
if not os.path.exists(args.tmp_dir):
    os.makedirs(args.tmp_dir)
scores = {}
print ("###################\nEvaluating the whole file")
if args.zh_key:
    filter_file_output = os.path.join(args.tmp_dir, file_name.split(".")[0]+"_filtered_all.json")
    with open(filter_file_output, "w", encoding="utf-8") as fo:
        json.dump(filtered_output, fo, indent=2, ensure_ascii=False)
    n_match, n_total, acc = eval_vqa_json(filter_file_output)
else:
    n_match, n_total, acc = eval_vqa_json(merged_file)
scores["overall"] = {
        "n_match": n_match,
        "n_tot": n_total,
        "acc": acc
    }
num_cnt = 0
for cat in result_by_cat:
    result = result_by_cat[cat]
    if len(result) == 0: continue
    output_file = os.path.join(args.tmp_dir, file_name.split(".")[0]+"_"+cat+".json")
    with open(output_file, "w", encoding="utf-8") as fo:
        json.dump(result, fo, indent=2, ensure_ascii=False)
    print ("###################\nEvaluating category: {} ({} examples)".format(cat, len(result)))
    n_match, n_tot, acc = eval_vqa_json(output_file)
    scores[cat] = {
        "n_match": n_match,
        "n_tot": n_tot,
        "acc": acc
    }
    num_cnt += n_tot
if not args.zh_key:
    assert num_cnt == n_total
print ("Remained/Total={}/{}".format(num_cnt, n_total))
with open(args.output_path, "w") as fs:
    json.dump(scores, fs, indent=2)
