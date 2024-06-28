import json, os, argparse

def zh_to_coord(box):
    bl_x, bl_y, w, h = box
    x1 = bl_x
    y1 = bl_y - h
    x2 = x1 + w
    y2 = bl_y
    coord = (x1,y1,x2,y2)
    return coord

def en_to_coord(box):
    x1, y1, w, h = box
    x2 = x1 + w
    y2 = y1 + h
    coord = (x1,y1,x2,y2)
    return coord

def qwen_to_coord(box):
    x1, y1 = box[0]
    x2, y2 = box[1]
    return (x1, y1, x2, y2)

def merge_vg(gold_path, pred_path, tmp_dir="tmp/"):
    with open(pred_path, "r", encoding="utf-8") as fp:
        preds = json.load(fp)

    with open(gold_path, "r", encoding="utf-8") as fg:
        data = json.load(fg)
    #print (len(data))
    #print (len(preds))
    pred_name = os.path.basename(pred_path)
    output_file = os.path.join(tmp_dir, "merged_"+pred_name)
    output = []
    with open(output_file, "w", encoding="utf-8") as fo:
        for gold, pred in zip(data, preds):
            assert gold["ref_id"] == pred["ref_id"]
            gold["pred"] = pred["bbox"]
            output.append(gold)
        json.dump(output, fo, indent=2, ensure_ascii=False)
    return output_file

def compute_iou(boxA, boxB):
    #print (boxA, boxB)
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    #print (interArea)
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def eval_vg_json(input_path):
    ious = []
    n_tot = 0
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            n_tot += 1
            pred_cor = item["pred"]
            if len(pred_cor) == 0:
                ious.append(0)
                continue
            pred_coord = en_to_coord(pred_cor)
            gold_coord = en_to_coord(item["bbox"])
            iou = compute_iou(gold_coord, pred_coord)
            #print (item["ref_id"], iou)
            #exit()
            ious.append(iou)
    avg_iou = float(sum(ious)) / n_tot
    print ("Total={}, Avg IoU={}".format(n_tot, avg_iou))
    return n_tot, avg_iou


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    #parser.add_argument("--input_path", required=True, help="path to input json file.")
    parser.add_argument("--gold_path", required=True, help="path to gold json file.")
    parser.add_argument("--pred_path", required=True, help="path to pred json file.")
    args = parser.parse_args()
    merged_file = merge_vg(args.gold_path, args.pred_path)
    eval_vg_json(merged_file)

