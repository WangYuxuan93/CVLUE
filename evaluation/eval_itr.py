import json, argparse

def eval_itr_json(gold_file, pre_i2t_file, pre_t2i_file):

    with open(gold_file, 'r', encoding='utf-8') as f:
        string = f.read()
        gold_data = json.loads(string)

    with open(pre_i2t_file, 'r', encoding='utf-8') as f:
        string = f.read()
        pre_i2t_data = json.loads(string)

    with open(pre_t2i_file, 'r', encoding='utf-8') as f:
        string = f.read()
        pre_t2i_data = json.loads(string)

    i2t_1, i2t_5, i2t_10 = 0, 0, 0
    t2i_1, t2i_5, t2i_10 = 0, 0, 0

    gold_t2i_data = {}
    gold_i2t_data = {}
    check_image = {}
    check_caption = {}

    for item in gold_data:
        img_path = item['image']
        gold_i2t_data[img_path] = item['caption']
        check_image[img_path] = 0

        for caption in item['caption']:
            if caption not in gold_t2i_data:
                gold_t2i_data[caption] = []
            if img_path not in gold_t2i_data[caption]:
                gold_t2i_data[caption].append(img_path)

            if caption not in check_caption:
                check_caption[caption] = 0

    for image, pre_cap in pre_i2t_data.items():
        gold_ = gold_i2t_data[image]
        pre_rank = 100
        check_image[image] = 1

        #assert len(pre_cap) == 10
        for gold_cap in gold_:
            if gold_cap in pre_cap:
                tmp_rank = pre_cap.index(gold_cap)
                if tmp_rank < pre_rank:
                    pre_rank = tmp_rank
        
        if pre_rank < 1:
            i2t_1 += 1
        if pre_rank < 5:
            i2t_5 += 1
        if pre_rank < 10:
            i2t_10 += 1

    #for image, flag in check_image.items():
    #    if flag != 1:
    #        print(image, ' not exists!')
    sum_image = len(pre_i2t_data)
    print('Image-to-Text Results ({} examples):'.format(sum_image))
    print('i2t_r1: ', i2t_1 / sum_image)
    print('i2t_r5: ', i2t_5 / sum_image)
    print('i2t_r10: ', i2t_10 / sum_image)
    i2t_scores = {
            'i2t_r1': i2t_1 / sum_image,
            'i2t_r5': i2t_5 / sum_image,
            'i2t_r10': i2t_10 / sum_image,
            'n_img': sum_image,
        }

    for txt, pre_images in pre_t2i_data.items():
        if txt not in gold_t2i_data:
            txt = ",".join(txt.split())
        gold_ = gold_t2i_data[txt]
        pre_rank = 100
        check_caption[txt] = 1

        #assert len(pre_images) == 10

        for gold_img in gold_:
            if gold_img in pre_images:
                tmp_rank = pre_images.index(gold_img)
                if tmp_rank < pre_rank:
                    pre_rank = tmp_rank
        
        if pre_rank < 1:
            t2i_1 += 1
        if pre_rank < 5:
            t2i_5 += 1
        if pre_rank < 10:
            t2i_10 += 1

    #for caption, flag in check_caption.items():
    #    if flag != 1:
    #        print(caption, ' not exists!')

    sum_txt = len(pre_t2i_data)
    print('Text-to-Image Results ({} examples):'.format(sum_txt))
    print('t2i_r1: ', t2i_1 / sum_txt)
    print('t2i_r5: ', t2i_5 / sum_txt)
    print('t2i_r10: ', t2i_10 / sum_txt)
    t2i_scores = {
        't2i_r1': t2i_1 / sum_txt,
        't2i_r5': t2i_5 / sum_txt,
        't2i_r10': t2i_10 / sum_txt,
        'n_txt': sum_txt,
        }
    return i2t_scores, t2i_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--gold_file", required=True, help="path to gold json file.")
    parser.add_argument("--pred_i2t_file", required=True, help="path to predicted i2t json file.")
    parser.add_argument("--pred_t2i_file", required=True, help="path to predicted t2i json file.")
    args = parser.parse_args()
    eval_itr_json(args.gold_file, args.pred_i2t_file, args.pred_t2i_file)

#gold_file = 'new_pred/test_gold/IC/IC_test_gold.json'
#pre_i2t_file = 'new_pred/2024-5-28 itr result/X2VLM/i2t_result.json'
#pre_t2i_file = 'new_pred/2024-5-28 itr result/X2VLM/t2i_result.json'
#eval_itr_json(gold_file, pre_i2t_file, pre_t2i_file)
