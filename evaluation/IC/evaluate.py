import json
from tqdm import tqdm

gold_file = 'IC_test_gold.json'

with open(gold_file, 'r', encoding='utf-8') as f:
    string = f.read()
    gold_data = json.loads(string)

pre_i2t_file = 'i2t_result.json'
pre_t2i_file = 't2i_result.json'

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

for image, pre_cap in tqdm(pre_i2t_data.items()):
    gold_ = gold_i2t_data[image]
    pre_rank = 100
    check_image[image] = 1

    assert len(pre_cap) == 10
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

for image, flag in check_image.items():
    if flag != 1:
        print(image, ' not exists!')
    
sum_image = len(pre_i2t_data)
print(sum_image)
print('i2t_r1: ', i2t_1 / sum_image)
print('i2t_r5: ', i2t_5 / sum_image)
print('i2t_r10: ', i2t_10 / sum_image)

for txt, pre_images in tqdm(pre_t2i_data.items()):

    gold_ = gold_t2i_data[txt]
    pre_rank = 100
    check_caption[txt] = 1

    assert len(pre_images) == 10

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

for caption, flag in check_caption.items():
    if flag != 1:
        print(caption, ' not exists!')

sum_txt = len(pre_t2i_data)
print(sum_txt)
print('t2i_r1: ', t2i_1 / sum_txt)
print('t2i_r5: ', t2i_5 / sum_txt)
print('t2i_r10: ', t2i_10 / sum_txt)





