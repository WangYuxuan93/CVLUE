import json
import cv2


# 图片 id 和 路径保存在 sample.json 文件中
# 图片都存储在 pics 文件夹内
# 最终标记后的新图片也保存在 pics 文件内，前面加上前缀 result_ 以区分

sample_path = 'sample.json'


with open(sample_path, 'r',encoding='utf-8') as f:
    string = f.read()
    samples = json.loads(string)

for sample in samples:
    image = sample['image']
    ref_id = sample['ref_id']
    
    new_image_index = image[-14:].find('/')
    new_image = image[-14:][new_image_index+1:]
    image_path = 'pics/' + new_image
    image = cv2.imread(image_path)

    gold_bbox = sample['bbox']
    gold_x_l_up = gold_bbox[0]  
    gold_y_l_up = gold_bbox[1] - gold_bbox[3]
    gold_x_r_down = gold_bbox[0] + gold_bbox[2]
    gold_y_r_down = gold_bbox[1]

    

    cv2.rectangle(image, (int(gold_x_l_up), int(gold_y_l_up)), (int(gold_x_r_down), int(gold_y_r_down)),(255, 0, 0) ,2)

    cv2.imwrite('pics/' + 'result_' + new_image, image)