python .\eval_itr.py --gold_file ..\test_example\IC\gold.json --pred_i2t_file ..\test_example\IC\i2t_submit.json --pred_t2i_file ..\test_example\IC\t2i_submit.json

python .\eval_itr_by_category.py --input_gold_path ..\test_example\IC\gold.json --input_i2t_path ..\test_example\IC\i2t_submit.json --input_t2i_path ..\test_example\IC\t2i_submit.json --output_i2t_path scores/i2t_scores.json --output_t2i_path scores/t2i_scores.json

python .\eval_vqa.py --gold_path ..\test_example\VQA\gold.json --pred_path ..\test_example\VQA\submit_VQA.json [--strict_match] 这个可以控制是否严格匹配

python .\eval_vqa_by_category.py --gold_path ..\test_example\VQA\gold.json --pred_path ..\test_example\VQA\submit_VQA.json --output_path .\scores\vqa_scores.json

