## Evaluation on All Examples

### ITR

python .\eval_itr.py --gold_file ..\test_example\IC\gold.json --pred_i2t_file ..\test_example\IC\i2t_submit.json --pred_t2i_file ..\test_example\IC\t2i_submit.json

### VQA

python .\eval_vqa.py --gold_path ..\test_example\VQA\gold.json --pred_path ..\test_example\VQA\submit_VQA.json [--strict_match] (This option controls whether to enforce strict matching.)

### VG

 python .\eval_vg.py --gold_path ..\test_example\VG\gold.json --pred_path ..\test_example\VG\submit_VG.json

### VD

python .\eval_vd.py --gold_path ..\test_example\VD\gold.json --pred_path ..\test_example\VD\submit_VD.json

## Evaluation by Category

### ITR

python .\eval_itr_by_category.py --input_gold_path ..\test_example\IC\gold.json --input_i2t_path ..\test_example\IC\i2t_submit.json --input_t2i_path ..\test_example\IC\t2i_submit.json --output_i2t_path scores/i2t_scores.json --output_t2i_path scores/t2i_scores.json

### VQA

python .\eval_vqa_by_category.py --gold_path ..\test_example\VQA\gold.json --pred_path ..\test_example\VQA\submit_VQA.json --output_path .\scores\vqa_scores.json

### VG

python .\eval_vg_by_category.py --gold_path ..\test_example\VG\gold.json --pred_path ..\test_example\VG\submit_VG.json --output_path .\scores\vg_scores.json

### VD

python .\eval_vd_by_category.py --gold_path ..\test_example\VD\gold.json --pred_path ..\test_example\VD\submit_VD.json --output_path .\scores\vd_scores.json
