main=/work/experiments/transformers/codes/run_vqa_1.py
# model=/work/models/taiyi-clip-roberta
# output_dir=./clip_taiyi-vqa-base-finetuned
# model=/work/models/taiyi-clip-roberta-new-for-vqa
# output_dir=./clip_taiyi-vd-base-finetuned
# model=/work/models/clip-chinese-roberta 

model=/work/experiments/transformers/saves/taiyi_clip-vqa-base-finetuned/checkpoint-600
cache=/tmp
decoder_model=/work/models/Taiyi-CLIP-Roberta-102M-Chinese

model_type=clip

train=/work/data/cvlue/VQA/VQA_train.json
valid=/work/data/cvlue/VQA/VQA_val.json
test=/work/data/cvlue/VQA/VQA_test.json
answer_list=/work/data/cvlue/VQA/answer_list.json
predict_result_path=/work/experiments/transformers/saves/taiyi_clip-vqa-base-finetuned/result_ch_clip.json
output_dir=/work/experiments/transformers/saves/taiyi_clip-vqa-base-finetuned


python $main \
	--output_dir $output_dir \
	--model_name_or_path $model \
	--decoder_model_path $decoder_model\
	--model_type $model_type \
    --cache_dir $cache \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--answer_list $answer_list \
	--predict_result_path $predict_result_path \
	--remove_unused_columns=False \
	--label_names answer_input_ids \
	--do_train  \
	--do_eval \
	--do_predict \
	--gradient_accumulation_steps="8" \
	--per_device_train_batch_size="32" \
	--per_device_eval_batch_size="32" \
	--save_steps="120" \
	--logging_steps="120" \
	--eval_steps="120" \
	--learning_rate="2e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--num_train_epochs="10" \
	--push_to_hub &> logs/taiyi_clip-vqa-base-finetuned.log

