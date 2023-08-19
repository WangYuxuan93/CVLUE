main=/work/experiments/transformers/codes/run_new_vd.py
# model=/work/models/Taiyi-BLIP-full-Chinese
model=/work/experiments/transformers/saves/taiyi_blip-vd-base-finetuned/checkpoint-648
cache=/tmp

model_type=blip

train=/work/data/cvlue/VD/new_VD_train.json
valid=/work/data/cvlue/VD/new_VD_val.json
test=/work/data/cvlue/VD/new_VD_test.json
answer_list_path=/work/data/cvlue/VD/vd_answer.json
question_list_path=/work/data/cvlue/VD/vd_question.json
predict_result_path=/work/experiments/transformers/saves/taiyi_blip-vd-base-finetuned/result_ch_blip.json

output_dir=/work/experiments/transformers/saves/taiyi_blip-vd-base-finetuned

python $main \
	--output_dir $output_dir \
	--model_name_or_path $model \
    --cache_dir $cache \
	--model_type $model_type \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--answer_list_file $answer_list_path \
	--question_list_file $question_list_path \
	--predict_result_path $predict_result_path \
	--remove_unused_columns=False \
	--label_names answer_input_ids \
	--do_train  \
	--do_eval \
	--do_predict \
	--gradient_accumulation_steps="32" \
	--per_device_train_batch_size="4" \
	--per_device_eval_batch_size="4" \
	--save_steps="72" \
	--logging_steps="72" \
	--eval_steps="72" \
	--learning_rate="2e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--num_train_epochs="20" \
	--push_to_hub &> logs/taiyi_blip-vd-base-finetuned.log
