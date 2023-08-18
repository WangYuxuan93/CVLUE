main=/work/experiments/transformers/codes/run_new_vg.py
model=/work/models/Taiyi-BLIP-full-Chinese
cache=/tmp
model_type=blip

train=/work/data/cvlue/VG/VG_train.json
valid=/work/data/cvlue/VG/VG_val.json
test=/work/data/cvlue/VG/VG_test.json
predict_result_path=/work/experiments/transformers/saves/taiyi_blip-vg-base-finetuned/result_ch_blip.json

output_dir=/work/experiments/transformers/saves/taiyi_blip-vg-base-finetuned

python $main \
	--output_dir $output_dir \
	--model_name_or_path $model \
    --cache_dir $cache \
	--model_type $model_type \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--remove_unused_columns=False \
	--label_names target_bbox \
	--do_train  \
	--do_eval \
	--do_predict \
	--predict_result_path $predict_result_path \
	--gradient_accumulation_steps="32" \
	--per_device_train_batch_size="8" \
	--per_device_eval_batch_size="8" \
	--save_steps="65" \
	--logging_steps="65" \
	--eval_steps="65" \
	--learning_rate="1e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--num_train_epochs="10" \
	--overwrite_output_dir True \
	--push_to_hub &> logs/taiyi_blip-vg-base-finetuned.log
