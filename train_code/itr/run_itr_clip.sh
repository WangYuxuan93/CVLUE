main=/work/experiments/transformers/codes/run_retrieval.py
model=/work/models/taiyi-clip-roberta-new
# model=/work/models/taiyi-clip-roberta
# output_dir=./clip_taiyi-vd-base-finetuned
# model=/work/models/clip-chinese-roberta 
output_dir=/work/experiments/transformers/saves/taiyi_clip-itr-base-finetuned
cache=/tmp
train=/work/data/cvlue/IC/IC_train.json
valid=/work/data/cvlue/IC/IC_val.json
test=/work/data/cvlue/IC/IC_test.json
model_type=clip

predict_i2t_result_path=/work/experiments/transformers/saves/taiyi_clip-itr-base-finetuned/result_i2t_ch_clip.json
predict_t2i_result_path=/work/experiments/transformers/saves/taiyi_clip-itr-base-finetuned/result_t2i_ch_clip.json

python $main \
	--output_dir $output_dir \
	--model_name_or_path $model \
	--model_type $model_type \
    --cache_dir $cache \
	--train_file $train \
	--validation_file $valid \
	--test_file $test \
	--predict_i2t_result_path $predict_i2t_result_path \
	--predict_t2i_result_path $predict_t2i_result_path \
    --image_column image \
    --caption_column caption \
	--remove_unused_columns=False \
	--do_train  \
	--do_eval \
	--do_predict \
	--save_steps="72" \
	--logging_steps="72" \
	--eval_steps="72" \
	--gradient_accumulation_steps="2" \
	--per_device_train_batch_size="128" \
	--per_device_eval_batch_size="128" \
	--learning_rate="1e-5" --warmup_steps="0" --weight_decay 0.05 \
	--lr_scheduler_type cosine \
	--num_train_epochs="6" \
	--push_to_hub &> logs/taiyi_clip-itr-base-finetuned.log

