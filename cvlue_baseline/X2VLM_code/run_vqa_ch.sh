

python3 run.py \
    --task "xgqa_ch" \
    --dist "gpu0" \
    --config "configs/finetune/VQA_ch.yaml" \
    --checkpoint "../X2-VLM/cclm_x2vlm_base.th"  \
    --output_dir "output_ch/vqa" \
    --bs 8 \
    --seed 42 \
    --epoch 5 \
    --master_port 10010 \
    --accumulate_steps 16