
python3 run.py \
    --task "vd" \
    --dist "gpu0" \
    --config "configs/finetune/VD_ch.yaml" \
    --checkpoint "../X2-VLM/cclm_x2vlm_base.th"  \
    --output_dir "output_ch/vd" \
    --bs 8 \
    --seed 42 \
    --epoch 10 \
    --master_port 10042 \
    --accumulate_steps 16
