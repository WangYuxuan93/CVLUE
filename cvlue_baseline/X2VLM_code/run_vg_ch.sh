

python3 run.py \
    --task "vg_ch" \
    --dist "gpu0" \
    --config "configs/finetune/VG_ch.yaml" \
    --checkpoint "../X2-VLM/cclm_x2vlm_base.th"  \
    --output_dir "output_ch/vg" \
    --master_port 11119 \
    --bs 32 \
    --seed 42 \
    --epoch 10 \
    --accumulate_steps 4