

python3 run.py \
    --task "itr_coco_mm" \
    --dist "gpu0" \
    --config "configs/finetune/ITR_ch.yaml" \
    --checkpoint "../X2-VLM/cclm_x2vlm_base.th"  \
    --output_dir "output_ch/itr" \
    --master_port 11199 \
    --bs 16 \
    --seed 42 \
    --epoch 10 \
    --accumulate_steps 8
