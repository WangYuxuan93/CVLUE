

model=../CCLM/cclm_4m_epoch_29.th


python3 run.py \
    --task "vg_ch" \
    --dist "gpu0" \
    --config "configs/cclm-base-ft/VG_ch.yaml" \
    --checkpoint $model  \
    --output_dir "output_ch/vg" \
    --master_port 11120 \
    --bs 32 \
    --seed 42 \
    --epoch 10 \
    --accumulate_steps 4