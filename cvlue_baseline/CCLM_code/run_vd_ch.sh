
model=../CCLM/cclm_4m_epoch_29.th


python3 run.py --dist "gpu0" \
               --task vd \
               --config configs/cclm-base-ft/VD_ch.yaml \
               --output_dir output_ch/vd \
               --bs 16 \
               --seed 42 \
               --epoch 10 \
               --checkpoint $model \
               --accumulate_steps 8 \
               --master_port 10010
