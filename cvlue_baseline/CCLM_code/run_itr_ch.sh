
model=../CCLM/cclm_4m_epoch_29.th


python3 run.py --dist "gpu0" \
               --task itr_coco \
               --config configs/cclm-base-ft/ITR_ch.yaml \
               --output_dir output_ch/itr \
               --bs 16 \
               --seed 42 \
               --epoch 10 \
               --checkpoint $model \
               --accumulate_steps 8 \
               --master_port 10012