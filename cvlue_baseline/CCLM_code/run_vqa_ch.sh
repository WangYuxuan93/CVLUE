

model=../CCLM/cclm_4m_epoch_29.th

python3 run.py --dist "gpu0" \
               --task gqa_ch \
               --config configs/cclm-base-ft/VQA_ch.yaml \
               --output_dir output_ch/vqa \
               --bs 16 \
               --seed 42 \
               --epoch 10 \
               --checkpoint $model \
               --accumulate_steps 8 \
               --master_port 10015