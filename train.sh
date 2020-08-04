#!/usr/bin/env bash
save_model=./models2/models
logs=./models2/log0.txt
lr=0.001

CUDA_VISIBLE_DEVICES=0 python3 -u train_model.py --model_dir=${save_model} \
                                                --learning_rate=${lr} \
                                                --lr_epoch='10,20,40,300,400' \
				                --level=L1 \
				                --image_size=28 \
				                --image_channels=3 \
				                --batch_size=64 \
					        --max_epoch=500 \
                                                > ${logs} 2>&1 &
tail -f ${logs}
