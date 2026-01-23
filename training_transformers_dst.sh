#!/bin/bash


for seed in {0..4}; do
    python new_main_tr.py -DSTS --epochs 100 --train --seed $seed --d_hid 32 --save_data 
    python new_main_tr.py -DSTLR --epochs 100 --train --seed $seed --d_hid 32 --save_data 
done

# Baseline model

for seed in {0..4}; do
    python new_main_tr.py -DSTS --epochs 100 --train  --seed $seed --save_data  --d_hid 32 --use_mlp_baseline --model_prefix basic
    python new_main_tr.py -DSTLR --epochs 100 --train  --seed $seed --save_data --d_hid 32 --use_mlp_baseline --model_prefix basic
done