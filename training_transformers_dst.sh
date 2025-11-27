#!/bin/bash


for seed in 0 1 2 3 4; do
    python new_main_tr.py -DSTS --epochs 100 --train  --seed $seed --save_data --d_hid 32 --with_lines
    python new_main_tr.py -DSTLR --epochs 100 --train  --seed $seed --save_data --d_hid 32 --with_lines
    python new_main_tr.py -DSTL --epochs 100 --train  --seed $seed --save_data --d_hid 32 --with_lines
done