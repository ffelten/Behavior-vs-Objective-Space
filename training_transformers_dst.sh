#!/bin/bash


for seed in 0 1 2 3 4; do
    python new_main_tr.py -DSTS --epochs 100 --train  --seed $seed --save_data --with_lines --d_hid 64
    python new_main_tr.py -DSTLR --epochs 100 --train  --seed $seed --save_data --with_lines --d_hid 32
done