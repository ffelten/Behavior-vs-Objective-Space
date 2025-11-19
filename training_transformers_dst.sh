#!/bin/bash


for seed in 0 1 2 3 4; do
    python new_main_tr.py -DSTS --epochs 200 --train  --seed $seed --save_data --with_lines
done

for seed in 0 1 2 3 4; do
    python new_main_tr.py -DSTLR --epochs 200 --train  --seed $seed --save_data --with_lines
done