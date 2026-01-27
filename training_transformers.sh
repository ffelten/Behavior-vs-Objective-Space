#!/bin/bash

# Transformer

for seed in {0..9}; do
    python new_main_tr.py -MHC --epochs 100 --train --seed $seed --save_data
done

for seed in {0..9}; do
    python new_main_tr.py -MHo --epochs 100 --train --seed $seed --save_data
done

# Baseline model

for seed in {0..9}; do
    python new_main_tr.py -MHC --epochs 100 --train --seed $seed --save_data --use_mlp_baseline --model_prefix basic
done

for seed in {0..9}; do
    python new_main_tr.py -MHo --epochs 100 --train --seed $seed --save_data --use_mlp_baseline --model_prefix basic
done