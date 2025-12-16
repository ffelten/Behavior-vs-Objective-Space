#!/bin/bash

# Transformer

# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHW --epochs 100 --train  --seed $seed --use_set_encoder --train_set --save_data --set_epochs 200
# done

# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHo2 --epochs 100 --train  --seed $seed --use_set_encoder --train_set --save_data --set_epochs 200
# done


# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHC --epochs 101 --train  --seed $seed --save_data  --info_weight 0.0 --dim_weight 0.0 --recon_weight 0.0
# done

# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHo --epochs 101 --train  --seed $seed --save_data  --info_weight 0.0 --dim_weight 0.0 --recon_weight 0.0
# done


# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHW --epochs 100 --train  --seed $seed --save_data --use_mlp_baseline --model_prefix mlp
# done

# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHo2 --epochs 100 --train  --seed $seed --save_data --use_mlp_baseline --model_prefix mlp
# done

# MLP Baseline

for seed in 0 1 2 3 4; do
    python new_main_tr.py -MHC --epochs 100 --train  --seed $seed --save_data --use_mlp_baseline --model_prefix basic --info_weight 1.0 --dim_weight 1.0 --recon_weight 1.0
done

for seed in 0 1 2 3 4; do
    python new_main_tr.py -MHo --epochs 100 --train  --seed $seed --save_data --use_mlp_baseline --model_prefix basic --info_weight 1.0 --dim_weight 1.0 --recon_weight 1.0
done