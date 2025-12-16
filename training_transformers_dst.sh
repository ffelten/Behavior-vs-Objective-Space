#!/bin/bash


# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -DSTS --epochs 101 --seed $seed --d_hid 32 --with_lines --train
#     python new_main_tr.py -DSTLR --epochs 101 --seed $seed --d_hid 32 --with_lines --train
#     # python new_main_tr.py -DSTL --epochs 100 --seed $seed --d_hid 32 --with_lines --info_weight 1.0 --dim_weight 1.0 --recon_weight 1.0
# done

for seed in 0 1 2 3 4; do
    python new_main_tr.py -DSTS --epochs 100 --train  --seed $seed --save_data --d_hid 32 --use_mlp_baseline --model_prefix basic --info_weight 1.0 --dim_weight 1.0 --recon_weight 1.0
    python new_main_tr.py -DSTLR --epochs 100 --train  --seed $seed --save_data --d_hid 32 --use_mlp_baseline --model_prefix basic --info_weight 1.0 --dim_weight 1.0 --recon_weight 1.0
    # python new_main_tr.py -DSTL --epochs 100 --train  --seed $seed --save_data --d_hid 32 --with_lines --use_mlp_baseline --model_prefix mlp
done