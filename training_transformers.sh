#!/bin/bash

# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHo2 --epochs 200 --train  --seed $seed --use_set_encoder --train_set --save_data --set_epochs 200
# done

# for seed in 0 1 2 3 4; do
#     python new_main_tr.py -MHW --epochs 200 --train  --seed $seed --use_set_encoder --train_set --save_data --set_epochs 200
# done

# for seed in 0 1 2 3 4; do
    # python new_main_tr.py -MHC --epochs 200 --train  --seed $seed --use_set_encoder --train_set --save_data  --set_epochs 200
# done

for seed in 4; do
    python new_main_tr.py -MHo --epochs 200 --train  --seed $seed --use_set_encoder --train_set --save_data --set_epochs 200
done