#!/bin/bash

for d in "eth" "hotel" "univ" "zara1" "zara2"; 
do
    echo starting $d;
    python main.py \
        --model_name 'VisSem' \
        --phase 'train_test' \
        --dataset eth5 \
        --test_set $d \
        --num_epochs 50 \
        --validate_every 1 \
        --shuffle_test_batches False \
        --num_test_runs 5 \
        --down_factor 4 \
        --batch_size 64 \
        --sampler_temperature 1.2 \
        --prompt_type arrow \
        --prompt_color red \
        --scheduler ExponentialLR;
done

