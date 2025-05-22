#!/bin/bash

for d in "eth" "hotel" "univ" "zara1" "zara2"; 
do
    echo starting $d;
    python main.py \
        --model_name 'VisSem' \
        --phase 'test' \
        --dataset eth5 \
        --test_set $d \
        --load_checkpoint best \
        --sampler_temperature 1.2 \
        --prompt_type arrow \
        --prompt_color red \
        --num_test_samples 20 \
        --batch_size 64 \
        --shuffle_test_batches False \
        --num_test_runs 1;
done
