#!/bin/bash
echo "Start training with Accelerate"

for d in "eth" "hotel" "univ" "zara1" "zara2"; 
do
    echo starting $d;
    accelerate launch trainval.py \
        --cfg ./config/config-pixel-g2p.json \
        --dataset $d \
        --tag LLM-$d-pixel
done
