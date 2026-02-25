#!/bin/bash

NUM_PARTS=50
MAX_JOBS=8

for ((i=1; i<=NUM_PARTS; i++)); do
    echo "Launching part_id=$i"
    python3 gemini_inference_coarse.py ../../eccv/ gemini/gemini \
        --num_parts $NUM_PARTS \
        --part_id $i &

    # Limit number of parallel jobs
    if (( $(jobs -r | wc -l) >= MAX_JOBS )); then
        wait -n
    fi
done

wait
echo "All jobs finished."

