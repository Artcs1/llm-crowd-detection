#!/usr/bin/env bash
#
# Usage:
#   chmod +x run_evaluation.sh
#   ./run_evaluation.sh
#
# Run this inside a screen session if you want it to continue after
# disconnecting from SSH.

set -euo pipefail

MODES=(
    AF
    AN
    CA
    EU
    GE
    LA
    LE
    ME
    NE
    O
    SA
    all
    crowded
    moderate
    scattered
)

MODES=(
    all
    crowded
    moderate
    scattered    
)

for MODE in "${MODES[@]}"; do
    echo "========================================"
    echo "Running evaluation for mode: $MODE"
    echo "========================================"

    python3 evaluation.py \
        --groundtruth \
            out/gt_gold_sekai_2.txt \
            out/gt_gold_sekai_22.txt \
            out/gt_gold_sekai_42.txt \
        --dataset_path \
            ../../results/detection_files/gold_SEKAI_900_3_2/ \
            ../../results/detection_files/gold_SEKAI_900_3_22/ \
            ../../results/detection_files/gold_SEKAI_900_3_42/ \
        --output "$MODE"

    echo "Finished mode: $MODE"
    echo
done

echo "All evaluations completed successfully."
