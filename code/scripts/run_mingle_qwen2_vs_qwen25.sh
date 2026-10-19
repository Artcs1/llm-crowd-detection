#!/bin/bash
# Builds the MINGLE pairwise records cache (CPU only), then trains the MINGLE-style pairwise judge on
# Qwen2-VL-7B and then Qwen2.5-VL-7B, each on 4 GPUs with a 500-pair validation set.
# Logs: sft_output/logs/mingle_{build,qwen2-vl,qwen2.5-vl}.log
# Run inside tmux, e.g.: tmux new -d -s mingle 'bash scripts/run_mingle_qwen2_vs_qwen25.sh'

umask 000  # files on the share stay writable from the other workstation
set -o pipefail

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
mkdir -p sft_output/logs

GPUS=${GPUS:-4,5,6,7}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)

echo "[$(date)] building records cache"
if ! CUDA_VISIBLE_DEVICES= python mingle_sft_training.py --build_only 2>&1 | tee sft_output/logs/mingle_build.log; then
    echo "[$(date)] records build failed, not training"
    exit 1
fi

for base in qwen2-vl qwen2.5-vl; do
    echo "[$(date)] training $base on GPUs $GPUS"
    CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --multi_gpu --num_processes="$NUM_GPUS" \
        --main_process_port 29511 mingle_sft_training.py --base_model "$base" --val_pairs 1500 \
        2>&1 | tee "sft_output/logs/mingle_${base}.log"
    echo "[$(date)] $base finished with exit code $?"
done
echo "[$(date)] all done"
