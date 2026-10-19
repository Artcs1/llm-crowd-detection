#!/bin/bash
# MINGLE pairwise SFT with 1:NEG Yes:No training pairs (--neg_per_pos NEG, default 2) and a balanced 1:1
# validation set (1500 pairs from the shared 60/60/60 EgoGroups validation clips), 5 epochs, one base model per
# invocation, both at lr 5e-5 + 3% warmup (lr 1e-4 diverged for Qwen2.5-VL 1:1 and Qwen2-VL 1:2):
#   bash scripts/run_mingle_neg2.sh qwen2-vl   0,1,2,3 29512 1.6    (gives 1:1.55 train = the paper's Table 1 split, 28,341/43,654)
#   bash scripts/run_mingle_neg2.sh qwen2.5-vl 4,5,6,7 29513 1.6
# Two invocations can run in parallel on different GPUs as long as their ports differ.
# Waits for the --neg_per_pos NEG records cache (built separately with --build_only) before training.
# Log: sft_output/logs/mingle_neg<NEG>_<base>.log

umask 000  # files on the share stay writable from the other workstation
set -o pipefail
export PYTHONUNBUFFERED=1

BASE=${1:?usage: run_mingle_neg2.sh <qwen2-vl|qwen2.5-vl> <gpus> <port>}
GPUS=${2:?gpus, e.g. 0,1,2,3}
PORT=${3:?port, e.g. 29512}
NEG=${4:-2}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
case "$BASE" in
    qwen2-vl|qwen2.5-vl) EXTRA=(--learning_rate 5e-5 --warmup_ratio 0.03) ;;
    *) echo "unknown base model: $BASE"; exit 1 ;;
esac

source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
mkdir -p sft_output/logs

CACHE=sft_output/mingle_records_cache/neg${NEG}_limitNone.json
echo "[$(date)] $BASE: waiting for $CACHE"
until [ -f "$CACHE" ]; do sleep 300; done

echo "[$(date)] training $BASE on GPUs $GPUS (port $PORT): ${EXTRA[*]}"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --multi_gpu --num_processes="$NUM_GPUS" \
    --main_process_port "$PORT" mingle_sft_training.py --base_model "$BASE" \
    --neg_per_pos "$NEG" --val_pairs 1500 --epochs 5 "${EXTRA[@]}" \
    2>&1 | tee "sft_output/logs/mingle_neg${NEG}_${BASE}.log"
echo "[$(date)] $BASE finished with exit code $?"
