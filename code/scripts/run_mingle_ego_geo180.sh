#!/bin/bash
# MINGLE pairwise SFT on EgoGroups_train with the geo180 val split (EGO_VAL_SPLIT=geo180: 6 clips per (region, density),
# code/egogroups_val_geo180.txt; train on the other 1620 clips). Same settings as the earlier EgoGroups Qwen2-VL 1:1.55
# run (run_mingle_neg2.sh qwen2-vl ... 1.6): --neg_per_pos 1.6, 1500 balanced val pairs, 5 epochs, lr 5e-5 + 3% warmup.
# Reuses the records cache neg1.6_limitNone.json (it holds every clip; the val split is applied after loading).
# Checkpoints: sft_output/qwen2-vl-7b-mingle-pairwise-merged-egogroups-train-neg1.6-lr5e-05-geo180-epochN
# Waits for the EgoGroups test of the JRDB-trained epoch 5 (MINGLE/run_q2vl_jrdbtrain_ego_cap_eval.sh) on GPUs 0-3.
#   bash scripts/run_mingle_ego_geo180.sh [gpus=0,1,2,3] [port=29515]
umask 000
set -o pipefail
export PYTHONUNBUFFERED=1
export EGO_VAL_SPLIT=geo180
GPUS=${1:-0,1,2,3}
PORT=${2:-29515}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
LOG=sft_output/logs/mingle_neg1.6_qwen2-vl_geo180.log
echo "[$(date)] waiting for the EgoGroups test of the JRDB-trained epoch 5"
until grep -q "\] done" ../../MINGLE/run_q2vl_jrdbtrain_ego_cap_eval.log 2>/dev/null; do sleep 120; done
sleep 60
echo "[$(date)] training qwen2-vl (geo180 val) on GPUs $GPUS (port $PORT)"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --multi_gpu --num_processes="$NUM_GPUS" \
    --main_process_port "$PORT" mingle_sft_training.py --base_model qwen2-vl \
    --neg_per_pos 1.6 --val_pairs 1500 --epochs 5 --learning_rate 5e-5 --warmup_ratio 0.03 \
    2>&1 | tee "$LOG"
echo "[$(date)] qwen2-vl geo180 finished with exit code $?"
