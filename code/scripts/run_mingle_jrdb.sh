#!/bin/bash
# MINGLE pairwise SFT on JRDB train (mingle_sft_training.py --dataset jrdb): 20 JRDB train scenes for training,
# 1500 balanced validation pairs from the 7 JRDB val scenes, 5 epochs, lr 5e-5 + 3% warmup (same as the EgoGroups
# 1:1.55 runs). NEG=1.92 gives ~1:1.55 Yes:No on the train scenes (33,535 Yes / ~52k No).
#   bash scripts/run_mingle_jrdb.sh <qwen2-vl|qwen2.5-vl> <gpus> <port> [NEG=1.92]
# Waits first for the EVP depth of all 1705 JRDB train target frames (EVP/depth/gold_jrdb_train_depth_res, made by
# EVP/depth/run_depth_jrdb_train_8gpu.sh on the workstation with the EVP env), then builds the records cache once
# (--build_only, CPU only) and trains. Log: sft_output/logs/mingle_jrdb_neg<NEG>_<base>.log
umask 000
set -o pipefail
export PYTHONUNBUFFERED=1
BASE=${1:?usage: run_mingle_jrdb.sh <qwen2-vl|qwen2.5-vl> <gpus> <port> [NEG]}
GPUS=${2:?gpus, e.g. 0,1,2,3}
PORT=${3:?port, e.g. 29514}
NEG=${4:-1.92}
NUM_GPUS=$(echo "$GPUS" | tr ',' '\n' | wc -l)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
mkdir -p sft_output/logs
LOG=sft_output/logs/mingle_jrdb_neg${NEG}_${BASE}.log
D=../../EVP/depth/gold_jrdb_train_depth_res

echo "[$(date)] waiting for EVP depth of the 1705 JRDB train frames in $D"
until [ "$(ls $D/*/00015.png 2>/dev/null | wc -l)" -ge 1705 ]; do sleep 120; done
sleep 60  # last PNGs fully written
HOLD=sft_output/logs/HOLD_mingle_jrdb_training  # exists while vLLM tests share GPUs 0-3 (training would run out of memory)
while [ -e "$HOLD" ]; do echo "[$(date)] waiting: $HOLD exists"; sleep 300; done
echo "[$(date)] depth ready; building the records cache"
python mingle_sft_training.py --dataset jrdb --base_model "$BASE" --neg_per_pos "$NEG" --build_only 2>&1 | tee "$LOG.build" || exit 1

echo "[$(date)] training $BASE on GPUs $GPUS (port $PORT)"
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch --multi_gpu --num_processes="$NUM_GPUS" \
    --main_process_port "$PORT" mingle_sft_training.py --dataset jrdb --base_model "$BASE" \
    --neg_per_pos "$NEG" --val_pairs 1500 --epochs 5 --learning_rate 5e-5 --warmup_ratio 0.03 \
    2>&1 | tee "$LOG"
echo "[$(date)] $BASE finished with exit code $?"
