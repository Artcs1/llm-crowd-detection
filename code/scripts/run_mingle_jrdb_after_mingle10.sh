#!/bin/bash
# Builds the JRDB MINGLE records cache now (CPU only), then waits for the MINGLE-1.0 JRDB test on GPUs 0-3
# (../MINGLE/run_mingle10_jrdb_eval.sh) to finish before training Qwen2-VL on GPUs 0-3 with scripts/run_mingle_jrdb.sh.
umask 000
cd "$(dirname "$0")/.."
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
export PYTHONUNBUFFERED=1
echo "[$(date)] building the JRDB records cache (CPU only)"
python mingle_sft_training.py --dataset jrdb --base_model qwen2-vl --neg_per_pos 1.92 --build_only \
  > sft_output/logs/mingle_jrdb_neg1.92_qwen2-vl.log.build 2>&1 || { echo "cache build failed"; exit 1; }
grep -E "pairwise examples|records cached" sft_output/logs/mingle_jrdb_neg1.92_qwen2-vl.log.build
echo "[$(date)] cache ready; waiting for the MINGLE-1.0 JRDB test to finish"
until grep -q "\] done" ../../MINGLE/run_mingle10_jrdb_eval.log 2>/dev/null; do sleep 120; done
sleep 30
bash scripts/run_mingle_jrdb.sh qwen2-vl 0,1,2,3 29514 1.92
