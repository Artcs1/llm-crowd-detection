#!/bin/bash
# g8p4 job of scripts/run_jrdb_rl_on_val_vlm_nobiaskl.sh (8 rollouts x 4 prompts per step, 3 epochs, no bias KL, RL on the
# 7 JRDB val scenes), moved from GPU 1 (stopped there at step 6/291 on Oct 19 01:2x, no checkpoint) to the GPU 0 queue.
# Usage: GPU=0 bash scripts/run_jrdb_rl_on_val_vlm_nobiaskl_g8p4_gpu0.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
GPU=${GPU:-0}
S=sft_output
LOGS=$S/logs/jrdb_rl_on_val_vlm_nobiaskl
echo "[$(date)] GPU $GPU: VLM RL-on-val 3 epochs, no bias KL, g8p4"
if MASTER_PORT=29594 python rl_training_vlm_nodspy_merged.py --dataset jrdb --variant p1_visual --epochs 3 \
     --jrdb_val_split --rl_on_val --no_bias_kl --num_generations 8 --prompts_per_step 4 --save-every-epoch --gpu "$GPU" \
     > "$LOGS/rl_vlm_g8p4.log" 2>&1; then
  find "$S"/qwen2.5-vl-7b-groups-grpo-*-jrdbtrain-rlval-nobiaskl* -type f ! -perm -o+r -exec chmod a+rw {} + 2>/dev/null
  echo "[$(date)] GPU $GPU: g8p4 done"
else
  echo "[$(date)] GPU $GPU: g8p4 FAILED (see $LOGS/rl_vlm_g8p4.log)"
fi
