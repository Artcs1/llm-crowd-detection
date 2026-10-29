#!/bin/bash
# One GPU, in order: Qwen3-VL-4B SFT on K60 and K86 (scripts/run_ksubsets_qwen3vl4b_sft_queue.sh, no per-epoch eval), then
# the benchmark3 RL restarted from scratch (same settings as scripts/run_benchmark3_qwen3vl4b_chain.sh; the first run was
# stopped at step 3253/7585 on 2026-10-26 with no checkpoint). RL log: sft_output/logs/benchmark3_qwen3vl4b/rl_train_restart1.log
# Usage (inside tmux): GPU=0 bash scripts/run_k60_k86_then_b3rl.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
GPU=${GPU:-0}
S=sft_output
GPU=$GPU KS="k60 k86" bash scripts/run_ksubsets_qwen3vl4b_sft_queue.sh
SFT=$S/qwen3-vl-4b-groups-merged-nodspy-egogroups-benchmark3
[ -f "$SFT/config.json" ] || { echo "[$(date)] no $SFT, skipping RL"; exit 1; }
echo "[$(date)] GPU $GPU: benchmark3 RL (restart), 1 epoch"
MASTER_PORT=29672 python rl_training_vlm_nodspy_merged.py --base qwen3-4b --dataset egogroups-benchmark3 --variant p1_visual \
  --mode single --no_bias_kl --num_generations 8 --prompts_per_step 1 --epochs 1 --gpu "$GPU" \
  > $S/logs/benchmark3_qwen3vl4b/rl_train_restart1.log 2>&1
rc=$?
echo "[$(date)] GPU $GPU: RL exit code $rc"
find "$S" -maxdepth 1 -name "qwen3-vl-4b-*benchmark3*" -exec chmod -R a+rX {} + 2>/dev/null
echo "[$(date)] GPU $GPU: done"
