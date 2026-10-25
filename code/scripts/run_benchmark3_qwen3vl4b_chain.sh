#!/bin/bash
# Qwen3-VL-4B-Instruct (--base qwen3-4b) on benchmark3 (README_benchmark3.md: train = the 1533 benchmark3_train clips,
# validation = the 219 id_val clips), p1_visual single, all-linear LoRA: SFT (3 epochs) -> RL on the same train clips,
# 1 epoch (use_bias_correction_kl off, 8 rollouts x 1 prompt, as run_jrdb_qwen3vl4b_chain.sh).
# Checkpoints: sft_output/qwen3-vl-4b-groups-merged-nodspy-egogroups-benchmark3
#              sft_output/qwen3-vl-4b-groups-grpo-merged-fourlosses-egogroups-benchmark3-1ep-nobiaskl-g8p1
# Logs: sft_output/logs/benchmark3_qwen3vl4b/{sft,rl_train}.log
# Usage (inside tmux): GPU=1 bash scripts/run_benchmark3_qwen3vl4b_chain.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/benchmark3_qwen3vl4b
mkdir -p "$LOGS"
BASE="--base qwen3-4b --dataset egogroups-benchmark3 --variant p1_visual --mode single"
RLARGS="--no_bias_kl --num_generations 8 --prompts_per_step 1"
SFT=$S/qwen3-vl-4b-groups-merged-nodspy-egogroups-benchmark3

echo "[$(date)] GPU $GPU: Qwen3-VL-4B benchmark3 SFT"
# shellcheck disable=SC2086
MASTER_PORT=29651 python sft_training_vlm_nodspy_merged.py $BASE --gpu "$GPU" > "$LOGS/sft.log" 2>&1
echo "[$(date)] GPU $GPU: SFT exit code $?"
if [ ! -f "$SFT/config.json" ]; then echo "[$(date)] GPU $GPU: no $SFT, skipping RL"; exit 1; fi
echo "[$(date)] GPU $GPU: RL on the benchmark3 train clips, 1 epoch"
# shellcheck disable=SC2086
MASTER_PORT=29652 python rl_training_vlm_nodspy_merged.py $BASE $RLARGS --epochs 1 --gpu "$GPU" > "$LOGS/rl_train.log" 2>&1
echo "[$(date)] GPU $GPU: RL exit code $?"
find "$S" -maxdepth 1 -name "qwen3-vl-4b-*benchmark3*" -exec chmod -R a+rwX {} + 2>/dev/null
echo "[$(date)] GPU $GPU: Qwen3-VL-4B benchmark3 chain done"
