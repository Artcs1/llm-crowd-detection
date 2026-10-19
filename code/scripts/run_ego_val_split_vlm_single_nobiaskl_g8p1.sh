#!/bin/bash
# EgoGroups_train with the 180 shared val clips held out (--ego_val_split): VLM p1_visual single SFT (3 epochs, script
# default), then 1-epoch GRPO with use_bias_correction_kl off and 8 rollouts x 1 prompt per step (as the LLM RL).
# Replaces the VLM part of scripts/run_ego_val_split_sft_rl_vlm.sh (VLM full not queued for now).
# Checkpoints: sft_output/qwen2.5-vl-7b-groups-merged-nodspy-egogroups-train-valsplit,
#              sft_output/qwen2.5-vl-7b-groups-grpo-*-fourlosses-egogroups-train-valsplit-1ep-nobiaskl-g8p1
# Logs: sft_output/logs/ego_val_split/{sft_vlm_single,rl_vlm_single_nobiaskl_g8p1}.log
# Usage (inside tmux): GPU=0 bash scripts/run_ego_val_split_vlm_single_nobiaskl_g8p1.sh

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-0}
LOGS=sft_output/logs/ego_val_split
mkdir -p "$LOGS"

echo "[$(date)] GPU $GPU: SFT vlm_single"
if ! MASTER_PORT=29543 python sft_training_vlm_nodspy_merged.py --dataset egogroups-train --variant p1_visual --mode single \
     --ego_val_split --gpu "$GPU" > "$LOGS/sft_vlm_single.log" 2>&1; then
  echo "[$(date)] GPU $GPU: SFT vlm_single FAILED, skipping RL (see $LOGS/sft_vlm_single.log)"; exit 1
fi
echo "[$(date)] GPU $GPU: SFT vlm_single done, RL vlm_single (no bias KL, 8x1)"
MASTER_PORT=29553 python rl_training_vlm_nodspy_merged.py --dataset egogroups-train --variant p1_visual --mode single \
  --epochs 1 --ego_val_split --no_bias_kl --num_generations 8 --prompts_per_step 1 --gpu "$GPU" \
  > "$LOGS/rl_vlm_single_nobiaskl_g8p1.log" 2>&1
echo "[$(date)] GPU $GPU: RL vlm_single finished with exit code $?"
