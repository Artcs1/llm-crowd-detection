#!/bin/bash
# VLM part of scripts/run_ego_val_split_sft_rl.sh (requeued after the JRDB VLM g4p2 run on GPU 0). EgoGroups_train with the 180 shared val clips held out (--ego_val_split: train on 1620 clips x 5 sub-clips, validate on
# 180 x 5), SFT (3 epochs, script default) then 1-epoch GRPO, for LLM single, LLM full, VLM p1_visual single and VLM
# p1_visual full (video), all in sequence on one GPU. Waits first for the JRDB val-split RL chain (VLM, LLM full on GPU 0,
# then scripts/run_jrdb_val_split_rl_single_gpu0.sh) to finish. Each RL run starts only after its own SFT succeeded.
# Checkpoints: sft_output/*-egogroups-train-valsplit* (never the paper's -egogroups-train dirs).
# Logs: sft_output/logs/ego_val_split/{sft,rl}_<name>.log
# Usage (inside tmux): GPU=0 bash scripts/run_ego_val_split_sft_rl_vlm.sh

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-0}
LOGS=sft_output/logs/ego_val_split
mkdir -p "$LOGS"


pipeline() {  # <name> <sft port> <rl port> <sft script> <rl script> <mode/variant args...>
  local name=$1 sft_port=$2 rl_port=$3 sft=$4 rl=$5; shift 5
  echo "[$(date)] GPU $GPU: SFT $name"
  if ! MASTER_PORT=$sft_port python "$sft" --dataset egogroups-train "$@" --ego_val_split --gpu "$GPU" > "$LOGS/sft_$name.log" 2>&1; then
    echo "[$(date)] GPU $GPU: SFT $name FAILED, skipping RL (see $LOGS/sft_$name.log)"; return 1
  fi
  echo "[$(date)] GPU $GPU: SFT $name done, RL $name"
  MASTER_PORT=$rl_port python "$rl" --dataset egogroups-train "$@" --epochs 1 --ego_val_split --gpu "$GPU" > "$LOGS/rl_$name.log" 2>&1
  echo "[$(date)] GPU $GPU: RL $name finished with exit code $?"
}

pipeline vlm_single 29543 29553 sft_training_vlm_nodspy_merged.py rl_training_vlm_nodspy_merged.py --variant p1_visual --mode single
pipeline vlm_full   29544 29554 sft_training_vlm_nodspy_merged.py rl_training_vlm_nodspy_merged.py --variant p1_visual --mode full
echo "[$(date)] all EgoGroups val-split SFT/RL runs done"
