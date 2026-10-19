#!/bin/bash
# JRDB VLM p1_visual single SFT on the 20 train scenes (--jrdb_val_split: the 7 val scenes held out), Qwen2.5-VL-7B with
# LoRA on the language model only (--lm_lora: vision tower + merger frozen). Compare on JRDB test with the all-linear
# SFT qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain (MINGLE F1 38.43).
# Checkpoint: sft_output/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain-lmlora. Log: sft_output/logs/jrdb_lmlora/sft_vlm_single.log
# Usage (inside tmux): GPU=0 bash scripts/run_jrdb_vlm_sft_lmlora.sh
umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
GPU=${GPU:-0}
LOGS=sft_output/logs/jrdb_lmlora
mkdir -p "$LOGS"
echo "[$(date)] GPU $GPU: JRDB VLM SFT, LM-only LoRA"
MASTER_PORT=29601 python sft_training_vlm_nodspy_merged.py --dataset jrdb --variant p1_visual --mode single --jrdb_val_split \
  --lm_lora --gpu "$GPU" > "$LOGS/sft_vlm_single.log" 2>&1
echo "[$(date)] GPU $GPU: SFT finished with exit code $?"
