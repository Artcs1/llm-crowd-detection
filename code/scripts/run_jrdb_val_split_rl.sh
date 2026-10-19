#!/bin/bash
# 1-epoch GRPO on JRDB with the 7 JRDB val scenes held out (--jrdb_val_split), on top of the -jrdbtrain SFT checkpoints.
# GPU 0 queue (after the VLM SFT, pid $WAIT_PID, exits), in sequence: VLM p1_visual, LLM full (video), LLM single.
# LLM single instead starts right away on GPU 1 if GPU 1 is free (no compute process, < 1 GB used) at launch.
# Each run checks that its SFT merged checkpoint exists. Outputs: sft_output/qwen2.5{,-vl}-7b-groups-grpo-*-fourlosses{,-full}-jrdbtrain-1ep
# Logs: sft_output/logs/jrdb_val_split_rl/rl_<name>.log
# Usage (inside tmux): WAIT_PID=2459418 bash scripts/run_jrdb_val_split_rl.sh

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
LOGS=sft_output/logs/jrdb_val_split_rl
mkdir -p "$LOGS"
S=sft_output

rl() {  # <name> <gpu> <port> <sft merged dir> <rl script> <mode/variant args...>
  local name=$1 gpu=$2 port=$3 sft=$4 script=$5; shift 5
  if [ ! -f "$sft/config.json" ]; then echo "[$(date)] GPU $gpu: no SFT checkpoint $sft, skipping RL $name"; return 1; fi
  echo "[$(date)] GPU $gpu: RL $name (from $sft)"
  MASTER_PORT=$port python "$script" --dataset jrdb "$@" --epochs 1 --jrdb_val_split --gpu "$gpu" > "$LOGS/rl_$name.log" 2>&1
  echo "[$(date)] GPU $gpu: RL $name finished with exit code $?"
}
llm_single() { rl llm_single "$1" 29531 $S/qwen2.5-7b-groups-merged-nodspy-jrdbtrain rl_training_llm_nodspy_merged.py --mode single; }

gpu1_free() {
  [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ] &&
  [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]
}
if gpu1_free; then SINGLE_ON_GPU1=1; llm_single 1 & else echo "[$(date)] GPU 1 busy, LLM single goes in the GPU 0 queue"; fi

(
  if [ -n "$WAIT_PID" ]; then
    echo "[$(date)] GPU 0: waiting for the VLM SFT (pid $WAIT_PID) to finish"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
    echo "[$(date)] GPU 0: pid $WAIT_PID exited"
  fi
  rl vlm_p1_visual 0 29533 $S/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain rl_training_vlm_nodspy_merged.py --variant p1_visual
  rl llm_full 0 29532 $S/qwen2.5-7b-groups-merged-nodspy-full-jrdbtrain rl_training_llm_nodspy_merged.py --mode full
  [ -z "$SINGLE_ON_GPU1" ] && llm_single 0
) &
wait
echo "[$(date)] all JRDB val-split RL runs done"
