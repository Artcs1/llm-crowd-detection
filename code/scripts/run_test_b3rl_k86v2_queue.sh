#!/bin/bash
# GPU 0 queue: wait for the K86-v2 SFT queue (PID $WAIT_PID) to exit, then the benchmark3 OOD_city / OOD_regions tests
# (scripts/test_benchmark3_qwen3vl4b.sh) of the benchmark3 SFT+RL model and the K86-v2 SFT model, one after the other.
# Usage (inside tmux): GPU=0 WAIT_PID=2945570 bash scripts/run_test_b3rl_k86v2_queue.sh
cd "$(dirname "$0")/.."
GPU=${GPU:-0}
while kill -0 ${WAIT_PID:-0} 2>/dev/null; do sleep 60; done
S=sft_output
for spec in "$S/qwen3-vl-4b-groups-grpo-merged-fourlosses-egogroups-benchmark3-1ep-nobiaskl-g8p1 Qwen/Qwen3-VL-4B-Instruct-RL-Benchmark3Train" \
            "$S/qwen3-vl-4b-groups-merged-nodspy-egogroups-k86v2 Qwen/Qwen3-VL-4B-Instruct-SFT-K86v2"; do
  set -- $spec
  [ -f "$1/config.json" ] || { echo "[$(date)] no $1, skipping"; continue; }
  echo "[$(date)] GPU $GPU: testing $2"
  GPU=$GPU PORT=$((8310 + GPU)) MODEL=$1 NAME=$2 bash scripts/test_benchmark3_qwen3vl4b.sh
done
echo "[$(date)] GPU $GPU: tests done"
