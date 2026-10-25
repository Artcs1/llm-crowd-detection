#!/bin/bash
# The 6 single-frame Qwen3-VL-4B JRDB checkpoints on EgoGroupsCity_Test_jrdbstyle (JRDB-frame 3D), one after the other
# on GPU 1, after the full-mode SFT test in tmux session egocity_gpu1 ends. Same names as their JRDB test runs.
# Run inside tmux: tmux new -d -s egocity_gpu1_single 'bash scripts/queue_egocity_jrdbstyle_qwen3vl4b_single.sh'
cd "$(dirname "$0")/.."
S=sft_output
LG=sft_output/logs
JOBS=(
  "qwen3-vl-4b-groups-merged-nodspy-jrdbtrain|SFT-JRDBTrain"
  "qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-nobiaskl-g8p1|RL-JRDBTrain"
  "qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain-rlval-nobiaskl-g8p1|RL-JRDBTrain-RLVal3ep"
  "qwen3-vl-4b-groups-merged-nodspy-jrdbtrain-lmlora|SFT-JRDBTrain-LMLoRA"
  "qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain-lmlora-1ep-nobiaskl-g8p1|RL-JRDBTrain-LMLoRA"
  "qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain-lmlora-rlval-nobiaskl-g8p1|RL-JRDBTrain-RLVal3ep-LMLoRA"
)
echo "[$(date)] waiting for tmux session egocity_gpu1"
while tmux has-session -t egocity_gpu1 2>/dev/null; do sleep 60; done
port=8051
for j in "${JOBS[@]}"; do
  IFS='|' read -r ckpt tag <<< "$j"
  echo "[$(date)] $tag ($ckpt) on port $port"
  GPU_UTIL=0.85 DATA=EgoGroupsCity_Test_jrdbstyle NAME=Qwen/Qwen3-VL-4B-Instruct-$tag \
    bash scripts/run_zeroshot_egogroupscity.sh 1 $S/$ckpt p1_visual $port 48 single \
    > $LG/egocity_jrdbstyle_qwen3vl4b_single_${tag}_driver.log 2>&1
  tail -3 $LG/egocity_jrdbstyle_qwen3vl4b_single_${tag}_driver.log
  port=$((port + 1))
done
echo "[$(date)] all 6 done"
