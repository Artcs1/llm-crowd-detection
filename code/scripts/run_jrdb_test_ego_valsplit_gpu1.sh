#!/bin/bash
# JRDB test (1,829 keyframes) of the EgoGroups-trained LLMs with the 180 val clips held out (-egogroups-train-valsplit),
# in sequence on one GPU: LLM single RL (1 epoch) and LLM full SFT. Served names follow the other workstation's
# convention (Qwen/Qwen2.5-7B-Instruct-{SFT,RL}-EgoGroupTrain-ValSplit) so results land in the same folders.
# Logs: sft_output/logs/jrdb_test_ego_valsplit/
# Usage (inside tmux): GPU=1 bash scripts/run_jrdb_test_ego_valsplit_gpu1.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_test_ego_valsplit
mkdir -p "$LOGS"
# name | vllm port | merged checkpoint | served name | prompt | type
JOBS=(
  "rl_single|8095|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-egogroups-train-valsplit-1ep|Qwen/Qwen2.5-7B-Instruct-RL-EgoGroupTrain-ValSplit|p1|single"
  "sft_full|8096|$S/qwen2.5-7b-groups-merged-nodspy-full-egogroups-train-valsplit|Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit|p1|full"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r name vport ckpt served prompt type <<< "$job"
  if [ ! -f "$ckpt/config.json" ]; then echo "[$(date)] GPU $GPU: no $ckpt, skipping $name"; continue; fi
  echo "[$(date)] GPU $GPU: serving $ckpt on $vport as $served"
  CUDA_VISIBLE_DEVICES=$GPU vllm serve "$ckpt" --served-model-name "$served" --api-key jrdb --port "$vport" > "$LOGS/${name}_server.log" 2>&1 &
  pid=$!
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$vport/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] GPU $GPU: server $vport died, see $LOGS/${name}_server.log"; continue 2; fi
    sleep 15
  done
  echo "[$(date)] GPU $GPU: server ready, JRDB test $name ($served $prompt $type)"
  bash scripts/run_groups_jrdb.bash "$served" "$vport" jrdb "$prompt" "$type" > "$LOGS/${name}_test.log" 2>&1
  echo "[$(date)] GPU $GPU: JRDB test $name finished with exit code $?"
  kill $pid; wait $pid 2>/dev/null
done
echo "[$(date)] all JRDB tests done"
