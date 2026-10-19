#!/bin/bash
# Queue of JRDB test runs for the GRPO-on-JRDB-val checkpoints not tested yet, one after another, each on GPUs 0-3
# with scripts/run_groups_jrdb_fast4gpu.sh. Logs: sft_output/logs/jrdb_rlval_queue/<name>/
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/jrdb_rlval_queue
# checkpoint | served name | prompt | type | log name
JOBS=(
  "$S/qwen2.5-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1|single|llm_single_plus1ep"
  "$S/qwen2.5-7b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1|full|llm_full_plus1ep"
  "$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLVal|p1_visual|single|vlm_single"
  "$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1_visual|single|vlm_single_plus1ep"
)
for j in "${JOBS[@]}"; do
  IFS='|' read -r ckpt name prompt type log <<< "$j"
  bash scripts/run_groups_jrdb_fast4gpu.sh "$ckpt" "$name" "$prompt" "$type" "$L/$log"
done
echo "[$(date)] all RL-on-val JRDB test runs done"
