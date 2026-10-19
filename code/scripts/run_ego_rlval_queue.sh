#!/bin/bash
# EgoGroups test (EgoGroups_test, 4,500 keyframes) of the 6 GRPO-on-JRDB-val checkpoints (-rlval and -rlval-plus1ep;
# LLM single, LLM full, VLM single), in two halves running at once, each model spread over 4 GPUs with
# scripts/run_groups_fastgpus.sh. Each half starts once its half of the RL-on-val JRDB test queue is done.
# Usage: bash scripts/run_ego_rlval_queue.sh <0-3|4-7>     Logs: sft_output/logs/ego_rlval_queue/<name>/
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/ego_rlval_queue
export DATASET=ego
unset EGO_DATA
P=qwen2.5-7b-groups-grpo-merged-fourlosses
V=qwen2.5-vl-7b-groups-grpo-merged-fourlosses
if [ "$1" = 0-3 ]; then
  export GPUS="0 1 2 3" UTILS="0.45 0.45 0.45 0.45" PORT0=8091
  WAIT="grep -q 'done Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus1ep (p1 full)' sft_output/logs/jrdb_rlval_queue_gpu03_chain.log"
  JOBS=(
    "$S/$P-full-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLVal|p1|full|llm_full"
    "$S/$P-full-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1|full|llm_full_plus1ep"
    "$S/$P-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLVal|p1|single|llm_single"
  )
else
  export GPUS="4 5 6 7" UTILS="0.45 0.45 0.45 0.3" PORT0=8095
  WAIT="grep -q 'GPU 4-7 RL-on-val queue done' sft_output/logs/jrdb_rlval_queue_gpu47_chain.log"
  JOBS=(
    "$S/$V-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLVal|p1_visual|single|vlm_single"
    "$S/$V-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1_visual|single|vlm_single_plus1ep"
    "$S/$P-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1|single|llm_single_plus1ep"
  )
fi
echo "[$(date)] GPUs $1: waiting for the JRDB half-queue to finish"
until eval "$WAIT"; do sleep 30; done
for j in "${JOBS[@]}"; do
  IFS='|' read -r ckpt name prompt type log <<< "$j"
  bash scripts/run_groups_fastgpus.sh "$ckpt" "$name" "$prompt" "$type" "$L/$log"
done
echo "[$(date)] GPUs $1: EgoGroups RL-on-val queue done"
