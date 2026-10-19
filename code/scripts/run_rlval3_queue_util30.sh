#!/bin/bash
# JRDB test (native JRDB_fixed_gold) and EgoGroups test (EgoGroups_test_jrdbstyle, the JRDB-style 3D cross-check)
# of the JRDB-trained GRPO-on-val checkpoints: 3rd epoch (-rlval-plus2ep: LLM single, LLM video, VLM) and VLM with the
# bias-correction KL off (-rlval-nobiaskl-epoch1/2/3). Two halves at once, each model over 4 GPUs with
# scripts/run_groups_fastgpus_v2.sh, vLLM capped at 45% (the GPUs also run MINGLE tests).
# Copy of run_rlval3_queue.sh with half A capped at 30% (so the JRDB MINGLE training also fits on GPUs 0-3).
# Usage: bash scripts/run_rlval3_queue_util30.sh <A|B>   A = GPUs 0-3, B = GPUs 4-7. Logs: sft_output/logs/rlval3_queue/
# The JRDB MINGLE training (scripts/run_mingle_jrdb.sh) holds while sft_output/logs/HOLD_mingle_jrdb_training exists;
# the second half to finish removes it.
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/rlval3_queue
P=$S/qwen2.5-7b-groups-grpo-merged-fourlosses
V=$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses
if [ "$1" = A ]; then
  export GPUS="0 1 2 3" UTILS="0.3 0.3 0.3 0.3" PORT0=8091
  M=(
    "$P-jrdbtrain-1ep-rlval-plus2ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus2ep|p1|single|llm_single_plus2ep"
    "$P-full-jrdbtrain-1ep-rlval-plus2ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus2ep|p1|full|llm_full_plus2ep"
    "$V-jrdbtrain-1ep-rlval-plus2ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValPlus2ep|p1_visual|single|vlm_single_plus2ep"
  )
else
  export GPUS="4 5 6 7" UTILS="0.45 0.45 0.45 0.45" PORT0=8095
  M=(
    "$V-jrdbtrain-rlval-nobiaskl-epoch1|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValNoBiasKL-ep1|p1_visual|single|vlm_nobiaskl_ep1"
    "$V-jrdbtrain-rlval-nobiaskl-epoch2|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValNoBiasKL-ep2|p1_visual|single|vlm_nobiaskl_ep2"
    "$V-jrdbtrain-rlval-nobiaskl-epoch3|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValNoBiasKL-ep3|p1_visual|single|vlm_nobiaskl_ep3"
  )
fi
for m in "${M[@]}"; do
  IFS='|' read -r ckpt name prompt type log <<< "$m"
  for ds in jrdb ego_jrdbstyle; do
    DATASET=$ds bash scripts/run_groups_fastgpus_v2.sh "$ckpt" "$name" "$prompt" "$type" "$L/${ds}_$log"
  done
done
touch "$L/queue_$1.done"
if [ -e "$L/queue_A.done" ] && [ -e "$L/queue_B.done" ]; then rm -f sft_output/logs/HOLD_mingle_jrdb_training; fi
echo "[$(date)] rlval3 queue $1 done"
