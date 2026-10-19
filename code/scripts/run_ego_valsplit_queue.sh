#!/bin/bash
# EgoGroups test (EgoGroups_test, 4,500 keyframes) of the EgoGroups-train-valsplit checkpoints not tested yet, one after
# the other, each spread over GPUs 0-3 (capped at 45%, shared with the MINGLE training) with scripts/run_groups_fastgpus.sh.
# Logs: sft_output/logs/ego_valsplit_queue/<name>/
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/ego_valsplit_queue
export DATASET=ego GPUS="0 1 2 3" UTILS="0.45 0.45 0.45 0.45" PORT0=8091
unset EGO_DATA
bash scripts/run_groups_fastgpus.sh $S/qwen2.5-7b-groups-grpo-merged-fourlosses-egogroups-train-valsplit-1ep \
  Qwen/Qwen2.5-7B-Instruct-RL-EgoGroupTrain-ValSplit p1 single $L/llm_single_rl
bash scripts/run_groups_fastgpus.sh $S/qwen2.5-7b-groups-merged-nodspy-full-egogroups-train-valsplit \
  Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit p1 full $L/llm_full_sft
echo "[$(date)] EgoGroups valsplit queue done"
