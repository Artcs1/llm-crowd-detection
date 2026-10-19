#!/bin/bash
# JRDB test with EgoGroups-style 3D (JRDB_fixed_gold_egostyle, 1,829 keyframes) of the three EgoGroups-train-valsplit
# checkpoints, one after the other, each over GPUs 0-3 (capped at 45%, shared with the MINGLE training) with
# scripts/run_groups_fastgpus_v2.sh. Starts once the EgoGroups valsplit queue (run_ego_valsplit_queue.sh) is done.
# Logs: sft_output/logs/jrdb_egostyle_valsplit_queue/<name>/
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/jrdb_egostyle_valsplit_queue
export DATASET=jrdb_egostyle GPUS="0 1 2 3" UTILS="0.45 0.45 0.45 0.45" PORT0=8091
echo "[$(date)] waiting for the EgoGroups valsplit queue"
until grep -q "EgoGroups valsplit queue done" sft_output/logs/ego_valsplit_queue_chain.log 2>/dev/null; do sleep 30; done
bash scripts/run_groups_fastgpus_v2.sh $S/qwen2.5-7b-groups-merged-nodspy-egogroups-train-valsplit \
  Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit p1 single $L/llm_single_sft
bash scripts/run_groups_fastgpus_v2.sh $S/qwen2.5-7b-groups-grpo-merged-fourlosses-egogroups-train-valsplit-1ep \
  Qwen/Qwen2.5-7B-Instruct-RL-EgoGroupTrain-ValSplit p1 single $L/llm_single_rl
bash scripts/run_groups_fastgpus_v2.sh $S/qwen2.5-7b-groups-merged-nodspy-full-egogroups-train-valsplit \
  Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit p1 full $L/llm_full_sft
echo "[$(date)] JRDB egostyle valsplit queue done"
