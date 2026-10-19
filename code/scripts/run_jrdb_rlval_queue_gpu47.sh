#!/bin/bash
# GPU 4-7 half of the RL-on-val JRDB test queue (the other half runs on GPUs 0-3): the two VLM checkpoints, one after
# the other, with scripts/run_groups_jrdb_fastgpus.sh. GPU 7 also holds the MINGLE eval and an EgoGroups vLLM server,
# so its server is capped at 30%. Logs: sft_output/logs/jrdb_rlval_queue/<name>/
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/jrdb_rlval_queue
export GPUS="4 5 6 7" UTILS="0.45 0.45 0.45 0.3" PORT0=8095
bash scripts/run_groups_jrdb_fastgpus.sh $S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval \
  Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLVal p1_visual single $L/vlm_single
bash scripts/run_groups_jrdb_fastgpus.sh $S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval-plus1ep \
  Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValPlus1ep p1_visual single $L/vlm_single_plus1ep
echo "[$(date)] GPU 4-7 RL-on-val queue done"
