#!/bin/bash
# The rest of rlval3 queue A (JRDB-trained 3-epoch RL-on-val checkpoints, -rlval-plus2ep) run in parallel on GPUs 0-3:
# one job per GPU (one vLLM server capped at 30%, 16 clients), next to the job already running over all four GPUs
# (LLM single on JRDB test) and the MINGLE-1.0 JRDB test. Lane GPU 1 runs a second job after its first.
# Logs: sft_output/logs/rlval3_queue/<dataset>_<name>/ ; done marker: "rlval3 parallel A done"
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/rlval3_queue
P=$S/qwen2.5-7b-groups-grpo-merged-fourlosses
V=$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses
LS="$P-jrdbtrain-1ep-rlval-plus2ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus2ep|p1|single|llm_single_plus2ep"
LF="$P-full-jrdbtrain-1ep-rlval-plus2ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus2ep|p1|full|llm_full_plus2ep"
VS="$V-jrdbtrain-1ep-rlval-plus2ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValPlus2ep|p1_visual|single|vlm_single_plus2ep"
run() {  # <gpu> <dataset> <model spec>
  local gpu=$1 ds=$2 ckpt name prompt type log
  IFS='|' read -r ckpt name prompt type log <<< "$3"
  GPUS="$gpu" UTILS="0.3" PORT0=$((8101 + gpu)) PER=16 DATASET=$ds \
    bash scripts/run_groups_fastgpus_v3.sh "$ckpt" "$name" "$prompt" "$type" "$L/${ds}_$log"
}
run 0 ego_jrdbstyle "$LS" &
( run 1 jrdb "$LF"; run 1 ego_jrdbstyle "$VS" ) &
run 2 ego_jrdbstyle "$LF" &
run 3 jrdb "$VS" &
wait
echo "[$(date)] rlval3 parallel A done"
