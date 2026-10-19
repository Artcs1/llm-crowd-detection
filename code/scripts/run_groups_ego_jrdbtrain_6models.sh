#!/bin/bash
# JRDB-trained -> EgoGroups_test (900 clips x 5 keyframes = 4,500) group-detection eval of the 6 JRDB-trained SFT/RL
# checkpoints (trained on all 27 JRDB scenes; the ones scored on JRDB test as "-JRDBTrain"), one vLLM server per GPU:
# start the server, wait until it answers, run scripts/run_groups.bash on
#   1. EgoGroups_test            (detany_3D in the EgoGroups camera frame, the input of every other EgoGroups row)
#   2. EgoGroups_test_jrdbstyle  (same people, 3D rewritten in the JRDB robot frame the models were trained on;
#                                 scripts/make_egogroups_jrdbstyle.py)
# then stop the server. Predictions: results/predictions/{EgoGroups_test,EgoGroups_test_jrdbstyle}/results[_full]/<name>/
# Logs: sft_output/logs/ego_jrdbtrain_6models/<port>_{server,ego,jrdbstyle}.log
# Usage (inside tmux): GPUS="0 1 2 3 4 5" bash scripts/run_groups_ego_jrdbtrain_6models.sh
#   GPUS = one GPU per job, in job order (default 0..5). Add --skip_existing via EXTRA="--skip_existing" to resume.

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
S=sft_output
LOGS=sft_output/logs/ego_jrdbtrain_6models
mkdir -p "$LOGS"
read -r -a GPU_LIST <<< "${GPUS:-0 1 2 3 4 5}"

# port | checkpoint | served model name | prompt method | type   (job i runs on GPU_LIST[i])
JOBS=(
  "8071|$S/qwen2.5-7b-groups-merged-nodspy|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain|p1|single"
  "8072|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain|p1|single"
  "8073|$S/qwen2.5-7b-groups-merged-nodspy-full|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain|p1|full"
  "8074|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-full-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain|p1|full"
  "8075|$S/qwen2.5-vl-7b-groups-merged-nodspy|Qwen/Qwen2.5-VL-7B-Instruct-SFT-JRDBTrain|p1_visual|single"
  "8076|$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain|p1_visual|single"
)

run_job() {  # <gpu> <job spec>
  local gpu=$1 port ckpt name prompt type
  IFS='|' read -r port ckpt name prompt type <<< "$2"
  echo "[$(date)] GPU $gpu: starting server $port ($ckpt)"
  CUDA_VISIBLE_DEVICES=$gpu vllm serve "$ckpt" --served-model-name "$name" --api-key jrdb --port "$port" \
    > "$LOGS/${port}_server.log" 2>&1 &
  local pid=$!
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] GPU $gpu: server $port died, see $LOGS/${port}_server.log"; return 1; fi
    sleep 15
  done
  echo "[$(date)] GPU $gpu: server $port ready, EgoGroups_test ($name $prompt $type)"
  EGO_DATA=EgoGroups_test bash scripts/run_groups.bash "$name" "$port" jrdb "$prompt" "$type" $EXTRA > "$LOGS/${port}_ego.log" 2>&1
  echo "[$(date)] GPU $gpu: EgoGroups_test $port exit $?, now EgoGroups_test_jrdbstyle"
  EGO_DATA=EgoGroups_test_jrdbstyle bash scripts/run_groups.bash "$name" "$port" jrdb "$prompt" "$type" $EXTRA > "$LOGS/${port}_jrdbstyle.log" 2>&1
  echo "[$(date)] GPU $gpu: EgoGroups_test_jrdbstyle $port exit $?"
  kill $pid; wait $pid 2>/dev/null
}

for i in "${!JOBS[@]}"; do run_job "${GPU_LIST[$i]}" "${JOBS[$i]}" & done
wait
echo "[$(date)] all 6 JRDB-train -> EgoGroups_test runs done"
