#!/bin/bash
# EgoGroups test with JRDB-style 3D (EgoGroups_test_jrdbstyle, 4,500 keyframes) group-detection eval of the 6 JRDB-trained
# SFT/RL checkpoints (Sept, no suffix), one vLLM server per GPU on GPUs 0-5: start the server, wait until it answers,
# run scripts/run_groups.bash against it, stop the server. Predictions land in results/predictions/EgoGroups_test_jrdbstyle/.
# Waits first for both JRDB egostyle VLM full splits (scripts/run_groups_jrdb_egostyle_split.sh) to finish.
# Logs: sft_output/logs/egogroups_jrdbstyle_6models/<port>_{server,client}.log
# Run inside tmux, e.g.: tmux new -d -s ego_jrdbstyle 'bash scripts/run_groups_egogroups_jrdbstyle_6models.sh'

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export EGO_DATA=EgoGroups_test_jrdbstyle
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
S=sft_output
LOGS=sft_output/logs/egogroups_jrdbstyle_6models
mkdir -p "$LOGS"

echo "[$(date)] waiting for both JRDB egostyle VLM full splits to finish"
until grep -q "all 4 parts" sft_output/logs/jrdb_egostyle_split_8067_chain.log 2>/dev/null \
   && grep -q "all 4 parts" sft_output/logs/jrdb_egostyle_split_8068_4x_chain.log 2>/dev/null; do sleep 60; done

# port | checkpoint | served model name | prompt method | type   (job i runs on GPU i)
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
  until [ "$(nvidia-smi -i $gpu --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 10; done
  echo "[$(date)] GPU $gpu: starting server $port ($ckpt)"
  CUDA_VISIBLE_DEVICES=$gpu vllm serve "$ckpt" --served-model-name "$name" --api-key jrdb --port "$port" \
    > "$LOGS/${port}_server.log" 2>&1 &
  local pid=$!
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] GPU $gpu: server $port died, see $LOGS/${port}_server.log"; return 1; fi
    sleep 15
  done
  echo "[$(date)] GPU $gpu: server $port ready, running client ($name $prompt $type)"
  bash scripts/run_groups.bash "$name" "$port" jrdb "$prompt" "$type" > "$LOGS/${port}_client.log" 2>&1
  echo "[$(date)] GPU $gpu: client $port finished with exit code $?"
  kill $pid; wait $pid 2>/dev/null
}

for g in 0 1 2 3 4 5; do run_job "$g" "${JOBS[$g]}" & done
wait
echo "[$(date)] all 6 EgoGroups jrdbstyle runs done"
