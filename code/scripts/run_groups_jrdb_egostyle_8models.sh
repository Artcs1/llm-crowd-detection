#!/bin/bash
# JRDB test with EgoGroups-style 3D (JRDB_fixed_gold_egostyle, 1,829 keyframes) group-detection eval of the 8
# EgoGroups-trained SFT/RL checkpoints, one vLLM server per GPU on GPUs 0-7: start the server, wait until it answers,
# run scripts/run_groups_jrdb.bash against it, stop the server. Predictions land in results/predictions/JRDB_fixed_gold_egostyle/.
# Logs: sft_output/logs/jrdb_egostyle_8models/<port>_{server,client}.log
# Run inside tmux, e.g.: tmux new -d -s jrdb_ego 'bash scripts/run_groups_jrdb_egostyle_8models.sh'

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export JRDB_SPLIT=JRDB_fixed_gold_egostyle
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
S=sft_output
LOGS=sft_output/logs/jrdb_egostyle_8models
mkdir -p "$LOGS"

# port | checkpoint | served model name | prompt method | type   (job i runs on GPU i)
JOBS=(
  "8061|$S/qwen2.5-7b-groups-merged-nodspy-egogroups-train|Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain|p1|single"
  "8062|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-egogroups-train-3ep-epoch1|Qwen/Qwen2.5-7B-Instruct-RL-EgoGroupTrain|p1|single"
  "8063|$S/qwen2.5-7b-groups-merged-nodspy-full-egogroups-train|Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain|p1|full"
  "8064|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-full-egogroups-train-3ep-epoch1|Qwen/Qwen2.5-7B-Instruct-RL-EgoGroupTrain|p1|full"
  "8065|$S/qwen2.5-vl-7b-groups-merged-nodspy-egogroups-train|Qwen/Qwen2.5-VL-7B-Instruct-SFT-EgoGroupTrain|p1_visual|single"
  "8066|$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-egogroups-train-epoch1|Qwen/Qwen2.5-VL-7B-Instruct-RL-EgoGroupTrain|p1_visual|single"
  "8067|$S/qwen2.5-vl-7b-groups-merged-nodspy-full-egogroups-train|Qwen/Qwen2.5-VL-7B-Instruct-SFT-EgoGroupTrain|p1_visual|full"
  "8068|$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-full-egogroups-train-1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-EgoGroupTrain|p1_visual|full"
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
  echo "[$(date)] GPU $gpu: server $port ready, running client ($name $prompt $type)"
  bash scripts/run_groups_jrdb.bash "$name" "$port" jrdb "$prompt" "$type" > "$LOGS/${port}_client.log" 2>&1
  echo "[$(date)] GPU $gpu: client $port finished with exit code $?"
  kill $pid; wait $pid 2>/dev/null
}

for g in 0 1 2 3 4 5 6 7; do run_job "$g" "${JOBS[$g]}" & done
wait
echo "[$(date)] all 8 egostyle runs done"
