#!/bin/bash
# EgoGroups test (EgoGroups_test, 4,500 keyframes) group-detection eval of the 6 JRDB train-only checkpoints
# ("-jrdbtrain", trained with the 7 JRDB val scenes held out; served as "-JRDBTrain-v2"), one vLLM server per GPU,
# two jobs per GPU back to back. Starts once the JRDB val/test evals of those checkpoints
# (run_groups_jrdb_jrdbtrain_eval.sh) and the valsplit JRDB test (run_groups_egogroups_valsplit_eval.sh) are done.
# GPUs: 4, 7, and whichever of 5/6 the valsplit EgoGroups retry (retry_egogroups_valsplit_ego.sh) is not using.
# The GPUs are shared with the MINGLE eval, so vLLM is capped at 45% of the 143 GB.
# Logs: sft_output/logs/egogroups_jrdbtrain_eval/<port>_{server,client}.log
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
S=sft_output
LOGS=sft_output/logs/egogroups_jrdbtrain_eval
mkdir -p "$LOGS"

echo "[$(date)] waiting for the JRDB-train JRDB evals and the valsplit JRDB test to finish"
until grep -q "all JRDB-train checkpoint evals done" sft_output/logs/jrdb_jrdbtrain_eval_chain.log 2>/dev/null \
   && grep -q "all done" sft_output/logs/egogroups_valsplit_eval_chain.log 2>/dev/null; do sleep 60; done
# the valsplit EgoGroups retry takes GPU 5 or 6, whichever freed first
until grep -q "running EgoGroups test on GPU" sft_output/logs/egogroups_valsplit_ego_retry.log 2>/dev/null; do sleep 30; done
if grep -q "on GPU 5" sft_output/logs/egogroups_valsplit_ego_retry.log; then G3=6; else G3=5; fi
sleep 30  # let the finished servers release their memory
echo "[$(date)] running on GPUs 4, 7, $G3"

# port | checkpoint | served model name | prompt method | type
JOBS=(
  "8081|$S/qwen2.5-7b-groups-merged-nodspy-jrdbtrain|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain-v2|p1|single"
  "8082|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-v2|p1|single"
  "8083|$S/qwen2.5-7b-groups-merged-nodspy-full-jrdbtrain|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain-v2|p1|full"
  "8084|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-v2|p1|full"
  "8085|$S/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain|Qwen/Qwen2.5-VL-7B-Instruct-SFT-JRDBTrain-v2|p1_visual|single"
  "8086|$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-v2|p1_visual|single"
)
# full (slower) runs first, one per GPU: GPU 4 -> 8083, 8081; GPU 7 -> 8084, 8082; GPU G3 -> 8085, 8086
GPU_JOBS=("4|2 0" "7|3 1" "$G3|4 5")

run_job() {  # <gpu> <job spec>
  local gpu=$1 port ckpt name prompt type
  IFS='|' read -r port ckpt name prompt type <<< "$2"
  echo "[$(date)] GPU $gpu: starting server $port ($ckpt)"
  CUDA_VISIBLE_DEVICES=$gpu vllm serve "$ckpt" --served-model-name "$name" --api-key jrdb --port "$port" \
    --gpu-memory-utilization 0.45 > "$LOGS/${port}_server.log" 2>&1 &
  local pid=$!
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] GPU $gpu: server $port died, see $LOGS/${port}_server.log"; return 1; fi
    sleep 15
  done
  echo "[$(date)] GPU $gpu: server $port ready, running EgoGroups test ($name $prompt $type)"
  bash scripts/run_groups.bash "$name" "$port" jrdb "$prompt" "$type" > "$LOGS/${port}_client.log" 2>&1
  echo "[$(date)] GPU $gpu: client $port finished with exit code $?"
  kill $pid; wait $pid 2>/dev/null
}

for spec in "${GPU_JOBS[@]}"; do
  IFS='|' read -r gpu idxs <<< "$spec"
  ( for i in $idxs; do run_job "$gpu" "${JOBS[$i]}"; done ) &
done
wait
echo "[$(date)] all 6 JRDB-train EgoGroups test evals done"
