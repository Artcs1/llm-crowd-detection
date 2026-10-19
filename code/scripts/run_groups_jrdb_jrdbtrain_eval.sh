#!/bin/bash
# JRDB val (JRDB_val_fixed_gold, 372 keyframes) then JRDB test (JRDB_fixed_gold, 1,829 keyframes) group-detection eval
# of the 6 checkpoints from scripts/run_jrdb_val_split_sft_rl.sh (SFT/RL trained with the 7 JRDB val scenes held out,
# "-jrdbtrain"), one vLLM server per GPU on GPUs 4-7: start the server, wait until it answers, run
# scripts/run_groups_jrdb.bash on val then test, stop the server, next job. A job whose checkpoint is missing (its
# training failed) is skipped, and so is one whose smoke-test reply is "!!!!" (NaN weights).
# Logs: sft_output/logs/jrdb_jrdbtrain_eval/<port>_{server,val,test}.log
# Run inside tmux, e.g.: tmux new -d -s jrdb_eval2 'bash scripts/run_groups_jrdb_jrdbtrain_eval.sh'

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
S=sft_output
LOGS=sft_output/logs/jrdb_jrdbtrain_eval
mkdir -p "$LOGS"

# Shares GPUs 4-7 with the MINGLE EgoGroups_test eval (~45-56 GB each), so vLLM is capped at 45% of the 143 GB.
UTIL=0.45

# port | checkpoint | served model name | prompt method | type
JOBS=(
  "8061|$S/qwen2.5-7b-groups-merged-nodspy-jrdbtrain|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain-v2|p1|single"
  "8062|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-v2|p1|single"
  "8063|$S/qwen2.5-7b-groups-merged-nodspy-full-jrdbtrain|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain-v2|p1|full"
  "8064|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-v2|p1|full"
  "8065|$S/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain|Qwen/Qwen2.5-VL-7B-Instruct-SFT-JRDBTrain-v2|p1_visual|single"
  "8066|$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-v2|p1_visual|single"
)
# jobs per GPU: GPU 4 -> 8061, 8065; GPU 5 -> 8062, 8066; GPU 6 -> 8063; GPU 7 -> 8064
GPU_JOBS=("4|0 4" "5|1 5" "6|2" "7|3")

run_job() {  # <gpu> <job spec>
  local gpu=$1 port ckpt name prompt type
  IFS='|' read -r port ckpt name prompt type <<< "$2"
  if [ ! -f "$ckpt/config.json" ]; then echo "[$(date)] GPU $gpu: $ckpt missing, skipping $port"; return 1; fi
  echo "[$(date)] GPU $gpu: starting server $port ($ckpt)"
  CUDA_VISIBLE_DEVICES=$gpu vllm serve "$ckpt" --served-model-name "$name" --api-key jrdb --port "$port" \
    --gpu-memory-utilization $UTIL > "$LOGS/${port}_server.log" 2>&1 &
  local pid=$!
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] GPU $gpu: server $port died, see $LOGS/${port}_server.log"; return 1; fi
    sleep 15
  done
  # smoke test: the NaN-broken checkpoints of Oct 15 answered "!!!!..." -- skip a checkpoint that does that
  local reply
  reply=$(curl -s -H "Authorization: Bearer jrdb" -H "Content-Type: application/json" "http://localhost:$port/v1/chat/completions" \
    -d "{\"model\": \"$name\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 20}" \
    | python -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>&1)
  echo "[$(date)] GPU $gpu: server $port smoke reply: $reply"
  if [[ "$reply" == *"!!!!"* || -z "$reply" ]]; then echo "[$(date)] GPU $gpu: $port looks broken, skipping"; kill $pid; wait $pid 2>/dev/null; return 1; fi
  echo "[$(date)] GPU $gpu: server $port ready, running val ($name $prompt $type)"
  JRDB_SPLIT=JRDB_val_fixed_gold bash scripts/run_groups_jrdb.bash "$name" "$port" jrdb "$prompt" "$type" > "$LOGS/${port}_val.log" 2>&1
  echo "[$(date)] GPU $gpu: val $port finished with exit code $?, running test"
  bash scripts/run_groups_jrdb.bash "$name" "$port" jrdb "$prompt" "$type" > "$LOGS/${port}_test.log" 2>&1
  echo "[$(date)] GPU $gpu: test $port finished with exit code $?"
  kill $pid; wait $pid 2>/dev/null
}

for spec in "${GPU_JOBS[@]}"; do
  IFS='|' read -r gpu idxs <<< "$spec"
  ( for i in $idxs; do run_job "$gpu" "${JOBS[$i]}"; done ) &
done
wait
echo "[$(date)] all JRDB-train checkpoint evals done"
