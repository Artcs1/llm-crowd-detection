#!/bin/bash
# JRDB test (JRDB_fixed_gold, 1,829 keyframes) of a GRPO-on-JRDB-val checkpoint (LLM p1), as fast as possible: one vLLM server
# on each of GPUs 0-3 (capped at 45% -- they share the GPUs with the MINGLE training) and 4 clients per server, each on
# an interleaved 1/16 of the keyframes not yet saved (--skip_existing --num_parts 16 --part_id k).
# Usage: bash scripts/run_groups_jrdb_rlval_fast.sh [single|full]   (default single)
#   single: qwen2.5-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval
#   full:   qwen2.5-7b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-rlval
# With full, waits first until that merged checkpoint is complete (index + every shard readable, sizes stable).
# Logs: sft_output/logs/jrdb_rlval_eval[_full]/{server_<port>,client_part<k>}.log
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
TYPE=${1:-single}
if [ "$TYPE" = full ]; then
  LOGS=sft_output/logs/jrdb_rlval_eval_full
  CKPT=sft_output/qwen2.5-7b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-rlval
  RES=results_full
else
  LOGS=sft_output/logs/jrdb_rlval_eval
  CKPT=sft_output/qwen2.5-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval
  RES=results
fi
NAME=Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLVal
mkdir -p "$LOGS"

complete() {  # merged checkpoint fully written: index present, every shard it lists present and readable
  [ -r "$CKPT/config.json" ] && [ -r "$CKPT/model.safetensors.index.json" ] || return 1
  local f
  for f in $(python -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" "$CKPT/model.safetensors.index.json"); do
    [ -r "$CKPT/$f" ] || return 1
  done
}
echo "[$(date)] waiting for $CKPT"
until complete; do sleep 60; done
s1=$(du -sb "$CKPT" | cut -f1); sleep 60; until [ "$(du -sb "$CKPT" | cut -f1)" = "$s1" ]; do s1=$(du -sb "$CKPT" | cut -f1); sleep 60; done
GPUS=(0 1 2 3) PER=4
N=$(( ${#GPUS[@]} * PER ))
OUT=../results/predictions/JRDB_fixed_gold/$RES/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLVal

pids=()
for i in "${!GPUS[@]}"; do
  port=$((8091 + i))
  echo "[$(date)] GPU ${GPUS[$i]}: starting server $port"
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} vllm serve "$CKPT" --served-model-name "$NAME" --api-key jrdb --port $port \
    --gpu-memory-utilization 0.45 > "$LOGS/server_$port.log" 2>&1 &
  pids+=($!)
done
for i in "${!GPUS[@]}"; do
  port=$((8091 + i))
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 ${pids[$i]} 2>/dev/null; then echo "[$(date)] server $port died, see $LOGS/server_$port.log"; kill ${pids[@]}; exit 1; fi
    sleep 10
  done
  echo "[$(date)] server $port ready"
done
reply=$(curl -s -H "Authorization: Bearer jrdb" -H "Content-Type: application/json" "http://localhost:8091/v1/chat/completions" \
  -d "{\"model\": \"$NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 20}" \
  | python -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>&1)
echo "[$(date)] smoke reply: $reply"
if [[ "$reply" == *"!!!!"* || -z "$reply" ]]; then echo "[$(date)] looks broken, stopping"; kill ${pids[@]}; exit 1; fi

echo "[$(date)] $(find $OUT -name '*.json' 2>/dev/null | wc -l)/1829 already saved; running $N clients"
cpids=()
for k in $(seq 0 $((N - 1))); do
  port=$((8091 + k % ${#GPUS[@]}))
  bash scripts/run_groups_jrdb.bash "$NAME" $port jrdb p1 "$TYPE" --skip_existing --num_parts $N --part_id $k \
    > "$LOGS/client_part$k.log" 2>&1 &
  cpids+=($!)
done
wait ${cpids[@]}
echo "[$(date)] all $N clients finished; saved $(find $OUT -name '*.json' | wc -l)/1829"
kill ${pids[@]}; wait ${pids[@]} 2>/dev/null
echo "[$(date)] done"
