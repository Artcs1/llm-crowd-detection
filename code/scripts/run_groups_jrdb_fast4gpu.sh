#!/bin/bash
# JRDB test (JRDB_fixed_gold, 1,829 keyframes) of one checkpoint as fast as possible: one vLLM server on each of
# GPUs 0-3 (capped at 45% -- they share the GPUs with the MINGLE training) and 4 clients per server, each on an
# interleaved 1/16 of the keyframes not yet saved (--skip_existing --num_parts 16 --part_id k), so a stopped run resumes.
# Waits first until the merged checkpoint is complete (index + every listed shard readable, size stable for a minute).
# Usage: bash scripts/run_groups_jrdb_fast4gpu.sh <checkpoint> <served name> <prompt: p1|p1_visual> <type: single|full> <log dir>
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
CKPT=$1 NAME=$2 PROMPT=$3 TYPE=$4 LOGS=$5
mkdir -p "$LOGS"
GPUS=(0 1 2 3) PER=4
N=$(( ${#GPUS[@]} * PER ))
if [ "$PROMPT" = p1 ]; then MODE=llm; else MODE=vlm_image; fi
[ "$TYPE" = full ] && RES=results_full || RES=results
OUT=../results/predictions/JRDB_fixed_gold/$RES/${NAME#*/}

complete() {
  [ -r "$CKPT/config.json" ] && [ -r "$CKPT/model.safetensors.index.json" ] || return 1
  local f
  for f in $(python -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" "$CKPT/model.safetensors.index.json"); do
    [ -r "$CKPT/$f" ] || return 1
  done
}
echo "[$(date)] $NAME ($PROMPT $TYPE): waiting for $CKPT"
until complete; do sleep 60; done
s1=$(du -sb "$CKPT" | cut -f1); sleep 60; until [ "$(du -sb "$CKPT" | cut -f1)" = "$s1" ]; do s1=$(du -sb "$CKPT" | cut -f1); sleep 60; done

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
    if ! kill -0 ${pids[$i]} 2>/dev/null; then echo "[$(date)] server $port died, see $LOGS/server_$port.log"; kill ${pids[@]}; wait ${pids[@]} 2>/dev/null; exit 1; fi
    sleep 10
  done
done
echo "[$(date)] 4 servers ready"
reply=$(curl -s -H "Authorization: Bearer jrdb" -H "Content-Type: application/json" "http://localhost:8091/v1/chat/completions" \
  -d "{\"model\": \"$NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 20}" \
  | python -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>&1)
echo "[$(date)] smoke reply: $reply"
if [[ "$reply" == *"!!!!"* || -z "$reply" ]]; then echo "[$(date)] looks broken, skipping"; kill ${pids[@]}; wait ${pids[@]} 2>/dev/null; exit 1; fi

echo "[$(date)] $(find "$OUT" -name '*.json' 2>/dev/null | wc -l)/1829 already saved; running $N clients"
cpids=()
for k in $(seq 0 $((N - 1))); do
  port=$((8091 + k % ${#GPUS[@]}))
  bash scripts/run_groups_jrdb.bash "$NAME" $port jrdb "$PROMPT" "$TYPE" --skip_existing --num_parts $N --part_id $k \
    > "$LOGS/client_part$k.log" 2>&1 &
  cpids+=($!)
done
wait ${cpids[@]}
echo "[$(date)] all $N clients finished; saved $(find "$OUT" -name '*.json' | wc -l)/1829 in $OUT"
kill ${pids[@]}; wait ${pids[@]} 2>/dev/null
sleep 20  # let the GPUs release the servers' memory before the next job
echo "[$(date)] done $NAME ($PROMPT $TYPE)"
