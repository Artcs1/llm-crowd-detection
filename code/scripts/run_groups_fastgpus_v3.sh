#!/bin/bash
# Group-detection test of one checkpoint as fast as possible on $DATASET: jrdb (JRDB_fixed_gold, 1,829 keyframes,
# scripts/run_groups_jrdb.bash), jrdb_egostyle (JRDB_fixed_gold_egostyle, same keyframes with EgoGroups-style 3D), ego_jrdbstyle
# (EgoGroups_test_jrdbstyle, same clips with JRDB-style 3D) or ego (EgoGroups_test, 4,500 keyframes, scripts/run_groups.bash; default): one vLLM server on each GPU of
# $GPUS (default 0 1 2 3), capped at the matching $UTILS fraction (default 0.45 -- the GPUs are shared with other jobs),
# ports from $PORT0 (default 8091), and 4 clients per server, each on an
# interleaved 1/16 of the keyframes not yet saved (--skip_existing --num_parts 16 --part_id k), so a stopped run resumes.
# Waits first until the merged checkpoint is complete (index + every listed shard readable, size stable for a minute).
# Copy of run_groups_fastgpus_v2.sh with $PER (clients per server, default 4) settable, e.g. PER=16 for one GPU.
# MAXLEN caps vLLM's max_model_len (e.g. MAXLEN=32768 for Qwen3-VL, whose 262K default needs ~36 GiB of KV cache).
# Usage: [PER=16 DATASET=ego|ego_jrdbstyle|jrdb|jrdb_egostyle GPUS="4 5 6 7" UTILS="0.45 0.45 0.45 0.3" PORT0=8095] bash scripts/run_groups_fastgpus_v3.sh <checkpoint> <served name> <prompt: p1|p1_visual> <type: single|full> <log dir>
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
CKPT=$1 NAME=$2 PROMPT=$3 TYPE=$4 LOGS=$5
mkdir -p "$LOGS"
GPUS=(${GPUS:-0 1 2 3}) UTILS=(${UTILS:-0.45 0.45 0.45 0.45}) PORT0=${PORT0:-8091} PER=${PER:-4}
N=$(( ${#GPUS[@]} * PER ))
if [ "$PROMPT" = p1 ]; then MODE=llm; else MODE=vlm_image; fi
[ "$TYPE" = full ] && RES=results_full || RES=results
if [ "${DATASET:-ego}" = jrdb ]; then
  export JRDB_SPLIT=JRDB_fixed_gold
  CLIENT=scripts/run_groups_jrdb.bash TOTAL=1829 OUT=../results/predictions/JRDB_fixed_gold/$RES/${NAME#*/}
elif [ "$DATASET" = jrdb_egostyle ]; then  # JRDB test with EgoGroups-style 3D, for EgoGroups-trained models
  export JRDB_SPLIT=JRDB_fixed_gold_egostyle
  CLIENT=scripts/run_groups_jrdb.bash TOTAL=1829 OUT=../results/predictions/JRDB_fixed_gold_egostyle/$RES/${NAME#*/}
elif [ "$DATASET" = ego_jrdbstyle ]; then  # EgoGroups test with JRDB-style 3D, for JRDB-trained models
  export EGO_DATA=EgoGroups_test_jrdbstyle
  CLIENT=scripts/run_groups.bash TOTAL=4500 OUT=../results/predictions/EgoGroups_test_jrdbstyle/$RES/${NAME#*/}
else
  export EGO_DATA=EgoGroups_test
  CLIENT=scripts/run_groups.bash TOTAL=4500 OUT=../results/predictions/EgoGroups_test/$RES/${NAME#*/}
fi

complete() {
  [ -r "$CKPT/config.json" ] && [ -r "$CKPT/model.safetensors.index.json" ] || return 1
  local f
  for f in $(python -c "import json,sys; print(' '.join(sorted(set(json.load(open(sys.argv[1]))['weight_map'].values()))))" "$CKPT/model.safetensors.index.json"); do
    [ -r "$CKPT/$f" ] || return 1
  done
}
echo "[$(date)] $NAME ($PROMPT $TYPE, ${DATASET:-ego}): waiting for $CKPT"
until complete; do sleep 60; done
s1=$(du -sb "$CKPT" | cut -f1); sleep 60; until [ "$(du -sb "$CKPT" | cut -f1)" = "$s1" ]; do s1=$(du -sb "$CKPT" | cut -f1); sleep 60; done

pids=()
for i in "${!GPUS[@]}"; do
  port=$((PORT0 + i))
  echo "[$(date)] GPU ${GPUS[$i]}: starting server $port"
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} vllm serve "$CKPT" --served-model-name "$NAME" --api-key jrdb --port $port \
    --gpu-memory-utilization ${UTILS[$i]} ${MAXLEN:+--max-model-len $MAXLEN} > "$LOGS/server_$port.log" 2>&1 &
  pids+=($!)
done
for i in "${!GPUS[@]}"; do
  port=$((PORT0 + i))
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 ${pids[$i]} 2>/dev/null; then echo "[$(date)] server $port died, see $LOGS/server_$port.log"; kill ${pids[@]}; wait ${pids[@]} 2>/dev/null; exit 1; fi
    sleep 10
  done
done
echo "[$(date)] 4 servers ready"
reply=$(curl -s -H "Authorization: Bearer jrdb" -H "Content-Type: application/json" "http://localhost:$PORT0/v1/chat/completions" \
  -d "{\"model\": \"$NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 20}" \
  | python -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>&1)
echo "[$(date)] smoke reply: $reply"
if [[ "$reply" == *"!!!!"* || -z "$reply" ]]; then echo "[$(date)] looks broken, skipping"; kill ${pids[@]}; wait ${pids[@]} 2>/dev/null; exit 1; fi

echo "[$(date)] $(find "$OUT" -name '*.json' 2>/dev/null | wc -l)/$TOTAL already saved; running $N clients"
cpids=()
for k in $(seq 0 $((N - 1))); do
  port=$((PORT0 + k % ${#GPUS[@]}))
  bash $CLIENT "$NAME" $port jrdb "$PROMPT" "$TYPE" --skip_existing --num_parts $N --part_id $k \
    > "$LOGS/client_part$k.log" 2>&1 &
  cpids+=($!)
done
wait ${cpids[@]}
echo "[$(date)] all $N clients finished; saved $(find "$OUT" -name '*.json' | wc -l)/$TOTAL in $OUT"
kill ${pids[@]}; wait ${pids[@]} 2>/dev/null
sleep 20  # let the GPUs release the servers' memory before the next job
echo "[$(date)] done $NAME ($PROMPT $TYPE, ${DATASET:-ego})"
