#!/bin/bash
# Resume one JRDB egostyle eval split across several GPUs: one vLLM server per GPU for the same checkpoint, and one
# client per GPU with --skip_existing (keyframes already saved are skipped) and --num_parts/--part_id (an interleaved
# split of the remaining keyframes).
# Usage: bash scripts/run_groups_jrdb_egostyle_split.sh <checkpoint> <served name> <prompt> <type> <gpu:port> [<gpu:port> ...]
# Logs: sft_output/logs/jrdb_egostyle_8models/<port>_{server,client}.log

umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export JRDB_SPLIT=JRDB_fixed_gold_egostyle
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
LOGS=sft_output/logs/jrdb_egostyle_8models
CKPT=$1 NAME=$2 PROMPT=$3 TYPE=$4; shift 4
N=$#

run_part() {  # <gpu> <port> <part id>
  local gpu=$1 port=$2 k=$3
  echo "[$(date)] GPU $gpu: starting server $port ($CKPT), part $k/$N"
  CUDA_VISIBLE_DEVICES=$gpu vllm serve "$CKPT" --served-model-name "$NAME" --api-key jrdb --port "$port" \
    > "$LOGS/${port}_server.log" 2>&1 &
  local pid=$!
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$port/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] GPU $gpu: server $port died, see $LOGS/${port}_server.log"; return 1; fi
    sleep 15
  done
  echo "[$(date)] GPU $gpu: server $port ready, running part $k/$N ($NAME $PROMPT $TYPE)"
  bash scripts/run_groups_jrdb.bash "$NAME" "$port" jrdb "$PROMPT" "$TYPE" --skip_existing --num_parts "$N" --part_id "$k" \
    > "$LOGS/${port}_client.log" 2>&1
  echo "[$(date)] GPU $gpu: client $port part $k/$N finished with exit code $?"
  kill $pid; wait $pid 2>/dev/null
}

k=0
for gp in "$@"; do run_part "${gp%%:*}" "${gp##*:}" $k & k=$((k + 1)); done
wait
echo "[$(date)] all $N parts of $NAME $PROMPT $TYPE done"
