#!/bin/bash
# JRDB test (JRDB_fixed_gold) and EgoGroups test (EgoGroups_test) group-detection eval of the EgoGroups-trained
# checkpoint with the validation clips held out ("-egogroups-train-valsplit"), one vLLM server on one GPU with both
# clients against it at once. Shares the GPU with the MINGLE eval, so vLLM is capped at 45% of the 143 GB.
# Usage: bash scripts/run_groups_egogroups_valsplit_eval.sh <gpu> <port> <checkpoint> <served name> <prompt> <type>
# Logs: sft_output/logs/egogroups_valsplit_eval/<port>_{server,jrdb_test,ego_test}.log
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc/CUDA toolkit here
LOGS=sft_output/logs/egogroups_valsplit_eval
mkdir -p "$LOGS"
GPU=$1 PORT=$2 CKPT=$3 NAME=$4 PROMPT=$5 TYPE=$6

echo "[$(date)] GPU $GPU: starting server $PORT ($CKPT)"
CUDA_VISIBLE_DEVICES=$GPU vllm serve "$CKPT" --served-model-name "$NAME" --api-key jrdb --port "$PORT" \
  --gpu-memory-utilization 0.45 > "$LOGS/${PORT}_server.log" 2>&1 &
pid=$!
until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$PORT/v1/models" > /dev/null; do
  if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] server $PORT died, see $LOGS/${PORT}_server.log"; exit 1; fi
  sleep 15
done
reply=$(curl -s -H "Authorization: Bearer jrdb" -H "Content-Type: application/json" "http://localhost:$PORT/v1/chat/completions" \
  -d "{\"model\": \"$NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"Say hello in five words.\"}], \"max_tokens\": 20}" \
  | python -c 'import sys, json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>&1)
echo "[$(date)] server $PORT smoke reply: $reply"
if [[ "$reply" == *"!!!!"* || -z "$reply" ]]; then echo "[$(date)] $PORT looks broken, stopping"; kill $pid; exit 1; fi

echo "[$(date)] running JRDB test and EgoGroups test ($NAME $PROMPT $TYPE)"
( bash scripts/run_groups_jrdb.bash "$NAME" "$PORT" jrdb "$PROMPT" "$TYPE" > "$LOGS/${PORT}_jrdb_test.log" 2>&1
  echo "[$(date)] JRDB test finished with exit code $?" ) &
c1=$!
( bash scripts/run_groups.bash "$NAME" "$PORT" jrdb "$PROMPT" "$TYPE" > "$LOGS/${PORT}_ego_test.log" 2>&1
  echo "[$(date)] EgoGroups test finished with exit code $?" ) &
c2=$!
wait $c1 $c2
kill $pid; wait $pid 2>/dev/null
echo "[$(date)] all done"
