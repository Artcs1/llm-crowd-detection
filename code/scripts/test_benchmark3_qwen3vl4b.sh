#!/bin/bash
# benchmark3 test (OOD_city = EgoGroupsOODCity_Test vs gt benchmark3_id_test, OOD_regions = EgoGroupsOODRegions_Test vs
# benchmark3_ood_test) of one Qwen3-VL-4B model, p1_visual single: one vLLM server on $GPU, both test sets at once through
# scripts/run_zeroshot_egogroupscity.sh (which reuses the server), then the server is stopped.
#   GPU=0 PORT=8301 MODEL=sft_output/qwen3-vl-4b-groups-merged-nodspy-egogroups-benchmark3 \
#     NAME=Qwen/Qwen3-VL-4B-Instruct-SFT-Benchmark3 bash scripts/test_benchmark3_qwen3vl4b.sh
#   GPU=2 PORT=8302 bash scripts/test_benchmark3_qwen3vl4b.sh          (zero-shot base model)
# Metrics: sft_output/logs/egogroupscity_zeroshot/<short>_p1_visual_<set>_metrics.txt
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0
GPU=${GPU:-0} PORT=${PORT:-8301} MODEL=${MODEL:-Qwen/Qwen3-VL-4B-Instruct} NPARTS=${NPARTS:-48}
export NAME=${NAME:-$MODEL}
# JRDBSTYLE=1: the *_jrdbstyle test folders (EgoGroups 3D in the JRDB frame), for models trained in that frame
# (e.g. --dataset jrdbego-benchmark3)
STYLE=$([ -n "$JRDBSTYLE" ] && echo _jrdbstyle)
# PROMPT=p1_social for --variant p1_social models (default p1_visual)
PROMPT=${PROMPT:-p1_visual}
L=sft_output/logs/egogroupscity_zeroshot/${NAME#*/}_benchmark3_server.log
mkdir -p "$(dirname "$L")"
CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" --served-model-name "$NAME" --api-key egocity --port "$PORT" \
  --gpu-memory-utilization 0.90 --max-model-len 32768 --max-num-seqs 128 --limit-mm-per-prompt '{"image": 1}' > "$L" 2>&1 &
pid=$!
until curl -sf -H "Authorization: Bearer egocity" "http://localhost:$PORT/v1/models" > /dev/null; do
  kill -0 $pid 2>/dev/null || { echo "[$(date)] server died, see $L"; exit 1; }
  sleep 10
done
echo "[$(date)] server up on GPU $GPU port $PORT ($NAME)"
DATA=EgoGroupsOODCity_Test$STYLE GT=benchmark3_id_test bash scripts/run_zeroshot_egogroupscity.sh "$GPU" "$MODEL" "$PROMPT" "$PORT" "$NPARTS" &
a=$!
DATA=EgoGroupsOODRegions_Test$STYLE GT=benchmark3_ood_test bash scripts/run_zeroshot_egogroupscity.sh "$GPU" "$MODEL" "$PROMPT" "$PORT" "$NPARTS" &
b=$!
wait $a $b
kill $pid; wait $pid 2>/dev/null
echo "[$(date)] $NAME benchmark3 tests done"
