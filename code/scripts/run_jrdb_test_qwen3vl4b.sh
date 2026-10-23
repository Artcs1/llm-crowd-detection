#!/bin/bash
# JRDB test (JRDB_fixed_gold, 1,829 keyframes, p1_visual single) of the six Qwen3-VL-4B checkpoints of
# scripts/run_jrdb_qwen3vl4b_chain.sh, one lane per LoRA type, each lane on one GPU in sequence (scripts/run_groups_fastgpus_v3.sh,
# one vLLM server + $PER clients; resumes with --skip_existing).
#   LANE=lmlora    : SFT, RL train scenes, RL val 3 ep -- LM-only LoRA
#   LANE=alllinear : the same, all-linear LoRA
# Predictions: ../results/predictions/JRDB_fixed_gold/results/Qwen3-VL-4B-Instruct-*/vlm_image/detany_3D/p1_visual/15/
# Logs: sft_output/logs/jrdb_test_qwen3vl4b/<served name>/
# Usage (inside tmux): LANE=lmlora GPU=1 PORT=8101 bash scripts/run_jrdb_test_qwen3vl4b.sh
umask 000
cd "$(dirname "$0")/.."
S=sft_output
GPU=${GPU:-1} PORT=${PORT:-8101} UTIL=${UTIL:-0.5} PER=${PER:-16} MAXLEN=${MAXLEN:-32768}  # Qwen3-VL default 262K context does not fit a 50% cap
if [ "$LANE" = alllinear ]; then sfx= tag=; else sfx=-lmlora tag=-LMLoRA; fi
JOBS=(
  "$S/qwen3-vl-4b-groups-merged-nodspy-jrdbtrain$sfx|Qwen/Qwen3-VL-4B-Instruct-SFT-JRDBTrain$tag"
  "$S/qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain$sfx-1ep-nobiaskl-g8p1|Qwen/Qwen3-VL-4B-Instruct-RL-JRDBTrain$tag"
  "$S/qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain$sfx-rlval-nobiaskl-g8p1|Qwen/Qwen3-VL-4B-Instruct-RL-JRDBTrain-RLVal3ep$tag"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r ckpt name <<< "$job"
  DATASET=jrdb MAXLEN=$MAXLEN GPUS="$GPU" UTILS="$UTIL" PORT0=$PORT PER=$PER \
    bash scripts/run_groups_fastgpus_v3.sh "$ckpt" "$name" p1_visual single "$S/logs/jrdb_test_qwen3vl4b/${name#*/}"
done
echo "[$(date)] lane ${LANE:-lmlora} done"
