#!/bin/bash
# Does GRPO help more on frames the SFT never saw? Starting from the 20-scene JRDB SFT checkpoints (-jrdbtrain), run
# 1-epoch GRPO on the 7 JRDB val scenes (--rl_on_val, 388 frames) instead of the 20 train scenes, then evaluate each
# on JRDB test (1,829 keyframes), all in sequence on one GPU: first all three trainings, then all three tests. Compare against the SFT (-JRDBTrain-v2 SFT) and the
# RL-on-train models (-JRDBTrain-v2 RL): if RL-on-val gains clearly more, a disjoint SFT/RL split of the train scenes is
# worth trying. Checkpoints: sft_output/*-jrdbtrain-1ep-rlval; results: Qwen2.5(-VL)-7B-Instruct-RL-JRDBTrain-RLval.
# Logs: sft_output/logs/jrdb_rl_on_val/
# Usage (inside tmux): GPU=1 bash scripts/run_jrdb_rl_on_val.sh

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_rl_on_val
mkdir -p "$LOGS"

# name | master port | vllm port | rl script | mode args | merged RL dir | served name | prompt | type
JOBS=(
  "llm_single|29561|8091|rl_training_llm_nodspy_merged.py|--mode single|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLval|p1|single"
  "llm_full|29562|8092|rl_training_llm_nodspy_merged.py|--mode full|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLval|p1|full"
  "vlm_single|29563|8093|rl_training_vlm_nodspy_merged.py|--variant p1_visual|$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLval|p1_visual|single"
)

# Phase 1: all three RL trainings
for job in "${JOBS[@]}"; do
  IFS='|' read -r name mport vport script margs ckpt served prompt type <<< "$job"
  echo "[$(date)] GPU $GPU: RL-on-val $name"
  # shellcheck disable=SC2086
  if MASTER_PORT=$mport python "$script" --dataset jrdb $margs --epochs 1 --jrdb_val_split --rl_on_val --gpu "$GPU" > "$LOGS/rl_$name.log" 2>&1; then
    find "$ckpt" -type f ! -perm -o+r -exec chmod a+rw {} + 2>/dev/null  # readable from the other workstation
    echo "[$(date)] GPU $GPU: RL $name done"
  else
    echo "[$(date)] GPU $GPU: RL $name FAILED (see $LOGS/rl_$name.log)"
  fi
done

# Phase 2: JRDB test of every model that trained
for job in "${JOBS[@]}"; do
  IFS='|' read -r name mport vport script margs ckpt served prompt type <<< "$job"
  if [ ! -f "$ckpt/config.json" ]; then echo "[$(date)] GPU $GPU: no $ckpt, skipping test $name"; continue; fi
  echo "[$(date)] GPU $GPU: serving $ckpt on $vport"
  CUDA_VISIBLE_DEVICES=$GPU vllm serve "$ckpt" --served-model-name "$served" --api-key jrdb --port "$vport" > "$LOGS/${vport}_server.log" 2>&1 &
  pid=$!
  until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:$vport/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] GPU $GPU: server $vport died, see $LOGS/${vport}_server.log"; continue 2; fi
    sleep 15
  done
  echo "[$(date)] GPU $GPU: server $vport ready, JRDB test $name ($served $prompt $type)"
  bash scripts/run_groups_jrdb.bash "$served" "$vport" jrdb "$prompt" "$type" > "$LOGS/${vport}_test.log" 2>&1
  echo "[$(date)] GPU $GPU: test $name finished with exit code $?"
  kill $pid; wait $pid 2>/dev/null
done
echo "[$(date)] all JRDB RL-on-val runs done"
