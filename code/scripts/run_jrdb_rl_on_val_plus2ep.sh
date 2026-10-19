#!/bin/bash
# A third GRPO epoch on the 7 JRDB val scenes (--rl_on_val), on top of each -1ep-rlval-plus1ep model
# (scripts/run_jrdb_rl_on_val_plus1ep.sh), since the second epoch still raised JRDB test F1. Starts from the merged
# -1ep-rlval-plus1ep checkpoint (--init_model): fresh LoRA, KL reference = that checkpoint, LR schedule restarts.
# Runs on one GPU in sequence after the RL-on-val chain (pid $WAIT_PID) has exited.
# Checkpoints: sft_output/*-jrdbtrain-1ep-rlval-plus2ep. Logs: sft_output/logs/jrdb_rl_on_val_plus2ep/
# Usage (inside tmux): GPU=1 bash scripts/run_jrdb_rl_on_val_plus2ep.sh

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_rl_on_val_plus2ep
mkdir -p "$LOGS"

if [ -n "$WAIT_PID" ]; then
  echo "[$(date)] GPU $GPU: waiting for the RL-on-val chain (pid $WAIT_PID) to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

# name | master port | rl script | mode args | 2-epoch RL-on-val checkpoint (start point)
JOBS=(
  "llm_single|29581|rl_training_llm_nodspy_merged.py|--mode single|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval-plus1ep"
  "llm_full|29582|rl_training_llm_nodspy_merged.py|--mode full|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-rlval-plus1ep"
  "vlm_single|29583|rl_training_vlm_nodspy_merged.py|--variant p1_visual|$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-1ep-rlval-plus1ep"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r name mport script margs init <<< "$job"
  if [ ! -f "$init/config.json" ]; then echo "[$(date)] GPU $GPU: no $init, skipping $name"; continue; fi
  echo "[$(date)] GPU $GPU: 3rd epoch RL-on-val $name (from $init)"
  # shellcheck disable=SC2086
  if MASTER_PORT=$mport python "$script" --dataset jrdb $margs --epochs 1 --jrdb_val_split --rl_on_val --init_model "$init" \
       --gpu "$GPU" > "$LOGS/rl_$name.log" 2>&1; then
    find "$S"/*-1ep-rlval-plus2ep -type f ! -perm -o+r -exec chmod a+rw {} + 2>/dev/null  # readable from the other workstation
    echo "[$(date)] GPU $GPU: 3rd epoch $name done"
  else
    echo "[$(date)] GPU $GPU: 3rd epoch $name FAILED (see $LOGS/rl_$name.log)"
  fi
done
echo "[$(date)] all 3rd epoch RL-on-val runs done"
