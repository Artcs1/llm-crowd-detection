#!/bin/bash
# VLM p1_visual GRPO on the 7 JRDB val scenes (--rl_on_val) from the 20-scene SFT (-jrdbtrain), 3 epochs, with
# use_bias_correction_kl off (--no_bias_kl, the TRL 1.7 behaviour of the original 27-scene models), in sequence on one GPU:
#   1. same rollout setup as before: 4 rollouts x 2 prompts per optimizer step -> *-jrdbtrain-rlval-nobiaskl
#   2. 8 rollouts x 4 prompts per optimizer step                                -> *-jrdbtrain-rlval-nobiaskl-g8p4
# Both save a merged checkpoint after every epoch (-epoch1/2/3; the final one = epoch 3) so each epoch can be tested.
# Waits first for pid $WAIT_PID (the 3rd-epoch chain, scripts/run_jrdb_rl_on_val_plus2ep.sh).
# Logs: sft_output/logs/jrdb_rl_on_val_vlm_nobiaskl/
# Usage (inside tmux): WAIT_PID=<pid> GPU=1 bash scripts/run_jrdb_rl_on_val_vlm_nobiaskl.sh

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_rl_on_val_vlm_nobiaskl
mkdir -p "$LOGS"

if [ -n "$WAIT_PID" ]; then
  echo "[$(date)] GPU $GPU: waiting for pid $WAIT_PID to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi

# name | master port | rollout args
JOBS=(
  "g4p2|29591|--num_generations 4 --prompts_per_step 2"
  "g8p4|29592|--num_generations 8 --prompts_per_step 4"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r name mport rargs <<< "$job"
  echo "[$(date)] GPU $GPU: VLM RL-on-val 3 epochs, no bias KL, $name"
  # shellcheck disable=SC2086
  if MASTER_PORT=$mport python rl_training_vlm_nodspy_merged.py --dataset jrdb --variant p1_visual --epochs 3 \
       --jrdb_val_split --rl_on_val --no_bias_kl $rargs --save-every-epoch --gpu "$GPU" > "$LOGS/rl_vlm_$name.log" 2>&1; then
    find "$S"/qwen2.5-vl-7b-groups-grpo-*-jrdbtrain-rlval-nobiaskl* -type f ! -perm -o+r -exec chmod a+rw {} + 2>/dev/null  # readable from the other workstation
    echo "[$(date)] GPU $GPU: $name done"
  else
    echo "[$(date)] GPU $GPU: $name FAILED (see $LOGS/rl_vlm_$name.log)"
  fi
done
echo "[$(date)] all VLM no-bias-KL RL-on-val runs done"
