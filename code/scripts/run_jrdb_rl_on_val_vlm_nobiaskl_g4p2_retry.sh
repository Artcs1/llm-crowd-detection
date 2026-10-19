#!/bin/bash
# Retry of the g4p2 job of scripts/run_jrdb_rl_on_val_vlm_nobiaskl.sh (SIGKILLed at step 75/582 on Oct 19 01:09, not by
# us, no checkpoint saved), queued on the same GPU after that chain (pid $WAIT_PID, now running g8p4) exits.
# Usage (inside tmux): WAIT_PID=<pid> GPU=1 bash scripts/run_jrdb_rl_on_val_vlm_nobiaskl_g4p2_retry.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_rl_on_val_vlm_nobiaskl
if [ -n "$WAIT_PID" ]; then
  echo "[$(date)] GPU $GPU: waiting for pid $WAIT_PID to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
echo "[$(date)] GPU $GPU: VLM RL-on-val 3 epochs, no bias KL, g4p2 (retry)"
if MASTER_PORT=29593 python rl_training_vlm_nodspy_merged.py --dataset jrdb --variant p1_visual --epochs 3 \
     --jrdb_val_split --rl_on_val --no_bias_kl --num_generations 4 --prompts_per_step 2 --save-every-epoch --gpu "$GPU" \
     > "$LOGS/rl_vlm_g4p2.log" 2>&1; then
  find "$S"/qwen2.5-vl-7b-groups-grpo-*-jrdbtrain-rlval-nobiaskl* -type f ! -perm -o+r -exec chmod a+rw {} + 2>/dev/null
  echo "[$(date)] GPU $GPU: g4p2 done"
else
  echo "[$(date)] GPU $GPU: g4p2 FAILED (see $LOGS/rl_vlm_g4p2.log)"
fi
