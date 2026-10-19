#!/bin/bash
# After the JRDB VLM LM-only-LoRA SFT (scripts/run_jrdb_vlm_sft_lmlora.sh, pid $WAIT_PID) exits, two GRPO runs on GPU $GPU,
# both LM-only LoRA (--lm_lora), use_bias_correction_kl off, 8 rollouts x 1 prompt (the VLM RL default):
#   1. RL on the same 20 JRDB train scenes as the SFT, 1 epoch (like the earlier RL-on-train runs)
#   2. RL on the 7 JRDB val scenes (--rl_on_val), 3 epochs with a merged checkpoint per epoch (epoch 1 compares with run 1)
# Checkpoints: sft_output/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-lmlora-1ep-nobiaskl-g8p1
#              sft_output/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-jrdbtrain-lmlora-rlval-nobiaskl-g8p1[-epoch{1,2,3}]
# Logs: sft_output/logs/jrdb_lmlora/rl_vlm_{train,val}.log
# Usage (inside tmux): WAIT_PID=1959326 GPU=0 bash scripts/run_jrdb_vlm_rl_lmlora.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-0}
S=sft_output
LOGS=$S/logs/jrdb_lmlora
mkdir -p "$LOGS"
if [ -n "$WAIT_PID" ]; then
  echo "[$(date)] GPU $GPU: waiting for the LM-only SFT (pid $WAIT_PID) to finish"
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
SFT=$S/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain-lmlora
if [ ! -f "$SFT/config.json" ]; then echo "[$(date)] GPU $GPU: no $SFT (SFT failed?), not starting RL"; exit 1; fi
COMMON="--dataset jrdb --variant p1_visual --mode single --jrdb_val_split --lm_lora --no_bias_kl --num_generations 8 --prompts_per_step 1"
echo "[$(date)] GPU $GPU: RL on the 20 train scenes, 1 epoch"
# shellcheck disable=SC2086
MASTER_PORT=29603 python rl_training_vlm_nodspy_merged.py $COMMON --epochs 1 --gpu "$GPU" > "$LOGS/rl_vlm_train.log" 2>&1
echo "[$(date)] GPU $GPU: RL train scenes finished with exit code $?"
echo "[$(date)] GPU $GPU: RL on the 7 val scenes, 3 epochs"
# shellcheck disable=SC2086
MASTER_PORT=29604 python rl_training_vlm_nodspy_merged.py $COMMON --rl_on_val --epochs 3 --save-every-epoch --gpu "$GPU" > "$LOGS/rl_vlm_val.log" 2>&1
echo "[$(date)] GPU $GPU: RL val scenes finished with exit code $?"
find "$S" -maxdepth 1 -name "*lmlora*" -newer "$SFT/config.json" -exec chmod -R a+rwX {} + 2>/dev/null
