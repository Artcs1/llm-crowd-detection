#!/bin/bash
# Qwen3-VL-4B-Instruct (--base qwen3-4b) on JRDB, p1_visual single, 20 train scenes (--jrdb_val_split: 7 val scenes held out),
# fixed JRDB loader. Same recipe as the Qwen2.5-VL LM-only chain on GPU 0 (run_jrdb_vlm_{sft,rl}_lmlora.sh), run twice in sequence:
#   A. LM-only LoRA (--lm_lora: vision tower + merger frozen)
#   B. all-linear LoRA (vision + LM)
# each: SFT -> RL on the same 20 train scenes (1 epoch) -> RL on the 7 val scenes (--rl_on_val, 3 epochs, merged ckpt per epoch).
# RL: use_bias_correction_kl off, 8 rollouts x 1 prompt.
# Checkpoints: sft_output/qwen3-vl-4b-groups-merged-nodspy-jrdbtrain[-lmlora]
#              sft_output/qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain[-lmlora]-1ep-nobiaskl-g8p1
#              sft_output/qwen3-vl-4b-groups-grpo-merged-fourlosses-jrdbtrain[-lmlora]-rlval-nobiaskl-g8p1[-epoch{1,2,3}]
# Logs: sft_output/logs/jrdb_qwen3vl4b/{lmlora,alllinear}_{sft,rl_train,rl_val}.log
# Usage (inside tmux): GPU=1 bash scripts/run_jrdb_qwen3vl4b_chain.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_qwen3vl4b
mkdir -p "$LOGS"
BASE="--base qwen3-4b --dataset jrdb --variant p1_visual --mode single --jrdb_val_split"
RLARGS="--no_bias_kl --num_generations 8 --prompts_per_step 1"

# name | lora flag | checkpoint suffix | port offset
for job in "lmlora|--lm_lora|-lmlora|0" "alllinear|||5"; do
  IFS='|' read -r name lflag suf off <<< "$job"
  SFT=$S/qwen3-vl-4b-groups-merged-nodspy-jrdbtrain$suf
  echo "[$(date)] GPU $GPU: Qwen3-VL-4B $name SFT"
  # shellcheck disable=SC2086
  MASTER_PORT=$((29611+off)) python sft_training_vlm_nodspy_merged.py $BASE $lflag --gpu "$GPU" > "$LOGS/${name}_sft.log" 2>&1
  echo "[$(date)] GPU $GPU: $name SFT exit code $?"
  if [ ! -f "$SFT/config.json" ]; then echo "[$(date)] GPU $GPU: no $SFT, skipping $name RL"; continue; fi
  echo "[$(date)] GPU $GPU: $name RL on the 20 train scenes, 1 epoch"
  # shellcheck disable=SC2086
  MASTER_PORT=$((29612+off)) python rl_training_vlm_nodspy_merged.py $BASE $lflag $RLARGS --epochs 1 --gpu "$GPU" > "$LOGS/${name}_rl_train.log" 2>&1
  echo "[$(date)] GPU $GPU: $name RL train exit code $?"
  echo "[$(date)] GPU $GPU: $name RL on the 7 val scenes, 3 epochs"
  # shellcheck disable=SC2086
  MASTER_PORT=$((29613+off)) python rl_training_vlm_nodspy_merged.py $BASE $lflag $RLARGS --rl_on_val --epochs 3 --save-every-epoch --gpu "$GPU" > "$LOGS/${name}_rl_val.log" 2>&1
  echo "[$(date)] GPU $GPU: $name RL val exit code $?"
  find "$S" -maxdepth 1 -name "qwen3-vl-4b-*" -exec chmod -R a+rwX {} + 2>/dev/null
done
echo "[$(date)] GPU $GPU: Qwen3-VL-4B chain done"
