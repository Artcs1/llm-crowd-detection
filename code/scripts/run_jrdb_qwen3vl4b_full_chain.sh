#!/bin/bash
# Qwen3-VL-4B-Instruct (--base qwen3-4b) on JRDB, p1_visual FULL mode (annotated target frame + each person's
# movement_direction from the earlier frames), 20 train scenes (--jrdb_val_split: 7 val scenes held out), all-linear LoRA:
# SFT -> RL on the same 20 train scenes, 1 epoch (use_bias_correction_kl off, 8 rollouts x 1 prompt, as
# run_jrdb_qwen3vl4b_chain.sh). Waits for the EgoGroupsCity zero-shot run on the same GPU to finish and stops its vLLM
# server (WAIT_SERVER_PID) first.
# Checkpoints: sft_output/qwen3-vl-4b-groups-merged-nodspy-full-jrdbtrain
#              sft_output/qwen3-vl-4b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-nobiaskl-g8p1
# Logs: sft_output/logs/jrdb_qwen3vl4b_full/{sft,rl_train}.log
#   GPU=1 WAIT_SERVER_PID=<pid> bash scripts/run_jrdb_qwen3vl4b_full_chain.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_qwen3vl4b_full
mkdir -p "$LOGS"
BASE="--base qwen3-4b --dataset jrdb --variant p1_visual --mode full --jrdb_val_split"
RLARGS="--no_bias_kl --num_generations 8 --prompts_per_step 1"
SFT=$S/qwen3-vl-4b-groups-merged-nodspy-full-jrdbtrain

if [ -n "$WAIT_SERVER_PID" ]; then
  echo "[$(date)] waiting for the EgoGroupsCity zero-shot run to finish (scoring done)"
  until tail -3 $S/logs/egogroupscity_zeroshot_driver.log | grep -q "\] done"; do sleep 60; done
  if ps -p "$WAIT_SERVER_PID" -o args= | grep -q "vllm serve"; then
    kill "$WAIT_SERVER_PID"; while kill -0 "$WAIT_SERVER_PID" 2>/dev/null; do sleep 5; done
    echo "[$(date)] stopped the zero-shot vLLM server $WAIT_SERVER_PID"
  fi
  sleep 30
fi
echo "[$(date)] GPU $GPU: Qwen3-VL-4B full SFT"
# shellcheck disable=SC2086
MASTER_PORT=29641 python sft_training_vlm_nodspy_merged.py $BASE --gpu "$GPU" > "$LOGS/sft.log" 2>&1
echo "[$(date)] GPU $GPU: SFT exit code $?"
if [ ! -f "$SFT/config.json" ]; then echo "[$(date)] GPU $GPU: no $SFT, skipping RL"; exit 1; fi
echo "[$(date)] GPU $GPU: RL on the 20 train scenes, 1 epoch"
# shellcheck disable=SC2086
MASTER_PORT=29642 python rl_training_vlm_nodspy_merged.py $BASE $RLARGS --epochs 1 --gpu "$GPU" > "$LOGS/rl_train.log" 2>&1
echo "[$(date)] GPU $GPU: RL exit code $?"
find "$S" -maxdepth 1 -name "qwen3-vl-4b-*full*" -exec chmod -R a+rwX {} + 2>/dev/null
echo "[$(date)] GPU $GPU: Qwen3-VL-4B full chain done"
