#!/bin/bash
# SFT (3 epochs, script default) then 1-epoch GRPO on JRDB with the 7 JRDB val scenes held out (--jrdb_val_split),
# three pipelines in parallel on GPUs 4-6: LLM single (GPU 4), LLM full (GPU 5), VLM p1_visual (GPU 6).
# Each RL run starts only after its own SFT succeeded. Checkpoints: sft_output/*-jrdbtrain*.
# Waits first for scripts/run_groups_jrdb_val_8models.sh to finish (it uses GPUs 4-7).
# Logs: sft_output/logs/jrdb_val_split/{sft,rl}_<name>.log
# Run inside tmux, e.g.: tmux new -d -s jrdb_train 'bash scripts/run_jrdb_val_split_sft_rl.sh'

umask 000  # files on the share stay writable from the other workstation
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
LOGS=sft_output/logs/jrdb_val_split
mkdir -p "$LOGS"

echo "[$(date)] waiting for the JRDB val eval (sft_output/logs/jrdb_val_8models_chain.log) to finish"
until grep -q "all 8 runs done" sft_output/logs/jrdb_val_8models_chain.log 2>/dev/null; do sleep 60; done

pipeline() {  # <name> <gpu> <sft port> <rl port> <sft script> <rl script> <mode/variant args...>
  local name=$1 gpu=$2 sft_port=$3 rl_port=$4 sft=$5 rl=$6; shift 6
  echo "[$(date)] GPU $gpu: SFT $name"
  if ! MASTER_PORT=$sft_port python "$sft" --dataset jrdb "$@" --jrdb_val_split --gpu "$gpu" > "$LOGS/sft_$name.log" 2>&1; then
    echo "[$(date)] GPU $gpu: SFT $name FAILED, skipping RL (see $LOGS/sft_$name.log)"; return 1
  fi
  echo "[$(date)] GPU $gpu: SFT $name done, RL $name"
  MASTER_PORT=$rl_port python "$rl" --dataset jrdb "$@" --epochs 1 --jrdb_val_split --gpu "$gpu" > "$LOGS/rl_$name.log" 2>&1
  echo "[$(date)] GPU $gpu: RL $name finished with exit code $?"
}

pipeline llm_single 4 29511 29521 sft_training_llm_nodspy_merged.py rl_training_llm_nodspy_merged.py --mode single &
pipeline llm_full   5 29512 29522 sft_training_llm_nodspy_merged.py rl_training_llm_nodspy_merged.py --mode full &
pipeline vlm_p1_visual 6 29513 29523 sft_training_vlm_nodspy_merged.py rl_training_vlm_nodspy_merged.py --variant p1_visual &
wait
echo "[$(date)] all JRDB val-split SFT/RL runs done"
