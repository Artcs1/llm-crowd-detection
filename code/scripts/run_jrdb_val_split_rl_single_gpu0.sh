#!/bin/bash
# Rerun of the LLM single 1-epoch JRDB val-split GRPO (the GPU 1 run was killed by someone else at step 1030/1385, no
# checkpoint), queued on GPU 0 after scripts/run_jrdb_val_split_rl.sh (VLM, then LLM full) is done.
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
CHAIN=sft_output/logs/jrdb_val_split_rl_chain.log
echo "[$(date)] GPU 0: waiting for the VLM / LLM full RL chain ($CHAIN)"
until grep -q "all JRDB val-split RL runs done" $CHAIN; do sleep 60; done
echo "[$(date)] GPU 0: RL llm_single"
MASTER_PORT=29531 python rl_training_llm_nodspy_merged.py --dataset jrdb --mode single --epochs 1 --jrdb_val_split --gpu 0 \
  > sft_output/logs/jrdb_val_split_rl/rl_llm_single.log 2>&1
echo "[$(date)] GPU 0: RL llm_single finished with exit code $?"
