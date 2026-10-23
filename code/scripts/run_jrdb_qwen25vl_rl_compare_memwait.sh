#!/bin/bash
# The four RL runs of scripts/run_jrdb_qwen25vl_lora_compare_chain.sh (Qwen2.5-VL-7B, JRDB 20 train scenes, fixed loader,
# bias KL off, 8 x 1), on GPU $GPU only, each started only when the GPU has >= $NEED_MIB free: GRPO's colocated vLLM
# refuses to start unless 55% of the GPU is free, and the GPU is shared. A run that still dies on that startup check is
# retried after the next wait; any other failure is reported and the queue moves on.
#   1. LM-only RL train, 1 ep   2. LM-only RL val, 3 ep   3. all-linear RL train, 1 ep   4. all-linear RL val, 3 ep
# Logs: sft_output/logs/jrdb_qwen25vl_lora_compare/<name>.log (a failed startup attempt is kept as <name>_nomem_<time>.log)
# Usage (inside tmux): GPU=1 bash scripts/run_jrdb_qwen25vl_rl_compare_memwait.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1} NEED_MIB=${NEED_MIB:-66000}  # what this RL used when it ran alone on a GPU (~65 GB)
S=sft_output
LOGS=$S/logs/jrdb_qwen25vl_lora_compare
mkdir -p "$LOGS"
BASE="--dataset jrdb --variant p1_visual --mode single --jrdb_val_split"
RLARGS="--no_bias_kl --num_generations 8 --prompts_per_step 1"
FIX="--tag=-fixedloader"  # with "=": argparse would read a separate "-fixedloader" as an option

free_mib() { nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
run() {  # name port args...
  local name=$1 port=$2; shift 2
  while true; do
    local waited=0
    until [ "$(free_mib)" -ge "$NEED_MIB" ]; do
      [ $waited = 0 ] && echo "[$(date)] GPU $GPU: $name waiting for ${NEED_MIB} MiB free (now $(free_mib))"
      waited=1; sleep 60
    done
    echo "[$(date)] GPU $GPU: $name starting ($(free_mib) MiB free)"
    MASTER_PORT=$port python rl_training_vlm_nodspy_merged.py "$@" --gpu "$GPU" > "$LOGS/$name.log" 2>&1
    local rc=$?  # before the echo: its $(date) would reset $?
    echo "[$(date)] GPU $GPU: $name exit code $rc"
    if [ $rc -ne 0 ] && grep -q "Free memory on device" "$LOGS/$name.log"; then
      mv "$LOGS/$name.log" "$LOGS/${name}_nomem_$(date +%m%d_%H%M).log"
      echo "[$(date)] GPU $GPU: $name hit vLLM's free-memory check, retrying after the next wait"
      sleep 120; continue
    fi
    return $rc
  done
}

# shellcheck disable=SC2086
run lmlora_rl_train 29622 $BASE --lm_lora $RLARGS --epochs 1
# shellcheck disable=SC2086
run lmlora_rl_val 29623 $BASE --lm_lora $RLARGS --rl_on_val --epochs 3 --save-every-epoch
# shellcheck disable=SC2086
run alllinear_rl_train 29624 $BASE $FIX $RLARGS --epochs 1
# shellcheck disable=SC2086
run alllinear_rl_val 29625 $BASE $FIX $RLARGS --rl_on_val --epochs 3 --save-every-epoch
find "$S" -maxdepth 1 -name "qwen2.5-vl-7b-*" \( -name "*fixedloader*" -o -name "*lmlora*" \) -exec chmod -R a+rwX {} + 2>/dev/null
echo "[$(date)] GPU $GPU: Qwen2.5-VL RL comparison runs done"
