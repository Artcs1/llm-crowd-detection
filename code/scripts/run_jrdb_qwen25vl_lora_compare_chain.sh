#!/bin/bash
# Qwen2.5-VL-7B on JRDB (p1_visual single, 20 train scenes, --jrdb_val_split), all with the fixed JRDB loader, so all-linear and
# LM-only LoRA compare cleanly (the older all-linear checkpoints used the old shard matching). On GPU $GPU in sequence, after
# pid $WAIT_PID (the Qwen3-VL-4B JRDB test queue) exits:
#   1. all-linear SFT          -> sft_output/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain-fixedloader
#   2. LM-only RL train, 1 ep  -> ...-grpo-merged-fourlosses-jrdbtrain-lmlora-1ep-nobiaskl-g8p1 (from the Oct 19 LM-only SFT)
#   3. LM-only RL val, 3 ep    -> ...-jrdbtrain-lmlora-rlval-nobiaskl-g8p1[-epoch{1,2,3}]
#   4. all-linear RL train     -> ...-jrdbtrain-fixedloader-1ep-nobiaskl-g8p1
#   5. all-linear RL val, 3 ep -> ...-jrdbtrain-fixedloader-rlval-nobiaskl-g8p1[-epoch{1,2,3}]
# RL: use_bias_correction_kl off, 8 rollouts x 1 prompt. Logs: sft_output/logs/jrdb_qwen25vl_lora_compare/
# Usage (inside tmux): WAIT_PID=<pid> GPU=1 bash scripts/run_jrdb_qwen25vl_lora_compare_chain.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
export VLLM_USE_FLASHINFER_SAMPLER=0  # GRPO's colocated vLLM; FlashInfer JIT-compiles its sampler and there is no nvcc here
GPU=${GPU:-1}
S=sft_output
LOGS=$S/logs/jrdb_qwen25vl_lora_compare
mkdir -p "$LOGS"
if [ -n "$WAIT_PID" ]; then
  echo "[$(date)] GPU $GPU: waiting for pid $WAIT_PID (Qwen3-VL-4B JRDB tests) to finish"
  # kill -0 also succeeds on a zombie (exited but not reaped), so check the process state too
  while kill -0 "$WAIT_PID" 2>/dev/null && [ "$(ps -o stat= -p "$WAIT_PID" | cut -c1)" != Z ]; do sleep 30; done
fi
BASE="--dataset jrdb --variant p1_visual --mode single --jrdb_val_split"
RLARGS="--no_bias_kl --num_generations 8 --prompts_per_step 1"
FIX="--tag=-fixedloader"  # with "=": argparse would read a separate "-fixedloader" as an option

run() {  # name port script args...
  local name=$1 port=$2 script=$3; shift 3
  echo "[$(date)] GPU $GPU: $name"
  MASTER_PORT=$port python "$script" "$@" --gpu "$GPU" > "$LOGS/$name.log" 2>&1
  local rc=$?  # before the echo: its $(date) would reset $?
  echo "[$(date)] GPU $GPU: $name exit code $rc"
}
need() { [ -f "$1/config.json" ] || { echo "[$(date)] GPU $GPU: no $1, skipping $2"; return 1; }; }

# shellcheck disable=SC2086
run alllinear_sft 29621 sft_training_vlm_nodspy_merged.py $BASE $FIX
if need $S/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain-lmlora "LM-only RL"; then
  # shellcheck disable=SC2086
  run lmlora_rl_train 29622 rl_training_vlm_nodspy_merged.py $BASE --lm_lora $RLARGS --epochs 1
  # shellcheck disable=SC2086
  run lmlora_rl_val 29623 rl_training_vlm_nodspy_merged.py $BASE --lm_lora $RLARGS --rl_on_val --epochs 3 --save-every-epoch
fi
if need $S/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain-fixedloader "all-linear RL"; then
  # shellcheck disable=SC2086
  run alllinear_rl_train 29624 rl_training_vlm_nodspy_merged.py $BASE $FIX $RLARGS --epochs 1
  # shellcheck disable=SC2086
  run alllinear_rl_val 29625 rl_training_vlm_nodspy_merged.py $BASE $FIX $RLARGS --rl_on_val --epochs 3 --save-every-epoch
fi
find "$S" -maxdepth 1 -name "qwen2.5-vl-7b-*" \( -name "*fixedloader*" -o -name "*lmlora*" \) -exec chmod -R a+rwX {} + 2>/dev/null
echo "[$(date)] GPU $GPU: Qwen2.5-VL LoRA comparison chain done"
