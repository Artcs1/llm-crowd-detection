#!/bin/bash
# Qwen3-VL-4B-Instruct (--base qwen3-4b) SFT, 3 epochs, p1_visual single, all-linear LoRA, on the benchmark3 train subsets
# K05 / K20 / K60 / K86 (egogroups_city.KSUBSETS: 300 train clips each, id_val held out), one after the other on one GPU,
# without the per-epoch validation pass (--no_eval). Weights are made readable for the other workstation after each run.
# Checkpoints: sft_output/qwen3-vl-4b-groups-merged-nodspy-egogroups-<k>
# Logs: sft_output/logs/ksubsets_qwen3vl4b/sft_<k>.log
# Usage (inside tmux): GPU=0 bash scripts/run_ksubsets_qwen3vl4b_sft_queue.sh   [KS="k05 k20 k60 k86"]
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
GPU=${GPU:-0} KS=${KS:-k05 k20 k60 k86}
S=sft_output
LOGS=$S/logs/ksubsets_qwen3vl4b
mkdir -p "$LOGS"
for k in $KS; do
  echo "[$(date)] GPU $GPU: Qwen3-VL-4B SFT on $k"
  MASTER_PORT=$((29660 + GPU)) python sft_training_vlm_nodspy_merged.py --base qwen3-4b --dataset "egogroups-$k" \
    --variant p1_visual --mode single --no_eval --gpu "$GPU" > "$LOGS/sft_$k.log" 2>&1
  rc=$?
  echo "[$(date)] GPU $GPU: $k SFT exit code $rc"
  find "$S" -maxdepth 1 -name "qwen3-vl-4b-groups-*-egogroups-$k" -exec chmod -R a+rX {} + 2>/dev/null
done
echo "[$(date)] GPU $GPU: K-subset SFT queue done"
