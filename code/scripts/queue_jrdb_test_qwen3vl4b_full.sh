#!/bin/bash
# JRDB test (JRDB_fixed_gold, 1,829 keyframes) of the full-mode Qwen3-VL-4B JRDB checkpoints (scripts/run_jrdb_qwen3vl4b_full_chain.sh):
# SFT then RL, p1_visual full (target frame + video of frames 1..15), one vLLM server on $GPU, then AP (JRDB_eval.py task_3)
# + MINGLE F1 (mingle_det.py) vs AP_evaluator/out/gt_group.txt. Resumes with --skip_existing.
# Predictions: ../results/predictions/JRDB_fixed_gold/results_full/Qwen3-VL-4B-Instruct-{SFT,RL}-JRDBTrain/
# ONLY=SFT or ONLY=RL runs just that model (e.g. one per GPU).
# Run inside tmux: tmux new -d -s jrdb_full_gpu2 'GPU=2 bash scripts/queue_jrdb_test_qwen3vl4b_full.sh'
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
S=sft_output
GPU=${GPU:-2} PORT=${PORT:-8121}
JOBS=(
  "$S/qwen3-vl-4b-groups-merged-nodspy-full-jrdbtrain|Qwen/Qwen3-VL-4B-Instruct-SFT-JRDBTrain"
  "$S/qwen3-vl-4b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-nobiaskl-g8p1|Qwen/Qwen3-VL-4B-Instruct-RL-JRDBTrain"
)
for job in "${JOBS[@]}"; do
  IFS='|' read -r ckpt name <<< "$job"
  [ -n "$ONLY" ] && [[ "$name" != *"-$ONLY-"* ]] && continue
  short=${name#*/} L=$S/logs/jrdb_test_qwen3vl4b/${name#*/}_full
  DATASET=jrdb MAXLEN=65536 GPUS="$GPU" UTILS=0.85 PORT0=$PORT PER=32 \
    bash scripts/run_groups_fastgpus_v3.sh "$ckpt" "$name" p1_visual full "$L"
  python3 compute_detections.py --dataset JRDB_fixed_gold --mode full --depth_method detany_3D --prompt_method p1_visual \
    --model "$short" --vlm_mode vlm_image --frame_id 15 > "$L/det.log" 2>&1
  det=../../results/detection_files/JRDB_fixed_gold_15/results_full_${short}_vlm_image_detany_3D_p1_visual.txt
  {
    echo "== $short p1_visual full on JRDB test ($(find ../results/predictions/JRDB_fixed_gold/results_full/$short -name '*.json' | wc -l) / 1829)"
    echo "AP, G1..G5: $(cd AP_evaluator && python3 JRDB_eval.py -g out/gt_group.txt -d $det -t task_3 \
        --labelmap ./label_map/task_3.pbtxt --output all 2>/dev/null | grep -E '^[0-9.]+$' | tr '\n' ' ')"
    echo "MINGLE F1:"; (cd AP_evaluator && python3 mingle_det.py -g out/gt_group.txt -d $det 2>&1 | grep -v -i warn | tail -2)
  } > "$L/metrics.txt"
  cat "$L/metrics.txt"
  PORT=$((PORT + 1))
done
echo "[$(date)] done"
