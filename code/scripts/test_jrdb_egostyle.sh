#!/bin/bash
# JRDB test (1,829 keyframes) of an EgoGroups-trained Qwen VLM checkpoint on JRDB_fixed_gold_egostyle (JRDB with
# EgoGroups-style 3D, scripts/make_jrdb_egostyle.py), p1_visual single (target frame 15): one vLLM server, NPARTS clients
# on a shared queue (resumes with --skip_existing), then AP (JRDB_eval.py task_3) + MINGLE F1 (mingle_det.py) vs
# AP_evaluator/out/gt_group.txt and our F1 half / exact with the G1..G5+ split (ours_jrdb_evaluation.py vs F1_evaluator/out/gt.pkl).
# Needs ../../JRDB_fixed_gold_egostyle/{jsons_gold,videos_frames} next to the repo (videos_frames may be the link
# ../JRDB_fixed_gold/videos_frames; single mode only reads 00015.jpeg). README_jrdb_egostyle_test.md.
#   GPU=0 PORT=8401 MODEL=sft_output/qwen3-vl-4b-groups-merged-nodspy-egogroups-benchmark3 \
#     NAME=Qwen/Qwen3-VL-4B-Instruct-SFT-Benchmark3Train bash scripts/test_jrdb_egostyle.sh
# Metrics: sft_output/logs/jrdb_egostyle_test/<short name>_metrics.txt
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ${ENV:-py10-dspy}
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0
GPU=${GPU:-0} PORT=${PORT:-8401} NPARTS=${NPARTS:-32} PROMPT=p1_visual
MODEL=${MODEL:?set MODEL to the merged checkpoint dir}
NAME=${NAME:?set NAME to a served name with one "/", e.g. Qwen/Qwen3-VL-4B-Instruct-SFT-Benchmark3}
SHORT=${NAME#*/}
DATA=JRDB_fixed_gold_egostyle
L=sft_output/logs/jrdb_egostyle_test/$SHORT
mkdir -p "$(dirname "$L")"
[ "$(ls ../../$DATA/jsons_gold | wc -l)" = 1829 ] || { echo "../../$DATA/jsons_gold must hold 1829 files"; exit 1; }

CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" --served-model-name "$NAME" --api-key egotest --port "$PORT" \
  --gpu-memory-utilization ${GPU_UTIL:-0.85} --max-model-len 32768 --max-num-seqs 64 --limit-mm-per-prompt '{"image": 1}' \
  > "${L}_server.log" 2>&1 &
pid=$!
until curl -sf -H "Authorization: Bearer egotest" "http://localhost:$PORT/v1/models" > /dev/null; do
  kill -0 $pid 2>/dev/null || { echo "[$(date)] server died, see ${L}_server.log"; exit 1; }
  sleep 10
done
CLAIMS=${L}_claims_$(date +%s)
echo "[$(date)] server up on GPU $GPU; $NPARTS clients on $DATA"
clients=()
for k in $(seq 0 $((NPARTS - 1))); do
  python3 batch_fetch_groups.py "../../$DATA/jsons_gold" single vlm_image "$NAME" 15 --depth_method detany_3D \
    --prompt_method $PROMPT --api_base "http://localhost:$PORT/v1" --api_key egotest --max_tokens 24000 \
    --frame_path "../../$DATA/videos_frames" --skip_existing --claim_dir "$CLAIMS" > "${L}_queue$k.log" 2>&1 &
  clients+=($!)
done
wait "${clients[@]}"
kill $pid; wait $pid 2>/dev/null
PRED=../results/predictions/$DATA/results/$SHORT/vlm_image/detany_3D/$PROMPT/15
N=$(find "$PRED" -name "*.json" | wc -l)
echo "[$(date)] inference done: $N / 1829 keyframes (rerun the same command to fill in missing ones)"

python3 compute_detections.py --dataset $DATA --mode single --depth_method detany_3D --prompt_method $PROMPT \
  --model "$SHORT" --vlm_mode vlm_image --frame_id 15 > "${L}_det.log" 2>&1
DET=../results/detection_files/${DATA}_15/results_${SHORT}_vlm_image_detany_3D_${PROMPT}.txt
PKL=../results/grouping_files/${DATA}_15/results_${SHORT}_vlm_image_detany_3D_${PROMPT}.pkl
mkdir -p "$(dirname "$PKL")"
python3 scripts/jrdb_groupings.py "$PRED" "$PKL" >> "${L}_det.log" 2>&1
export CUDA_VISIBLE_DEVICES=""
{
  echo "== $SHORT $PROMPT single on JRDB test, EgoGroups-style 3D ($N / 1829 keyframes predicted)"
  echo "AP, G1..G5: $(cd AP_evaluator && python3 JRDB_eval.py -g out/gt_group.txt -d "../$DET" -t task_3 \
      --labelmap ./label_map/task_3.pbtxt --output all 2>/dev/null | grep -E '^[0-9.]+$' | tr '\n' ' ')"
  echo "MINGLE (name mIoU P R F1 n_pred n_gt): $(cd AP_evaluator && python3 mingle_det.py -g out/gt_group.txt -d "../$DET" 2>&1 | grep -vi warn | tail -1)"
  for crit in half exact; do
    echo "F1 $crit (P / R / F1 per group size; Group = overall, singletons excluded):"
    (cd F1_evaluator && python3 ours_jrdb_evaluation.py --det "../$PKL" --gt out/gt.pkl --o all --crit $crit --by_size 2>&1 |
        grep -E "^(G[1-5]|Group|All)")
  done
} > "${L}_metrics.txt"
cat "${L}_metrics.txt"
