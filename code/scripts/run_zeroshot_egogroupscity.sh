#!/bin/bash
# Zero-shot VLM grouping on EgoGroupsCity_Test (egogroups-city test split, scripts/make_egogroupscity_folders.py):
# one vLLM server on one GPU, NPARTS batch_fetch_groups.py clients in parallel (--num_parts/--part_id, --skip_existing so
# it resumes), single setting (target frame 10), then detection file + grouping pkl + AP / F1 vs the city test GT.
#   bash scripts/run_zeroshot_egogroupscity.sh [gpu=1] [model=Qwen/Qwen3-VL-4B-Instruct] [prompt=p1_visual] [port=8041] [nparts=32]
# Logs: sft_output/logs/egogroupscity_zeroshot/<model>_<prompt>_*.log; metrics: ..._metrics.txt
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no nvcc / CUDA toolkit here for FlashInfer's JIT sampler
GPU=${1:-1} MODEL=${2:-Qwen/Qwen3-VL-4B-Instruct} PROMPT=${3:-p1_visual} PORT=${4:-8041} NPARTS=${5:-32}
SHORT=${MODEL#*/}
DATA=EgoGroupsCity_Test
LOGS=sft_output/logs/egogroupscity_zeroshot
mkdir -p "$LOGS"
L=$LOGS/${SHORT}_${PROMPT}

echo "[$(date)] GPU $GPU: serving $MODEL on port $PORT"
CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" --api-key egocity --port "$PORT" --gpu-memory-utilization 0.90 \
  --max-model-len 32768 --max-num-seqs 64 --limit-mm-per-prompt '{"image": 1}' > "${L}_server.log" 2>&1 &
pid=$!
until curl -sf -H "Authorization: Bearer egocity" "http://localhost:$PORT/v1/models" > /dev/null; do
  if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] server died, see ${L}_server.log"; exit 1; fi
  sleep 10
done
echo "[$(date)] server up; $NPARTS clients on $DATA ($PROMPT, single, frame 10)"
for k in $(seq 0 $((NPARTS - 1))); do
  python3 batch_fetch_groups.py "../../$DATA/jsons_step6" single vlm_image "$MODEL" 10 --depth_method detany_3D \
    --prompt_method "$PROMPT" --api_base "http://localhost:$PORT/v1" --api_key egocity --max_tokens 24000 \
    --frame_path "../../$DATA/videos_frames" --save_image --skip_existing --num_parts "$NPARTS" --part_id $k \
    > "${L}_part$k.log" 2>&1 &
done
wait $(jobs -p | grep -v "^$pid$")
kill $pid; wait $pid 2>/dev/null
N=$(find "../results/predictions/$DATA/results/$SHORT/vlm_image/detany_3D/$PROMPT/10" -name "*.json" | wc -l)
echo "[$(date)] inference done: $N / 4500 predictions; server stopped"

python3 compute_detections.py --dataset $DATA --mode single --depth_method detany_3D --prompt_method "$PROMPT" \
  --model "$SHORT" --vlm_mode vlm_image --frame_id 10 > "${L}_det.log" 2>&1
DET=../results/detection_files/${DATA}_10/results_${SHORT}_vlm_image_detany_3D_${PROMPT}.txt
PKL=../results/grouping_files/${DATA}_10/results_${SHORT}_vlm_image_detany_3D_${PROMPT}.pkl
mkdir -p "$(dirname "$PKL")"
python3 F1_evaluator/det_txt_to_pkl.py "$DET" "$PKL" --split test >> "${L}_det.log" 2>&1
{
  echo "== $SHORT $PROMPT zero-shot on $DATA ($N / 4500 sub-clips predicted)"
  echo "AP, G1..G5: $(cd AP_evaluator && python3 JRDB_eval.py -g out/gt_group_10fps_city_test.txt -d "../$DET" -t task_3 \
      --labelmap ./label_map/task_3.pbtxt --output all 2>/dev/null | grep -E '^[0-9.]+$' | tr '\n' ' ')"
  echo "F1 (P, R, F1): $(cd F1_evaluator && python3 ours_jrdb_evaluation.py --det "../$PKL" \
      --gt out/gt_group_10fps_city_test.pkl --o all 2>&1 | grep -v -i warn | tail -3 | tr '\n' ' ')"
} > "${L}_metrics.txt"
echo "[$(date)] done"; cat "${L}_metrics.txt"
