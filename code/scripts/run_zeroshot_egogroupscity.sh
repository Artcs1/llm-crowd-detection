#!/bin/bash
# Zero-shot VLM grouping on EgoGroupsCity_Test (egogroups-city test split, scripts/make_egogroupscity_folders.py):
# one vLLM server on one GPU, NPARTS batch_fetch_groups.py clients in parallel pulling from a shared queue (--claim_dir:
# each takes the next keyframe nobody has claimed, so a few very long generations don't leave other clients idle;
# --skip_existing so it resumes), single setting (target frame 10), then detection file + grouping pkl + AP / F1 vs the
# city test GT. If a server already answers on PORT it is reused and left running.
#   bash scripts/run_zeroshot_egogroupscity.sh [gpu=1] [model=Qwen/Qwen3-VL-4B-Instruct] [prompt=p1_visual] [port=8041] [nparts=32] [setting=single]
# setting full: the target frame + the video of frames 1..10 (11 images per request, see utils.full_inference) and each
# person's movement_direction; the server then gets a 64k context and up to 16 images per prompt.
# Trained checkpoints: MODEL = the local checkpoint dir and NAME = the served model name (must contain one '/', e.g.
# Qwen/Qwen3-VL-4B-Instruct-SFT-JRDBTrain); DATA=EgoGroupsCity_Test_jrdbstyle for JRDB-trained models (JRDB-frame 3D,
# scripts/make_egogroups_jrdbstyle.py EgoGroupsCity_Test). Both are env vars, default: zero-shot on EgoGroupsCity_Test.
# GT = the ground truth to score against (out/gt_group_10fps_<GT>.{txt,pkl}), default city_test; e.g.
# DATA=EgoGroupsOODCity_Test GT=benchmark3_id_test, DATA=EgoGroupsOODRegions_Test GT=benchmark3_ood_test (README_benchmark3.md).
# Logs: sft_output/logs/egogroupscity_zeroshot/<model>_<prompt>_*.log; metrics: ..._metrics.txt
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0  # no nvcc / CUDA toolkit here for FlashInfer's JIT sampler
GPU=${1:-1} MODEL=${2:-Qwen/Qwen3-VL-4B-Instruct} PROMPT=${3:-p1_visual} PORT=${4:-8041} NPARTS=${5:-32} SETTING=${6:-single}
RES=$([ "$SETTING" = full ] && echo results_full || echo results)
MAXLEN=$([ "$SETTING" = full ] && echo 65536 || echo 32768)
NIMG=$([ "$SETTING" = full ] && echo 16 || echo 1)
NAME=${NAME:-$MODEL}
SHORT=${NAME#*/}
DATA=${DATA:-EgoGroupsCity_Test}
GT=${GT:-city_test}
TOTAL=$(ls "../../$DATA/jsons_step6" | wc -l)
LOGS=sft_output/logs/egogroupscity_zeroshot
mkdir -p "$LOGS"
L=$LOGS/${SHORT}_${PROMPT}$([ "$SETTING" = full ] && echo _full)$([ "$DATA" != EgoGroupsCity_Test ] && echo "_$DATA")

pid=
if curl -sf -H "Authorization: Bearer egocity" "http://localhost:$PORT/v1/models" > /dev/null; then
  echo "[$(date)] reusing the server already on port $PORT (left running at the end)"
else
  echo "[$(date)] GPU $GPU: serving $MODEL on port $PORT"
  CUDA_VISIBLE_DEVICES=$GPU vllm serve "$MODEL" --served-model-name "$NAME" --api-key egocity --port "$PORT" --gpu-memory-utilization ${GPU_UTIL:-0.90} \
    --max-model-len $MAXLEN --max-num-seqs 64 --limit-mm-per-prompt "{\"image\": $NIMG}" > "${L}_server.log" 2>&1 &
  pid=$!
  until curl -sf -H "Authorization: Bearer egocity" "http://localhost:$PORT/v1/models" > /dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] server died, see ${L}_server.log"; exit 1; fi
    sleep 10
  done
fi
CLAIMS=${L}_claims_$(date +%s)
echo "[$(date)] server up; $NPARTS clients on $DATA ($PROMPT, $SETTING, frame 10), shared queue $CLAIMS"
clients=()
for k in $(seq 0 $((NPARTS - 1))); do
  python3 batch_fetch_groups.py "../../$DATA/jsons_step6" "$SETTING" vlm_image "$NAME" 10 --depth_method detany_3D \
    --prompt_method "$PROMPT" --api_base "http://localhost:$PORT/v1" --api_key egocity --max_tokens 24000 \
    --frame_path "../../$DATA/videos_frames" --save_image --skip_existing --claim_dir "$CLAIMS" \
    > "${L}_queue$k.log" 2>&1 &
  clients+=($!)
done
wait "${clients[@]}"
[ -n "$pid" ] && { kill $pid; wait $pid 2>/dev/null; }
N=$(find "../results/predictions/$DATA/$RES/$SHORT/vlm_image/detany_3D/$PROMPT/10" -name "*.json" | wc -l)
echo "[$(date)] inference done: $N / $TOTAL predictions"

python3 compute_detections.py --dataset $DATA --mode "$SETTING" --depth_method detany_3D --prompt_method "$PROMPT" \
  --model "$SHORT" --vlm_mode vlm_image --frame_id 10 > "${L}_det.log" 2>&1
DET=../results/detection_files/${DATA}_10/${RES}_${SHORT}_vlm_image_detany_3D_${PROMPT}.txt
PKL=../results/grouping_files/${DATA}_10/${RES}_${SHORT}_vlm_image_detany_3D_${PROMPT}.pkl
mkdir -p "$(dirname "$PKL")"
python3 F1_evaluator/det_txt_to_pkl.py "$DET" "$PKL" --split test >> "${L}_det.log" 2>&1
{
  echo "== $SHORT $PROMPT $SETTING on $DATA ($N / $TOTAL sub-clips predicted)"
  echo "AP, G1..G5: $(cd AP_evaluator && python3 JRDB_eval.py -g out/gt_group_10fps_$GT.txt -d "../$DET" -t task_3 \
      --labelmap ./label_map/task_3.pbtxt --output all 2>/dev/null | grep -E '^[0-9.]+$' | tr '\n' ' ')"
  echo "F1 (P, R, F1): $(cd F1_evaluator && python3 ours_jrdb_evaluation.py --det "../$PKL" \
      --gt out/gt_group_10fps_$GT.pkl --o all 2>&1 | grep -v -i warn | tail -3 | tr '\n' ' ')"
} > "${L}_metrics.txt"
cat "${L}_metrics.txt"; echo "[$(date)] done"
