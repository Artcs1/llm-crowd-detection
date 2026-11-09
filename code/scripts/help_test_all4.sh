#!/bin/bash
# Speed up the all-four video test running on GPU 1 (scripts/test_all4_video.sh, tmux test_all4_gpu1).
# For each GPU in HELP_GPUS: wait until it has no compute process (its own ablation test finished), serve the same
# checkpoint under the same served name on port 8370+G, and add 24 clients per split on the SAME claim queues as the
# GPU 1 run (claims are O_EXCL files, so no keyframe is done twice).
# Finalizer: GPU 1's clients can exit while a helper is still on its last keyframes, so once the GPU 1 run AND all
# helpers are done, it rebuilds both detection files and replaces the all-four lines in the scores / density TSVs.
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd /mnt/paola-shared/jmurrugarral/llm-crowd-detection/code
export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
HELP_GPUS=${HELP_GPUS:-0 2 3}
M=sft_output/qwen3-vl-4b-groups-grpo-merged-fourlosses-full-social-egogroups-benchmark3-1ep-nobiaskl-g8p1
N=Qwen/Qwen3-VL-4B-Instruct-RL-B3-Social-RW-v-bce-f1-ari-Full SHORT=${N#*/}
ZL=sft_output/logs/egogroupscity_zeroshot LOGS=sft_output/logs/reward_ablation RES=../results
F=results_full_${SHORT}_vlm_image_detany_3D_p1_social.txt
SPLITS="EgoGroupsOODCity_Test:benchmark3_id_test EgoGroupsOODRegions_Test:benchmark3_ood_test"
note() { echo "[$(date)] $*"; }
claims() { ls -d ${ZL}/${SHORT}_p1_social_full_$1_claims_* | tail -1; }
left() {  # keyframes of split $1 not claimed yet (total - claimed)
  echo $(( $(ls ../../$1/jsons_step6 | wc -l) - $(ls $(claims $1) | wc -l) )); }

helper() {
  local G=$1 PORT=$((8370 + $1)) L=$LOGS/help_all4_gpu$1
  until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i $G)" ]; do
    tmux has-session -t '=test_all4_gpu1' 2>/dev/null || { note "GPU $G: GPU 1 test already finished, not helping"; return; }
    sleep 30; done
  local l1=$(left EgoGroupsOODCity_Test) l2=$(left EgoGroupsOODRegions_Test)
  [ $((l1 + l2)) -lt 50 ] && { note "GPU $G: free, but only $l1 + $l2 keyframes unclaimed, not helping"; return; }
  note "GPU $G: free, helping ($l1 Cities + $l2 Regions keyframes unclaimed); server on port $PORT"
  CUDA_VISIBLE_DEVICES=$G vllm serve "$M" --served-model-name "$N" --api-key egocity --port $PORT --gpu-memory-utilization 0.90 \
    --max-model-len 65536 --max-num-seqs 128 --limit-mm-per-prompt '{"image": 16}' > ${L}_server.log 2>&1 &
  local pid=$!
  until curl -sf -H "Authorization: Bearer egocity" "http://localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $pid 2>/dev/null || { note "GPU $G: server died, see ${L}_server.log"; return; }; sleep 10; done
  local cl=()
  for t in $SPLITS; do
    local DATA=${t%%:*}
    for k in $(seq 0 23); do
      python3 batch_fetch_groups.py "../../$DATA/jsons_step6" full vlm_image "$N" 10 --depth_method detany_3D \
        --prompt_method p1_social --api_base "http://localhost:$PORT/v1" --api_key egocity --max_tokens 24000 \
        --frame_path "../../$DATA/videos_frames" --save_image --skip_existing --claim_dir "$(claims $DATA)" \
        > ${L}_${DATA}_client$k.log 2>&1 &
      cl+=($!)
    done
  done
  wait "${cl[@]}"
  kill $pid; wait $pid 2>/dev/null
  note "GPU $G: helper clients done, server stopped"
}

hp=()
for G in $HELP_GPUS; do helper $G & hp+=($!); done
wait "${hp[@]}"
while tmux has-session -t '=test_all4_gpu1' 2>/dev/null; do sleep 30; done
note "GPU 1 test and all helpers done; rebuilding detections and scores for $SHORT"
for t in $SPLITS; do
  DATA=${t%%:*} GT=${t#*:}
  n=$(find ../results/predictions/$DATA/results_full/$SHORT/vlm_image/detany_3D/p1_social/10 -name "*.json" | wc -l)
  python3 compute_detections.py --dataset $DATA --mode full --depth_method detany_3D --prompt_method p1_social \
    --model "$SHORT" --vlm_mode vlm_image --frame_id 10 > $LOGS/help_all4_${DATA}_det.log 2>&1
  grep -vF "$(printf '%s\t%s' $DATA $F)" $RES/reward_ablation_scores.tsv > $RES/reward_ablation_scores.tsv.tmp
  echo -e "$DATA\t$F\t$(cd AP_evaluator && python3 mingle_det.py -g out/gt_group_10fps_$GT.txt -d ../$RES/detection_files/${DATA}_10/$F 2>&1 | grep -vi warn | tail -1 | cut -f2-)" >> $RES/reward_ablation_scores.tsv.tmp
  mv $RES/reward_ablation_scores.tsv.tmp $RES/reward_ablation_scores.tsv
  note "$DATA: $n / $(ls ../../$DATA/jsons_step6 | wc -l) predictions; $(grep -F "$(printf '%s\t%s' $DATA $F)" $RES/reward_ablation_scores.tsv | cut -f4-6)"
done
( cd AP_evaluator && python3 mingle_det_density.py --out ../$LOGS/density_$SHORT.tsv $F > /dev/null 2>&1 )
grep -vF "$F" $RES/reward_ablation_density.tsv > $RES/reward_ablation_density.tsv.tmp
tail -n +2 $LOGS/density_$SHORT.tsv >> $RES/reward_ablation_density.tsv.tmp
mv $RES/reward_ablation_density.tsv.tmp $RES/reward_ablation_density.tsv
note "all-four video scores final"
