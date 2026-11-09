#!/bin/bash
# Test-only: the all-four video RL saved to the unsuffixed checkpoint (REWARDS == all four -> no -rw- suffix),
# so run_reward_ablation_all4_video.sh skipped its test. Same test_model as the queue; GPU 1.
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
S=sft_output LOGS=sft_output/logs/reward_ablation Q=sft_output/logs/reward_ablation/claims$TAG
mkdir -p $LOGS $Q
RES=../results
JOBS="full:validity,bce,f1,ari"  # all four rewards, video (copy of run_reward_ablation_queue.sh with only this job)
note() { echo "[$(date)] $*"; }
code() { echo "$1" | sed 's/validity/v/' | tr , -; }   # validity,bce,f1 -> v-bce-f1 (= the checkpoint suffix order)

test_model() {  # test_model <gpu> <ckpt dir> <served name> <single|full>
  local G=$1 M=$2 N=$3 SET=$4 PORT=$((8360 + $1)) L=$LOGS/test_${3#*/}.log
  CUDA_VISIBLE_DEVICES=$G vllm serve "$M" --served-model-name "$N" --api-key egocity --port $PORT --gpu-memory-utilization 0.90 \
    --max-model-len 65536 --max-num-seqs 128 --limit-mm-per-prompt '{"image": 16}' > "$L" 2>&1 &
  local pid=$!
  until curl -sf -H "Authorization: Bearer egocity" "http://localhost:$PORT/v1/models" > /dev/null; do
    kill -0 $pid 2>/dev/null || { note "GPU $G: server for $N died, see $L"; return 1; }; sleep 10; done
  NAME=$N DATA=EgoGroupsOODCity_Test GT=benchmark3_id_test bash scripts/run_zeroshot_egogroupscity.sh $G "$M" p1_social $PORT 24 $SET > /dev/null 2>&1 &
  local a=$!
  NAME=$N DATA=EgoGroupsOODRegions_Test GT=benchmark3_ood_test bash scripts/run_zeroshot_egogroupscity.sh $G "$M" p1_social $PORT 24 $SET > /dev/null 2>&1 &
  local b=$!
  wait $a $b
  kill $pid; wait $pid 2>/dev/null
  local f=$([ $SET = full ] && echo results_full || echo results)_${N#*/}_vlm_image_detany_3D_p1_social.txt
  for t in "EgoGroupsOODCity_Test benchmark3_id_test" "EgoGroupsOODRegions_Test benchmark3_ood_test"; do
    read DATA GT <<< "$t"
    echo -e "$DATA\t$f\t$(cd AP_evaluator && python3 mingle_det.py -g out/gt_group_10fps_$GT.txt -d ../$RES/detection_files/${DATA}_10/$f 2>&1 | grep -vi warn | tail -1 | cut -f2-)" >> $RES/reward_ablation_scores.tsv
  done
  ( cd AP_evaluator && python3 mingle_det_density.py --out ../$LOGS/density_${N#*/}.tsv $f > /dev/null 2>&1 )
  tail -n +2 $LOGS/density_${N#*/}.tsv >> $RES/reward_ablation_density.tsv
  note "GPU $G: $N tested: $(grep -F "$f" $RES/reward_ablation_scores.tsv | cut -f1,4-6 | tr '\n' ' ')"
}
test_model 1 sft_output/qwen3-vl-4b-groups-grpo-merged-fourlosses-full-social-egogroups-benchmark3-1ep-nobiaskl-g8p1 Qwen/Qwen3-VL-4B-Instruct-RL-B3-Social-RW-v-bce-f1-ari-Full full
