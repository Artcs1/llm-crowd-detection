#!/bin/bash
# GRouPO reward ablation (Qwen3-VL-4B, benchmark3, --variant p1_social; LoRA scope LORA: '' = all-linear, as the GRouPO
# SFT checkpoints qwen3-vl-4b-groups-merged-nodspy[-full]-social-egogroups-benchmark3, or --lm_merger_lora; 1 RL epoch, no bias KL, 8 rollouts x 1 prompt, from the same p1_social SFT
# checkpoint), single and video (mode full). TAG (e.g. -groupo) selects the SFT checkpoints to start from: RL loads
# sft_output/qwen3-vl-4b-groups-merged-nodspy[-full]-social-egogroups-benchmark3[-lmmergerlora]$TAG (the --tag of
# rl_training_vlm_nodspy_merged.py), and every ablation checkpoint / served name carries TAG too. Reward sets (--rewards; each kept reward at its full-run weight
# f1 0.4 / bce 0.15 / ari 0.3 / validity 0.15):
#   R1 validity   R2 validity+bce   R3 validity+bce+f1 (= without ari)   R4 validity+bce+ari (= without f1)
#   R5 validity+f1+ari (= without bce)        (all four = GRouPO, not rerun)
# One lane per GPU in GPUS (default 0 1 2 3); each lane waits for the tmux sessions in WAIT_<gpu> to end, then takes the
# next unclaimed job (single R1..R5 first, then full R1..R5): RL, then OOD_city + OOD_regions tests (prompt p1_social,
# test setting = training mode) with MINGLE F1 and the per-density scores.
# Checkpoints: sft_output/qwen3-vl-4b-groups-grpo-merged-fourlosses[-full]-social-egogroups-benchmark3[-lmmergerlora]$TAG-1ep-nobiaskl-g8p1-rw-<set>
# Logs: sft_output/logs/reward_ablation/; scores: results/reward_ablation_scores.tsv, results/reward_ablation_density.tsv
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1 VLLM_USE_FLASHINFER_SAMPLER=0
GPUS=${GPUS:-0 1 2 3} TAG=${TAG:-} LORA=${LORA:-}
LSUF=$([ "$LORA" = --lm_merger_lora ] && echo -lmmergerlora)  # name piece of the LoRA scope
# MODES_<gpu>: the modes a lane takes (default both)
MODES_0=${MODES_0:-single full} MODES_1=${MODES_1:-single full} MODES_2=${MODES_2:-single full} MODES_3=${MODES_3:-single full}
WAIT_0=${WAIT_0:-runtime_gpu0} WAIT_1=${WAIT_1:-nan_check2 runtime_mingle} WAIT_2=${WAIT_2:-runtime_gpu2} WAIT_3=${WAIT_3:-runtime_gpu3}
S=sft_output LOGS=sft_output/logs/reward_ablation Q=sft_output/logs/reward_ablation/claims$TAG
mkdir -p $LOGS $Q
RES=../results
JOBS="full:validity,f1 full:validity,ari"  # the two remaining video ablations (copy of run_reward_ablation_queue.sh with only these jobs)
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

lane() {  # lane <gpu>
  local G=$1 w m; eval "w=\$WAIT_$G"; eval "m=\$MODES_$G"
  for s in $w; do while tmux has-session -t "=$s" 2>/dev/null; do sleep 120; done; done
  note "GPU $G: free"
  for job in $JOBS; do
    IFS=: read MODE RW <<< "$job"; local C=$(code $RW)
    [[ " $m " == *" $MODE "* ]] || continue   # not this lane's mode
    local SFT=$S/qwen3-vl-4b-groups-merged-nodspy$([ $MODE = full ] && echo -full)-social-egogroups-benchmark3$LSUF$TAG
    [ -f "$SFT/config.json" ] || { note "GPU $G: no SFT checkpoint $SFT, $MODE $RW skipped"; continue; }
    mkdir "$Q/${MODE}_$C" 2>/dev/null || continue   # claimed by another lane
    local MS=$([ $MODE = full ] && echo -full) MN=$([ $MODE = full ] && echo -Full)
    local CK=$S/qwen3-vl-4b-groups-grpo-merged-fourlosses$MS-social-egogroups-benchmark3$LSUF$TAG-1ep-nobiaskl-g8p1-rw-$C
    note "GPU $G: RL $MODE rewards $RW"
    MASTER_PORT=$((29800 + G)) python rl_training_vlm_nodspy_merged.py --base qwen3-4b --dataset egogroups-benchmark3 \
      --variant p1_social --mode $MODE $LORA --no_bias_kl --num_generations 8 --prompts_per_step 1 --epochs 1 \
      --rewards $RW --tag="$TAG" --gpu $G > $LOGS/rl_${MODE}_$C$TAG.log 2>&1
    note "GPU $G: RL $MODE $RW exit $?"
    chmod -R a+rwX "$CK" 2>/dev/null
    [ -f "$CK/config.json" ] || { note "GPU $G: no $CK, not tested"; continue; }
    test_model $G "$CK" "Qwen/Qwen3-VL-4B-Instruct-RL-B3-Social$([ -n "$LSUF" ] && echo -LMMerger)$TAG-RW-$C$MN" $MODE
  done
  note "GPU $G: lane done"
}

note "reward ablation queue on GPUs $GPUS"
pids=()
for g in $GPUS; do lane $g & pids+=($!); done
wait "${pids[@]}"
note "all done"
