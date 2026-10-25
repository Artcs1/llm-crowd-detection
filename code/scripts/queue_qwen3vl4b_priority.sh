#!/bin/bash
# Priority queues (Oct 24): base Qwen3-VL-4B on JRDB test first, then the egogroups-city runs, which resume from their
# saved predictions (--skip_existing).
#   LANE=gpu1: base full (video) on JRDB test -> full SFT (JRDB) on EgoGroupsCity_Test_jrdbstyle -> the 6 single JRDB models there
#   LANE=gpu2: base single on JRDB test -> zero-shot full on EgoGroupsCity_Test -> full RL (JRDB) on EgoGroupsCity_Test_jrdbstyle
# JRDB test scoring: AP (JRDB_eval.py task_3) + MINGLE F1 (mingle_det.py) vs AP_evaluator/out/gt_group.txt.
# Run inside tmux: tmux new -d -s q_gpu1 'LANE=gpu1 bash scripts/queue_qwen3vl4b_priority.sh'
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
S=sft_output
LG=sft_output/logs
BASE=Qwen/Qwen3-VL-4B-Instruct
NAME=Qwen/Qwen3-VL-4B-Instruct

jrdb_base() {  # <gpu> <single|full> <port> <maxlen> <clients>
  local gpu=$1 type=$2 port=$3 res L=$LG/jrdb_test_qwen3vl4b/Qwen3-VL-4B-Instruct_$2
  [ "$type" = full ] && res=results_full || res=results
  DATASET=jrdb MAXLEN=$4 GPUS="$gpu" UTILS=0.85 PORT0=$port PER=$5 \
    bash scripts/run_groups_fastgpus_v3.sh "$BASE" "$NAME" p1_visual "$type" "$L"
  python3 compute_detections.py --dataset JRDB_fixed_gold --mode "$type" --depth_method detany_3D --prompt_method p1_visual \
    --model Qwen3-VL-4B-Instruct --vlm_mode vlm_image --frame_id 15 > "$L/det.log" 2>&1
  local det=../../results/detection_files/JRDB_fixed_gold_15/${res}_Qwen3-VL-4B-Instruct_vlm_image_detany_3D_p1_visual.txt
  {
    echo "== Qwen3-VL-4B-Instruct p1_visual $type zero-shot on JRDB test ($(find ../results/predictions/JRDB_fixed_gold/$res/Qwen3-VL-4B-Instruct -name '*.json' | wc -l) / 1829)"
    echo "AP, G1..G5: $(cd AP_evaluator && python3 JRDB_eval.py -g out/gt_group.txt -d $det -t task_3 \
        --labelmap ./label_map/task_3.pbtxt --output all 2>/dev/null | grep -E '^[0-9.]+$' | tr '\n' ' ')"
    echo "MINGLE F1:"; (cd AP_evaluator && python3 mingle_det.py -g out/gt_group.txt -d $det 2>&1 | grep -v -i warn | tail -2)
  } > "$L/metrics.txt"
  cat "$L/metrics.txt"
}

if [ "$LANE" = gpu1 ]; then
  jrdb_base 1 full 8111 65536 32
  GPU_UTIL=0.85 DATA=EgoGroupsCity_Test_jrdbstyle NAME=Qwen/Qwen3-VL-4B-Instruct-SFT-JRDBTrain \
    bash scripts/run_zeroshot_egogroupscity.sh 1 $S/qwen3-vl-4b-groups-merged-nodspy-full-jrdbtrain p1_visual 8043 48 full \
    > $LG/egocity_jrdbstyle_qwen3vl4b_sft_full_driver_resume2.log 2>&1
  tail -4 $LG/egocity_jrdbstyle_qwen3vl4b_sft_full_driver_resume2.log
  bash scripts/queue_egocity_jrdbstyle_qwen3vl4b_single.sh
else
  jrdb_base 2 single 8112 32768 32
  GPU_UTIL=0.85 bash scripts/run_zeroshot_egogroupscity.sh 2 Qwen/Qwen3-VL-4B-Instruct p1_visual 8042 48 full \
    > $LG/egogroupscity_zeroshot_full_driver_resume2.log 2>&1
  tail -4 $LG/egogroupscity_zeroshot_full_driver_resume2.log
  GPU_UTIL=0.85 DATA=EgoGroupsCity_Test_jrdbstyle NAME=Qwen/Qwen3-VL-4B-Instruct-RL-JRDBTrain \
    bash scripts/run_zeroshot_egogroupscity.sh 2 $S/qwen3-vl-4b-groups-grpo-merged-fourlosses-full-jrdbtrain-1ep-nobiaskl-g8p1 p1_visual 8044 48 full \
    > $LG/egocity_jrdbstyle_qwen3vl4b_rl_full_driver.log 2>&1
  tail -4 $LG/egocity_jrdbstyle_qwen3vl4b_rl_full_driver.log
fi
echo "[$(date)] lane $LANE done"
