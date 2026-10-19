#!/bin/bash
# Score one finished EgoGroups_test prediction run (after scripts/run_groups.bash) like every other method:
# detection file (AP_evaluator) -> grouping pkl (F1_evaluator) -> AP (task_3).
# Usage (from code/):
#   bash scripts/score_egogroups_test.bash <label without local/> <single|full> <llm|vlm_image> <p1|p1_visual>
# Example:
#   bash scripts/score_egogroups_test.bash Qwen2.5-7B-Instruct-egogroups-synth-detail-SFT single llm p1
set -e
MODEL=$1; TYPE=$2; VLM_MODE=$3; PROMPT=$4
[ "$#" -eq 4 ] || { echo "Usage: $0 <label> <single|full> <llm|vlm_image> <p1|p1_visual>"; exit 1; }
PREFIX=results; [ "$TYPE" = full ] && PREFIX=results_full
NAME=${PREFIX}_${MODEL}_${VLM_MODE}_detany_3D_${PROMPT}
DET=../results/detection_files/EgoGroups_test_10/$NAME.txt
PKL=../results/grouping_files/EgoGroups_test_10/$NAME.pkl

python3 compute_detections.py --dataset EgoGroups_test --mode "$TYPE" --depth_method detany_3D --prompt_method "$PROMPT" \
    --model "$MODEL" --vlm_mode "$VLM_MODE" --frame_id 10 --output_mode finegrained > /dev/null
python3 F1_evaluator/det_txt_to_pkl.py "$DET" "$PKL"
cd AP_evaluator
echo "$NAME AP (mean G1 G2 G3 G4 G5+): $(python3 JRDB_eval.py -g out/gt_group_10fps_test.txt -d ../$DET -t task_3 \
    --labelmap ./label_map/task_3.pbtxt --output all 2>/dev/null | grep -E '^[0-9.]+$' | tr '\n' ' ')"
