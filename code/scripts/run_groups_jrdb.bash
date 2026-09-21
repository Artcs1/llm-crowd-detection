#!/bin/bash
# Usage: bash run_groups.bash <model_name> <port> <api_key> <prompt_method> <type>
# Example:
#   bash run_groups.bash Qwen/Qwen2.5-VL-3B-Instruct 8081 testkey1 p1 full
#
# type          = "single" | "full"           --- only predict the target frame
# prompt_method = "p1" | "p1_bbox" | "p1_visual" | "p1_only_visual"
# modes         = auto-picked from model_name: "vlm_image" (Cosmos/VL models) or "llm" (others)
# depth_methods = ("naive_3D_60FOV" "unidepth_3D" "detany_3D" "wilddet_3D") --- fixed to wilddet_3D below

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <model_name> <port> <api_key> <prompt_method> <type>"
    echo "Example:"
    echo "  $0 Qwen/Qwen2.5-VL-3B-Instruct 8081 testkey1 p1 full"
    exit 1
fi

MODEL_NAME=$1
PORT=$2
API_KEY=$3
PROMPT_METHOD=$4
TYPE=$5
API_BASE="http://localhost:${PORT}/v1"

json_path="../../JRDB_fixed_gold/jsons_gold"
frame_path="../../JRDB_fixed_gold/videos_frames"
#json_path="../../gold_SEKAI_900_3/jsons_step5"
#frame_path="../../gold_SEKAI_900_3/videos_frames"
#json_path="../../EgoGroups_test/jsons_step6"
#frame_path="../../EgoGroups_test/videos_frames"

# List of group IDs to run
# IDS=(42 22 2)
IDS=(15)
# 10 20 30 40 50)

if [[ "$MODEL_NAME" == *"Cosmos"* ]]; then
  modes=("vlm_image" "llm") 
elif [[ "$MODEL_NAME" == *"Qwen3.6-27B"* ]]; then
  modes=("llm")
elif [[ "$MODEL_NAME" == *"VL"* ]]; then
  modes=("vlm_image")
else
  modes=("llm")
fi
#modes=("vlm_image")

depth_methods=("detany_3D")
#depth_methods=("3D")

for ID in "${IDS[@]}"; do
  echo "Running for ID ${ID}..."
  for mode in "${modes[@]}"; do
    # llm mode is only meaningful with the p1 prompt -- skip any other combination
    if [[ "$mode" == "llm" && "$PROMPT_METHOD" != "p1" ]]; then
      continue
    fi
    if [[ "$mode" == "llm" ]]; then
      max_tokens=12000      # ← choose your LLM value
    else
      max_tokens=24000     # ← VLM / others
    fi
    for depth_method in "${depth_methods[@]}"; do
      echo "Running: $TYPE | $mode | $PROMPT_METHOD | $depth_method | max_tokens=$max_tokens"
      echo python3 batch_fetch_groups.py "$json_path" "$TYPE" "$mode" "$MODEL_NAME" "$ID" --depth_method "$depth_method" --prompt_method "$PROMPT_METHOD" --api_base "$API_BASE" --api_key "$API_KEY" --max_tokens "$max_tokens" --frame_path "$frame_path" --save_image
      python3 batch_fetch_groups.py "$json_path" "$TYPE" "$mode" "$MODEL_NAME" "$ID" \
        --depth_method "$depth_method" \
        --prompt_method "$PROMPT_METHOD" \
        --api_base "$API_BASE" \
        --api_key "$API_KEY" \
        --max_tokens "$max_tokens" \
        --frame_path "$frame_path" \
        --save_image
    done
  done
done
echo "All jobs finished."
