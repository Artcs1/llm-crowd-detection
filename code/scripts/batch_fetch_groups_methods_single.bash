#!/bin/bash

# types = ("single" "full")  --- only predict the target frame
# depth_methods=("naive_3D_60FOV" "unidepth_3D" "detany_3D" "wilddet_3D") --- try different 3D estimation (x,y,z)
# modes = ("vlm_image" "llm" "vlm_text") --- mode depending of the LLM
# prompt_method = "p1" | "p1_bbox" | "p1_visual" | "p1_only_visual"  --- single prompt method, user-supplied
# EXAMPLE: bash scripts/batch_fetch_groups_methods_single.bash detany_3D_model http://localhost:8000/v1 testkey 15 p1

model_name=$1
api_base=$2
api_key=$3
frame_id=$4
prompt_method=$5

#json_path="../../JRDB_fixed_gold/jsons_gold"
#frame_path="../../JRDB_fixed_gold/videos_frames"

json_path="../../gold_SEKAI_900_3/jsons_step5"
frame_path="../../gold_SEKAI_900_3/videos_frames"

if [[ "$model_name" == *"Cosmos"* ]]; then
  modes=("llm" "vlm_image")
elif [[ "$model_name" == *"VL"* ]]; then
  modes=("vlm_image")
else
  modes=("llm")
fi

#modes=("vlm_image")
types=("single")
depth_methods=("wilddet_3D")

for type in "${types[@]}"; do
  for mode in "${modes[@]}"; do

    # llm mode is only meaningful with the p1 prompt -- skip any other combination
    if [[ "$mode" == "llm" && "$prompt_method" != "p1" ]]; then
      continue
    fi

    if [[ "$mode" == "llm" ]]; then
      max_tokens=12000      # ← choose your LLM value
    else
      max_tokens=24000     # ← VLM / others
    fi

    for depth_method in "${depth_methods[@]}"; do
      echo "Running: $type | $mode | $prompt_method | $depth_method | max_tokens=$max_tokens"

      echo python3 batch_fetch_groups.py "$json_path" "$type" "$mode" "$model_name" "$frame_id" --depth_method "$depth_method" --prompt_method "$prompt_method" --api_base "$api_base" --api_key "$api_key" --max_tokens "$max_tokens" --frame_path "$frame_path" --save_image

      python3 batch_fetch_groups.py "$json_path" "$type" "$mode" "$model_name" "$frame_id" \
        --depth_method "$depth_method" \
        --prompt_method "$prompt_method" \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --max_tokens "$max_tokens" \
        --frame_path "$frame_path" \
        --save_image

    done
  done
done
