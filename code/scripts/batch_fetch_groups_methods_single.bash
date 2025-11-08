#!/bin/bash

model_name=$1
api_base=$2
api_key=$3

json_path="../../SEKAI_OURS/jsons_step4"
frame_path="../../SEKAI_OURS/videos_frames"
frame_id=1

types=("single" "full")
#modes=("llm")
modes=("vlm_image" "vlm_text")
prompts=("p1" "p3")
depth_methods=("naive_3D_60FOV" "unidepth_3D" "detany_3D")

#prompts=("p1" "p3" "baseline1" "baseline2")
for type in "${types[@]}"; do
  for mode in "${modes[@]}"; do
    
    if [[ "$mode" == "llm" || "$mode" == "vlm_text" ]]; then
      prompts=("p1" "p3")
    else
      prompts=("p1" "p3" "baseline1" "baseline2")
    fi

    for prompt in "${prompts[@]}"; do
      # Skip p3 for full type
      if [[ "$type" == "full" && "$prompt" == "p3" ]]; then
        continue
      fi
      for depth_method in "${depth_methods[@]}"; do
        echo "Running: $type | $mode | $prompt | $depth_method"
        python3 batch_fetch_groups.py "$json_path" "$type" "$mode" "$model_name" "$frame_id" \
	  --depth_method "$depth_method" \
          --prompt_method "$prompt" \
          --api_base "$api_base" \
          --api_key "$api_key" \
          --frame_path "$frame_path" \
          --save_image
      done
    done
  done
done

