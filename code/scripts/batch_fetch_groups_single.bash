#!/bin/bash



model_name=$1
api_base=$2
api_key=$3

json_path="../../JRDB/jsons_gold/"
frame_path="../../JRDB/videos_frames/"
frame_id=15

types=("single" "full")
#modes=("llm")
modes=("vlm_image" "vlm_text")
prompts=("p1" "p3")
#prompts=("p1" "p3" "baseline1" "baseline2")
for type in "${types[@]}"; do
  for mode in "${modes[@]}"; do
    
    if [[ "$mode" == "llm" || "$mode" == "vlm_text" ]]; then
      prompts=("p1" "p3")
    else
      prompts=("p1" "p3")
    fi

    for prompt in "${prompts[@]}"; do
      # Skip p3 for full type
      if [[ "$type" == "full" && "$prompt" == "p3" ]]; then
        continue
      fi

      echo "Running: $type | $mode | $prompt"
      python3 batch_fetch_groups.py "$json_path" "$type" "$mode" "$model_name" "$frame_id" \
        --prompt_method "$prompt" \
        --api_base "$api_base" \
        --api_key "$api_key" \
        --frame_path "$frame_path" \
        --save_image
    done
  done
done

