#!/bin/bash

model_name=$1
api_base=$2
api_key=$3

json_path="../../JRDB/jsons_step5"
frame_path="../../JRDB/videos_frames"
frame_id=15

modes=("vlm_image" "vlm_text")
if [[ "$model_name" == *"VL"* ]]; then
  modes=("vlm_image" "vlm_text")
else
  modes=("llm")
fi


#types=("full")
types=("single" "full")
prompts=("p1" "p2" "p3" "p4")
depth_methods=("naive_3D_60FOV" "unidepth_3D" "detany_3D")
#depth_methods=("detany_3D")

#prompts=("p1" "p3" "baseline1" "baseline2")
for type in "${types[@]}"; do
  for mode in "${modes[@]}"; do
    
    if [[ "$mode" == "llm" || "$mode" == "vlm_text" ]]; then
      prompts=("p1")
    else
      #prompts=("p1" "p2" "p3" "p4" "baseline1" "baseline2")
      prompts=("p1")
    fi

    for prompt in "${prompts[@]}"; do
      if [[ "$type" == "full" && ( "$prompt" == "p2" || "$prompt" == "p3" || "$prompt" == "p4" || "$prompt" == "p5" ) ]]; then
        continue
      fi
      for depth_method in "${depth_methods[@]}"; do
        echo "Running: $type | $mode | $prompt | $depth_method"
        python3 batch_fetch_groups.py "$json_path" "$type" "$mode" "$model_name" "$frame_id" \
	  --depth_method "$depth_method" \
          --prompt_method "$prompt" \
          --api_base "$api_base" \
          --api_key "$api_key" \
	  --max_tokens 12000 \
          --frame_path "$frame_path" \
          --save_image
      done
    done
  done
done

