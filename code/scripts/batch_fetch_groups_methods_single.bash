#!/bin/bash

# types = ("single" "full")  --- only predict the target frame 
# depth_methods=("naive_3D_60FOV" "unidepth_3D" "detany_3D") --- try different 3D estimation (x,y,z)
# modes = ("vlm_image" "llm" "vlm_text") --- mode depending of the LLM
# prompts = ("p1" "p2" "p3" "p4" "p5" "baseline1" "baseline2") --- different prompts

model_name=$1
api_base=$2
api_key=$3
frame_id=$4

json_path="../../../SEKAI_OURS_200/jsons_step4"
frame_path="../../../SEKAI_OURS_200/videos_frames"

if [[ "$model_name" == *"VL"* ]]; then
  modes=("vlm_image")
else
  modes=("llm")
fi

types=("single" "full")
prompts=("p1" "p2" "p3" "p4")
depth_methods=("naive_3D_60FOV" "unidepth_3D" "detany_3D")

for type in "${types[@]}"; do
  for mode in "${modes[@]}"; do

    if [[ "$mode" == "llm" ]]; then
      max_tokens=12000      # ← choose your LLM value
    else
      max_tokens=24000     # ← VLM / others
    fi
    
    if [[ "$mode" == "llm" || "$mode" == "vlm_text" ]]; then
      prompts=("p1")
      #prompts=("p1" "p2" "p3" "p4" "p5")
    else
      prompts=("p1")
      #prompts=("p1" "p2" "p3" "p4" "p5" "baseline1" "baseline2")
    fi

    for prompt in "${prompts[@]}"; do
      if [[ "$type" == "full" && ( "$prompt" == "p2" || "$prompt" == "p3" || "$prompt" == "p4" || "$prompt" == "p5" ) ]]; then
        continue
      fi
      for depth_method in "${depth_methods[@]}"; do
        echo "Running: $type | $mode | $prompt | $depth_method | max_tokens=$max_tokens"

        python3 batch_fetch_groups.py "$json_path" "$type" "$mode" "$model_name" "$frame_id" \
    	  --depth_method "$depth_method" \
          --prompt_method "$prompt" \
          --api_base "$api_base" \
          --api_key "$api_key" \
          --max_tokens "$max_tokens" \
          --frame_path "$frame_path" \
          --save_image

      done
    done
  done
done
