#!/bin/bash

modes=("single" "full")
#prompt_methods=("p1" "p3" "baseline1" "baseline2")
prompt_methods=("p1" "p3")
models=(
  "Qwen2.5-72B-Instruct"
  "Qwen2.5-32B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-3B-Instruct"
)
#models=(
#  "Qwen2.5-VL-72B-Instruct"
#  "Qwen2.5-VL-32B-Instruct"
#  "Qwen2.5-VL-7B-Instruct"
#  "Qwen2.5-VL-3B-Instruct"
#)
vlm_modes=("llm")
#vlm_modes=("vlm_text" "vlm_image")


for model in "${models[@]}"; do
  for prompt_method in "${prompt_methods[@]}"; do
    for mode in "${modes[@]}"; do
      # skip p3 for full mode
      if [[ "$mode" == "full" && "$prompt_method" == "p3" ]]; then
        continue
      fi
      for vlm_mode in "${vlm_modes[@]}"; do
        echo "Running: $model | $prompt_method | $mode | $vlm_mode"
        python3 compute_detections.py \
          --mode "$mode" \
          --prompt_method "$prompt_method" \
          --model "$model" \
          --vlm_mode "$vlm_mode"
      done
    done
  done
done
