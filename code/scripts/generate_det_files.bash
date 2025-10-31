#!/bin/bash

modes=("single" "full")
prompt_methods=("p1" "p3" "baseline1" "baseline2")
#prompt_methods=("p1" "p3")
vlm_modes=("llm" "vlm_text" "vlm_image")
models=("")

for vlm_mode in "${vlm_modes[@]}"; do
  if [[ "$vlm_mode" == "llm" ]]; then
    models=("Qwen2.5-72B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-3B-Instruct")
  else
    models=("Qwen2.5-VL-72B-Instruct" "Qwen2.5-VL-32B-Instruct" "Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-3B-Instruct")
  fi
  for model in "${models[@]}"; do
    for prompt_method in "${prompt_methods[@]}"; do
      for mode in "${modes[@]}"; do
        # skip baseline for only text methods
	if [[ "$vlm_mode" == "vlm_text" || "$vlm_mode" == "llm" ]] && [[ "$prompt_method" == "baseline1" || "$prompt_method" == "baseline2" ]]; then
          continue
        fi
        # skip p3 for full mode
        if [[ "$mode" == "full" && "$prompt_method" == "p3" ]]; then
          continue
        fi
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
