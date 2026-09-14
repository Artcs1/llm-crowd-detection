#!/bin/bash

# types = ("single" "full")  --- only predict the target frame 
# depth_methods=("naive_3D_60FOV" "unidepth_3D" "detany_3D" "wilddet_3D") --- try different 3D estimation (x,y,z)
# modes = ("vlm_image" "llm" "vlm_text") --- mode depending of the LLM
# prompts = ("p1" "p2" "p3" "p4" "p5" "baseline1" "baseline2") --- different prompts


dataset=$1
output_mode=$2
frame_id=$3

modes=("single" "full")
#"full")
prompt_methods=("p1" "p1_visual")
#"p1_bbox" "p1_visual" "p1_visual_only" "baseline1")
depth_methods=("detany_3D")
# "detany_3D" "wilddet_3D")
vlm_modes=("llm" "vlm_image")
#vlm_modes=("llm")

for vlm_mode in "${vlm_modes[@]}"; do
  if [[ "$vlm_mode" == "llm" ]]; then
    #models=("Qwen2.5-3B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct" "Qwen3-4B-Instruct-2507" "Qwen3-30B-A3B-Instruct-2507" "Cosmos-Reason2-8B" "Cosmos-Reason2-2B" "gemini" "Qwen2.5-7B-Instruct-SFT" "Qwen2.5-7B-Instruct-RL2F"  "Qwen2.5-7B-Instruct-RL4F" "Qwen2.5-7B-Instruct-RLS4F" "Qwen2.5-7B-Instruct-egogroups-RL2F" "Qwen2.5-7B-Instruct-egogroups-RL4F" "Qwen2.5-7B-Instruct-egogroups-RLS4F" "Qwen2.5-7B-Instruct-egogroups-SFT")
    models=("Qwen2.5-3B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct" "Qwen3-4B-Instruct-2507" "Qwen3-30B-A3B-Instruct-2507" "Cosmos-Reason2-8B" "Cosmos-Reason2-2B" "gemini" "NaiveCluster3D_t0.4" "Qwen2.5-7B-Instruct-egogroups-SFT" "Qwen2.5-7B-Instruct-egogroups-RL-ep1" "Qwen2.5-7B-Instruct-egogroups-RL-ep2" "Qwen2.5-7B-Instruct-egogroups-RL-ep3" "Qwen2.5-7B-Instruct-egogroups-subset-SFT" "Qwen2.5-7B-Instruct-egogroups-subset-RL-ep1" "Qwen2.5-7B-Instruct-egogroups-subset-RL-ep2" "Qwen2.5-7B-Instruct-egogroups-subset-RL-ep3" "Qwen2.5-VL-7B-Instruct-egogroups-SFT" "Qwen2.5-VL-7B-Instruct-egogroups-subset-SFT")
  else
    #models=("Qwen2.5-VL-3B-Instruct" "Qwen2.5-VL-7B-Instruct" "Qwen2.5-VL-32B-Instruct" "Qwen2.5-VL-72B-Instruct" "Qwen3-VL-4B-Instruct" "Qwen3-VL-30B-A3B-Instruct" "Qwen3-VL-235B-A22B-Instruct" "Cosmos-Reason2-8B" "Cosmos-Reason2-2B" "gemini" "Qwen2.5-VL-7B-Instruct-SFT" "Qwen2.5-VL-7B-Instruct-RL2F"  "Qwen2.5-VL-7B-Instruct-RL4F" "Qwen2.5-VL-7B-Instruct-RLS4F" "Qwen2.5-VL-7B-Instruct-egogroups-RL2F" "Qwen2.5-VL-7B-Instruct-egogroups-RL4F" "Qwen2.5VL--7B-Instruct-egogroups-RLS4F" "Qwen2.5-VL-7B-Instruct-egogroups-SFT")
    models=("Qwen2.5-3B-Instruct" "Qwen2.5-7B-Instruct" "Qwen2.5-32B-Instruct" "Qwen2.5-72B-Instruct" "Qwen3-4B-Instruct-2507" "Qwen3-30B-A3B-Instruct-2507" "Cosmos-Reason2-8B" "Cosmos-Reason2-2B" "gemini" "NaiveCluster3D_t0.4" "Qwen2.5-7B-Instruct-egogroups-SFT" "Qwen2.5-7B-Instruct-egogroups-RL-ep1" "Qwen2.5-7B-Instruct-egogroups-RL-ep2" "Qwen2.5-7B-Instruct-egogroups-RL-ep3" "Qwen2.5-7B-Instruct-egogroups-subset-SFT" "Qwen2.5-7B-Instruct-egogroups-subset-RL-ep1" "Qwen2.5-7B-Instruct-egogroups-subset-RL-ep2" "Qwen2.5-7B-Instruct-egogroups-subset-RL-ep3" "Qwen2.5-VL-7B-Instruct-egogroups-SFT" "Qwen2.5-VL-7B-Instruct-egogroups-subset-SFT")
  fi
  for model in "${models[@]}"; do
    for prompt_method in "${prompt_methods[@]}"; do
      for mode in "${modes[@]}"; do
        # skip baseline for only text methods
	if [[ "$vlm_mode" == "vlm_text" || "$vlm_mode" == "llm" ]] && [[ "$prompt_method" == "baseline1" || "$prompt_method" == "baseline2" ]]; then
          continue
        fi
        # skip p3 for full mode
	if [[ "$mode" == "full" && ( "$prompt_method" == "p2" || "$prompt_method" == "p3" || "$prompt_method" == "p4" || "$prompt_method" == "p5" ) ]]; then
          continue
        fi
       	for depth_method in "${depth_methods[@]}"; do	
	  echo "Running: $model | $prompt_method | $mode | $vlm_mode| $depth_method"
          python3 compute_detections.py \
	    --depth_method "$depth_method" \
	    --dataset "$dataset" \
            --mode "$mode" \
            --prompt_method "$prompt_method" \
            --model "$model" \
            --vlm_mode "$vlm_mode" \
            --frame_id "$frame_id" \
	    --output_mode "$output_mode"
	done
      done
    done
  done
done
