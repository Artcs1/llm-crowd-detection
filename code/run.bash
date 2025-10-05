#!/bin/bash

python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method baseline1 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method baseline2 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p1 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p2 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p3 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p4 --save_image

python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_text Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p1 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_text Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p2 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_text Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p3 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ single vlm_text Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p4 --save_image


python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ full vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method baseline1 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ full vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method baseline2 --save_image
python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ full vlm_image Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p1 --save_image

python3 batch_fetch_groups.py VBIG_dataset/jsons_step5/ full vlm_text Qwen/Qwen2.5-VL-72B-Instruct 1 --prompt_method p1 --save_image
