"""Isolated repro for the sft_training_vlm_nodspy.py sanity-check crash -- loads the merged
checkpoint directly (no training, no SFTTrainer involved) and calls generate() on the same
held-out example, so we can see the full traceback and try different max_new_tokens values
without re-running the whole training script.
"""
import json
import os

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')  # pin to a single free GPU (see nvidia-smi)

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from sft_data_utils import build_sft_examples

MERGED_DIR = 'sft_output/qwen2.5-vl-7b-groups-merged-nodspy'

SYSTEM_PROMPT = (
    "Given a list of people with their 3D positions, group them into sets where each set "
    "contains people who are close to each other in space. Compute all pairwise distances "
    "between people. Choose a reasonable grouping threshold based on the distribution of "
    "these distances. People belong to the same group if their pairwise distances are below "
    "this threshold. Return only non-empty groups. Do not merge distant people into the same "
    "group.\n\n"
    "You are given an image with each person's bounding box and id label drawn on them, plus "
    "a JSON array of the people in it, each an object with keys 'person_id', 'x', 'y', 'z'.\n"
    "Respond with ONLY a JSON object of the form {\"groups\": [[person_id, ...], ...]} "
    "and no other text. All ids should appear at least once."
)

ANNOTATED_CACHE_DIR = 'sft_output/vn_annotated_frames_cache'


def get_annotated_image_path(example):
    shard_name = os.path.basename(os.path.dirname(example['image_path']))
    frame_file = os.path.basename(example['image_path'])
    cache_path = os.path.join(ANNOTATED_CACHE_DIR, shard_name, frame_file)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f'{cache_path} not cached -- run sft_training_vlm_nodspy.py first')
    return cache_path


print(f'Loading processor + model from {MERGED_DIR} ...')
processor = AutoProcessor.from_pretrained(MERGED_DIR)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MERGED_DIR, torch_dtype=torch.bfloat16).to('cuda')
model.eval()

examples = build_sft_examples(scenario_range=None, require_image=True)
scenario_ids = sorted({e['scenario_idx'] for e in examples})
val_scenarios = set(scenario_ids[-3:])
val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]

holdout = val_examples[0]
holdout_image = Image.open(get_annotated_image_path(holdout)).convert('RGB')

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': [
        {'type': 'image', 'image': holdout_image},
        {'type': 'text', 'text': json.dumps(holdout['frame_input_data'])},
    ]},
]
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt_text], images=[holdout_image], return_tensors='pt').to(model.device)

print('input_ids shape:', inputs['input_ids'].shape)
print('inputs keys:', list(inputs.keys()))
if 'image_grid_thw' in inputs:
    print('image_grid_thw:', inputs['image_grid_thw'])

for max_new_tokens in [4096]:
    print(f'\n=== trying max_new_tokens={max_new_tokens} ===')
    try:
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        completion = processor.batch_decode(
            generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
        )[0]
        print('SUCCESS. raw completion:', repr(completion[:300]))
    except Exception:
        import traceback
        print('FAILED:')
        traceback.print_exc()
