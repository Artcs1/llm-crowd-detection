# Merged counterpart of sft_training_vlm_nodspy.py + sft_training_vlm_nodspy_visualonly.py
# (image-grounded, DSPy-free SFT of Qwen2.5-VL-7B-Instruct via trl.SFTTrainer, single-setting
# only -- mirrors sft_training_llm_nodspy_merged.py's approach for the text-only scripts). The two
# source scripts are near-identical -- same model, LoRA config, SFTConfig, and
# training/save/sanity-check flow -- differing only in the system prompt and whether the user
# turn includes numeric x,y,z coordinates (p1_visual) or just a bare person_ids list (idsonly,
# i.e. p1_visual_only), so this file picks between them with --variant instead of being two
# separate files. sft_training_vlm_nodspy.py / sft_training_vlm_nodspy_visualonly.py are left in
# place, unchanged -- this is an additional entry point, not a replacement. (VLM full-setting
# scripts -- sft_training_vlm_nodspy_full.py / _full_video.py -- are out of scope for this merge.)
import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='SFT fine-tune Qwen2.5-VL-7B-Instruct (image-grounded, no DSPy) on real ground truth.'
    )
    parser.add_argument(
        '--dataset', type=str,
        choices=['jrdb', 'egogroups', 'egogroups-subset', 'egogroups-train', 'egogroups-train-subset'],
        default='jrdb',
        help="'jrdb' uses sft_data_utils.py (JRDB_train_fixed_gold, F1_evaluator/out/gt.pkl); "
             "'egogroups' uses egogroups_data_utils.py (gold_SEKAI_900_3, "
             "F1_evaluator/out/gt_gold_sekai_{2,22,42}.pkl); 'egogroups-train' uses "
             "egogroups_train_data_utils.py (EgoGroups_train, "
             "F1_evaluator/out/gt_group_10fps_train.pkl); the '-subset' variants are the "
             "same source further filtered to drop all-singleton examples (zero real/"
             "non-singleton groups), on top of the empty-groups filter that always applies."
    )
    parser.add_argument(
        '--variant', type=str, choices=['p1_visual', 'idsonly'], default='p1_visual',
        help="'p1_visual' sends the bbox/id-annotated image plus numeric x,y,z coordinates "
             "(matching sft_training_vlm_nodspy.py); 'idsonly' sends only the annotated image "
             "plus a bare person_ids list, no coordinates at all (matching "
             "sft_training_vlm_nodspy_visualonly.py / prompt_method='p1_visual_only'). Only "
             "'p1_visual' is supported when --mode full (see --mode)."
    )
    parser.add_argument(
        '--mode', type=str, choices=['single', 'full'], default='single',
        help="'single' trains on a bare single-frame input (build_sft_examples, matching this "
             "script's default); 'full' trains on the target frame's detections enriched with a "
             "per-person 'movement_direction' label computed from every earlier frame in the "
             "scenario (build_sft_examples_full, matching sft_training_vlm_nodspy_full.py -- "
             "single annotated image + movement_direction as text, NOT a multi-image video like "
             "sft_training_vlm_nodspy_full_video.py). Only --variant p1_visual is supported in "
             "full mode -- no idsonly+full prompt exists in this codebase."
    )
    parser.add_argument(
        '--gpu', type=str, default=None,
        help="CUDA_VISIBLE_DEVICES value to pin this run to (e.g. '3'), overriding the hardcoded "
             "default GPU below. If omitted, respects an already-exported CUDA_VISIBLE_DEVICES "
             "env var, falling back to the hardcoded default otherwise. Parsed this early in the "
             "file (before torch/etc. are imported) specifically so this can take effect."
    )
    return parser.parse_args()


args = parse_args()
print(f'dataset: {args.dataset}  variant: {args.variant}  mode: {args.mode}  gpu: {args.gpu or "(default)"}')

if args.mode == 'full' and args.variant == 'idsonly':
    raise ValueError("--mode full only supports --variant p1_visual (no idsonly+full prompt exists)")

# Must happen before torch is imported (see below) -- avoids the multi-GPU device_map='auto' +
# Trainer label/hidden-state device-mismatch crash. --gpu overrides the hardcoded default; if
# omitted, an already-exported CUDA_VISIBLE_DEVICES env var wins, else fall back to '1'.
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

import cv2
import torch
from datasets import Dataset, Image as HFImage
from peft import LoraConfig
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

import egogroups_data_utils
import egogroups_train_data_utils
import sft_data_utils

# Suffix applied to every saved checkpoint dir below so runs never clobber each other. Matches
# the two source scripts' conventions exactly: 'p1_visual'+'jrdb' keeps the original, unsuffixed
# names; 'idsonly' always adds '-idsonly' before the optional '-egogroups[-subset]'. 'full' adds
# '-full' before everything else, matching sft_training_vlm_nodspy_full.py's own DATASET_SUFFIX
# convention ('-full' or '-full-<dataset>').
mode_suffix = '-full' if args.mode == 'full' else ''
variant_infix = '-idsonly' if args.variant == 'idsonly' else ''
dataset_suffix = f'-{args.dataset}' if args.dataset != 'jrdb' else ''
DATASET_SUFFIX = f'{mode_suffix}{variant_infix}{dataset_suffix}'

# Load ground-truth examples and split by scenario. require_image=True so every example has a
# matching frame image on disk. Split by scenario_idx (not by individual example) so frames from
# the same scenario never appear in both train and val. exclude_all_singleton is only a parameter
# on the egogroups_data_utils.py builder (meaningless for jrdb), so the jrdb branch calls its
# builder plain; egogroups_data_utils.build_sft_examples also doesn't accept sft_data_utils.py's
# JRDB-only scenario_range kwarg, so it's omitted for that branch too.
if args.dataset == 'jrdb':
    build_examples = sft_data_utils.build_sft_examples if args.mode == 'single' else sft_data_utils.build_sft_examples_full
    examples = build_examples(require_image=True)
elif args.dataset in ('egogroups', 'egogroups-subset'):
    build_examples = egogroups_data_utils.build_sft_examples if args.mode == 'single' else egogroups_data_utils.build_sft_examples_full
    examples = build_examples(
        require_image=True, exclude_all_singleton=(args.dataset == 'egogroups-subset'),
    )
else:
    build_examples = egogroups_train_data_utils.build_sft_examples if args.mode == 'single' else egogroups_train_data_utils.build_sft_examples_full
    examples = build_examples(
        require_image=True, exclude_all_singleton=(args.dataset == 'egogroups-train-subset'),
    )
print(f'{len(examples)} image-grounded ground-truth examples across {len({e["scenario_idx"] for e in examples})} scenarios')

scenario_ids = sorted({e['scenario_idx'] for e in examples})
val_scenarios = set(scenario_ids[-3:])  # same held-out convention as sft_training_llm_nodspy.py

train_examples = [e for e in examples]  # if e['scenario_idx'] not in val_scenarios]
val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]

print(f'train: {len(train_examples)} examples, val: {len(val_examples)} examples')
print(f'held-out scenarios: {sorted(val_scenarios)}')

# Copied verbatim from prompts.py's vlm_IdentifyGroupsImage / vlm_IdentifyGroupsImage_idsonly
# docstring/field descriptions (the signatures used at inference time for prompt_method='p1_visual'
# / 'p1_visual_only' respectively), so the task description stays equivalent to each -- only the
# output-format instruction changes, from DSPy's [[ ## groups ## ]] markup to a plain JSON object.
SYSTEM_PROMPT_P1_VISUAL = (
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

SYSTEM_PROMPT_IDSONLY = (
    "Given an image with people annotated by bounding boxes and id labels, group the given "
    "person_ids into sets where each set contains people who are close to each other in the "
    "image. Use only the visual positions of the labeled boxes to judge proximity -- no numeric "
    "coordinates are provided. Return only non-empty groups. Do not merge distant people into "
    "the same group. Do not hallucinate non-existent person_id.\n\n"
    "You are given an image with each person's bounding box and id label drawn on them, plus "
    "a JSON array of the person_ids visible in it.\n"
    "Respond with ONLY a JSON object of the form {\"groups\": [[person_id, ...], ...]} "
    "and no other text. All ids should appear at least once."
)

# Copied verbatim from sft_training_vlm_nodspy_full.py's SYSTEM_PROMPT -- single annotated image +
# movement_direction folded into the detections JSON as text (not a multi-image video).
SYSTEM_PROMPT_FULL = (
    "Given detections of people with their 3D positions in a single video frame, compute groups "
    "of people who are close together. Compute pairwise distances between people and choose a "
    "reasonable grouping threshold based on the distribution of these distances. People belong to "
    "the same group if they are spatially close. Return only non-empty groups. Do not merge "
    "distant people into the same group. Do not hallucinate non-existent person_id.\n\n"
    "You are given an image with each person's bounding box and id label drawn on them, plus "
    "a JSON array of the target frame's detections. Each is an object with keys 'person_id', "
    "'x', 'y', 'z', and 'movement_direction' (their net movement direction across the frames "
    "leading up to this one, or 'stationary').\n"
    "Respond with ONLY a JSON object of the form {\"groups\": [[person_id, ...], ...]} "
    "and no other text. All ids should appear at least once."
)

if args.mode == 'full':
    SYSTEM_PROMPT = SYSTEM_PROMPT_FULL
else:
    SYSTEM_PROMPT = SYSTEM_PROMPT_P1_VISUAL if args.variant == 'p1_visual' else SYSTEM_PROMPT_IDSONLY


def build_user_content(example):
    """The only other thing that differs by --variant: p1_visual sends the full detections JSON
    (numeric x,y,z included), idsonly sends just the bare person_ids list."""
    if args.variant == 'p1_visual':
        return json.dumps(example['frame_input_data'])
    person_ids = [d['person_id'] for d in example['frame_input_data']]
    return json.dumps(person_ids)


def draw_bboxes_with_ids(image_path, boundingboxes):
    img = cv2.imread(image_path)
    color = (255, 0, 0)  # blue (BGR), same for every person
    for box in boundingboxes:
        t, l, b, r = box['t'], box['l'], box['b'], box['r']
        cv2.rectangle(img, (int(t), int(l)), (int(b), int(r)), color, 2)
        cv2.putText(img, str(box['person_id']), (int(t), max(int(l) - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# Same cache dir as sft_training_vlm_nodspy.py / _visualonly.py -- the bbox/id overlay itself is
# identical across variants (only the text prompt/user-content differs), so any frame already
# annotated by either script's runs is a free cache hit here too, and vice versa.
ANNOTATED_CACHE_DIR = 'sft_output/vn_annotated_frames_cache'


def get_annotated_image_path(example):
    # Cache key must be derived from the actual image_path, not (sequence_name, frame_id) --
    # a single sequence can be split into multiple shards (e.g. bytes-cafe-2019-02-07_0_00000
    # vs _00003), each contributing its own distinct "frame 15" image, so sequence_name+frame_id
    # collides across genuinely different images. image_path's own (shard folder, frame file)
    # is unique per distinct image; personid2bbox (and therefore the drawn annotation) is
    # identical for every example sharing the same image_path, so caching on the image itself
    # is both correct and avoids redundant redraws for examples that only differ in gt_groups.
    shard_name = os.path.basename(os.path.dirname(example['image_path']))
    frame_file = os.path.basename(example['image_path'])
    cache_path = os.path.join(ANNOTATED_CACHE_DIR, shard_name, frame_file)
    if not os.path.exists(cache_path):
        boundingboxes = [
            {'person_id': pid, 't': b[0], 'l': b[1], 'b': b[2], 'r': b[3]}
            for pid, b in example['personid2bbox'].items()
        ]
        annotated_image = draw_bboxes_with_ids(example['image_path'], boundingboxes)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        annotated_image.save(cache_path, quality=95)
    return cache_path


def build_record(example):
    return {
        'image': get_annotated_image_path(example),
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': build_user_content(example)},
        ],
        'completion': [
            {'role': 'assistant', 'content': json.dumps({'groups': example['gt_groups']})},
        ],
    }


train_records = [build_record(e) for e in train_examples]
val_records = [build_record(e) for e in val_examples]

print(train_records[0]['image'])
print(train_records[0]['prompt'][1]['content'])
print(train_records[0]['completion'][0]['content'])

# .cast_column('image', HFImage()) lazily decodes each cached path into a PIL.Image only when
# that row is actually accessed (during collation), instead of loading every frame up front.
train_dataset = Dataset.from_list(train_records).cast_column('image', HFImage())
val_dataset = Dataset.from_list(val_records).cast_column('image', HFImage())
print(train_dataset, val_dataset)

MODEL_NAME = 'Qwen/Qwen2.5-VL-7B-Instruct'

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16,
).to('cuda')

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias='none',
    target_modules='all-linear',
    task_type='CAUSAL_LM',
)

sft_config = SFTConfig(
    output_dir=f'sft_output/qwen2.5-vl-7b-groups-lora-nodspy{DATASET_SUFFIX}',
    num_train_epochs=3,  # VL model + images -- start here, extend once you've confirmed this works
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    bf16=True,
    completion_only_loss=True,
    packing=False,
    padding_free=False,
    max_length=4096,  # SFTConfig defaults to 1024, but Qwen2.5-VL expands each image into many
                       # placeholder tokens -- these wide 3760x480 JRDB frames alone can exceed
                       # that, so a fixed cap truncates input_ids while the vision tower still
                       # produces features for the full image, desyncing the two ("current
                       # max_length is too short and causes image placeholder tokens ... to be
                       # truncated").
    logging_steps=10,
    eval_strategy='epoch',
    save_strategy='epoch',
    report_to='none',
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    processing_class=processor,
)

trainer.train()

adapter_dir = f'sft_output/qwen2.5-vl-7b-groups-lora-nodspy{DATASET_SUFFIX}'
merged_dir = f'sft_output/qwen2.5-vl-7b-groups-merged-nodspy{DATASET_SUFFIX}'

trainer.save_model(adapter_dir)
processor.save_pretrained(adapter_dir)

merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size='5GB')
processor.save_pretrained(merged_dir)

print(f'LoRA adapter saved to {adapter_dir}')
print(f'Merged checkpoint saved to {merged_dir}')

# Reload the checkpoint fresh from disk for the sanity check instead of reusing the in-process
# `merged_model` object straight after training -- debug_vlm_generate.py found that Qwen2.5-VL's
# generate() can hit an internal position_ids bug (TypeError: 'NoneType' object is not
# subscriptable, in modeling_qwen2_5_vl.py's prepare_inputs_for_generation) when reused directly
# off a model object that just finished a full training run in the same process; never reproduced
# against a freshly-loaded checkpoint. Freeing the training-time objects first also gives the
# reload room.
del model, merged_model, trainer
torch.cuda.empty_cache()

eval_processor = AutoProcessor.from_pretrained(merged_dir)
eval_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(merged_dir, torch_dtype=torch.bfloat16).to('cuda')
eval_model.eval()

# Uses the same annotated (bbox+id) image via get_annotated_image_path() that training used, not
# the raw frame -- checking against the raw frame would test the model on an out-of-distribution
# input it never trained on. Unlike DSPy's ChatAdapter, there is no automatic retry here -- if the
# completion isn't valid JSON, json.loads raises.
holdout = val_examples[0]
holdout_image = Image.open(get_annotated_image_path(holdout)).convert('RGB')

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': [
        {'type': 'image', 'image': holdout_image},
        {'type': 'text', 'text': build_user_content(holdout)},
    ]},
]
prompt_text = eval_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = eval_processor(text=[prompt_text], images=[holdout_image], return_tensors='pt').to(eval_model.device)

with torch.no_grad():
    generated = eval_model.generate(**inputs, max_new_tokens=4096, do_sample=False)

completion = eval_processor.batch_decode(
    generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
)[0]
print('raw completion:', completion)

predicted = json.loads(completion)
print('predicted groups:', predicted['groups'])
print('ground-truth groups:', holdout['gt_groups'])

# Using the result: point inference.ipynb's VLLMOfflineLM(model=...) at merged_dir, or serve it
# with `vllm serve <merged_dir>` / `vllm serve Qwen/Qwen2.5-VL-7B-Instruct --enable-lora
# --lora-modules groups=<adapter_dir>`. The serving side needs to speak this script's plain-JSON
# prompt/response convention (not DSPy's ChatAdapter markup) -- a custom dspy.Adapter subclass
# reproducing SYSTEM_PROMPT_P1_VISUAL/SYSTEM_PROMPT_IDSONLY (whichever matches --variant) and the
# structured image+text message shape above, the same pattern inference_sft.ipynb uses for the
# text-only checkpoint (see PlainJSONAdapter there). The image itself must also carry the same
# bbox/id overlay this script trained on -- draw it with utils.draw_bboxes_with_ids() (the same
# visual annotation the matching prompt_method applies at inference time) before passing the
# image in; serving the raw, unannotated frame would be an out-of-distribution input relative to
# training. Also note --variant='idsonly' requests use person_ids=... instead of detections=...,
# with no numeric coordinates at all.
#
# Run with `--dataset {jrdb,egogroups,egogroups-subset}` (default jrdb) and `--variant
# {p1_visual,idsonly}` (default p1_visual) to pick the ground-truth source and prompt shape;
# checkpoint dirs above are suffixed accordingly (p1_visual+jrdb keeps the original, unsuffixed
# names to stay compatible with anything hardcoding sft_output/qwen2.5-vl-7b-groups-merged-nodspy).
#
# This file is equivalent to running sft_training_vlm_nodspy.py (--variant p1_visual) or
# sft_training_vlm_nodspy_visualonly.py (--variant idsonly) -- those two files are unchanged and
# still work standalone; this is just a single entry point covering both, now also with
# egogroups-subset support neither original script has. VLM full-setting scripts
# (sft_training_vlm_nodspy_full.py / _full_video.py) are untouched and out of scope here.
