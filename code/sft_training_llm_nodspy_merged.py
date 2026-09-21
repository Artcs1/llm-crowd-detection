# Merged counterpart of sft_training_llm_nodspy.py + sft_training_llm_nodspy_full.py (text-only,
# DSPy-free SFT of Qwen2.5-7B-Instruct via trl.SFTTrainer). The two source scripts were
# near-identical -- same model, LoRA config, SFTConfig, and training/save/sanity-check flow --
# differing only in which data-loader function they call and the resulting prompt/checkpoint
# naming, so this file picks between them with --mode instead of being two separate files.
# sft_training_llm_nodspy.py / sft_training_llm_nodspy_full.py are left in place, unchanged --
# this is an additional entry point, not a replacement.
import argparse
import json
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='SFT fine-tune Qwen2.5-7B-Instruct (text-only, no DSPy) on real ground truth.'
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
        '--mode', type=str, choices=['single', 'full'], default='single',
        help="'single' trains on a bare single-frame input (get_frame_bboxes / build_sft_examples, "
             "matching sft_training_llm_nodspy.py); 'full' trains on the target frame's detections "
             "enriched with a per-person 'movement_direction' label computed from every earlier "
             "frame in the scenario (get_allframes_bboxes / build_sft_examples_full, matching "
             "sft_training_llm_nodspy_full.py and args.setting == 'full' at inference time)."
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
print(f'dataset: {args.dataset}  mode: {args.mode}  gpu: {args.gpu or "(default)"}')

# Must happen before torch is imported (see below) -- avoids the multi-GPU device_map='auto' +
# Trainer label/hidden-state device-mismatch crash. --gpu overrides the hardcoded default; if
# omitted, an already-exported CUDA_VISIBLE_DEVICES env var wins, else fall back to '2'.
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '2')

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import egogroups_data_utils
import egogroups_train_data_utils
import sft_data_utils

# Suffix applied to every saved checkpoint dir below so runs never clobber each other. Matches
# the two source scripts' conventions exactly: 'single'+'jrdb' keeps the original, unsuffixed
# names (so existing checkpoints/paths and anything hardcoding
# sft_output/qwen2.5-7b-groups-merged-nodspy, e.g. rl_training_llm_nodspy.py's MODEL_NAME, keep
# working unchanged); 'full' always adds '-full' before the optional '-egogroups'.
mode_suffix = '-full' if args.mode == 'full' else ''
dataset_suffix = f'-{args.dataset}' if args.dataset != 'jrdb' else ''
DATASET_SUFFIX = f'{mode_suffix}{dataset_suffix}'

# Load ground-truth examples and split by scenario. For 'jrdb', build_sft_examples[_full] (in
# sft_data_utils.py) scans all 27 JRDB gold scenarios, matches each gt.pkl frame entry to its
# detections JSON by track-id overlap. For 'egogroups'/'egogroups-subset', the
# egogroups_data_utils.py counterpart does the same job against gold_SEKAI_900_3's 900 clips and
# its three per-target-frame gt_gold_sekai_{2,22,42}.pkl files. All four builder functions return
# the same dict schema -- frame_input_data paired with the real ground-truth gt_groups
# (list[list[int]]) -- so everything below is dataset- and mode-agnostic. require_image=False
# since this path doesn't need images. Split by scenario_idx (not by individual example) so
# frames from the same scenario never appear in both train and val -- otherwise near-duplicate
# frames from one held-in scenario would leak into validation.
#
# exclude_all_singleton is only a parameter on the egogroups_data_utils.py builders (it's
# meaningless for jrdb per the user's request), so the jrdb branch calls its builder plain.
if args.dataset == 'jrdb':
    build_examples = sft_data_utils.build_sft_examples if args.mode == 'single' else sft_data_utils.build_sft_examples_full
    examples = build_examples(require_image=False)
elif args.dataset in ('egogroups', 'egogroups-subset'):
    build_examples = egogroups_data_utils.build_sft_examples if args.mode == 'single' else egogroups_data_utils.build_sft_examples_full
    examples = build_examples(require_image=False, exclude_all_singleton=(args.dataset == 'egogroups-subset'))
else:
    build_examples = egogroups_train_data_utils.build_sft_examples if args.mode == 'single' else egogroups_train_data_utils.build_sft_examples_full
    examples = build_examples(require_image=False, exclude_all_singleton=(args.dataset == 'egogroups-train-subset'))
print(f'{len(examples)} ground-truth examples across {len({e["scenario_idx"] for e in examples})} scenarios')

scenario_ids = sorted({e['scenario_idx'] for e in examples})
print(scenario_ids)
val_scenarios = set(scenario_ids[-100:])  # hold out the last of N scenarios entirely

train_examples = [e for e in examples]  # if e['scenario_idx'] not in val_scenarios]
val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]

print(f'train: {len(train_examples)} examples, val: {len(val_examples)} examples')
print(f'held-out scenarios: {sorted(val_scenarios)}')

# Hand-written prompts (no dspy.Signature), copied verbatim from prompts.py's IdentifyGroups /
# IdentifyGroups_AllFrames docstring/field descriptions (the signatures used at inference time
# for mode='llm', prompt_method='p1', single vs. 'full' setting), so the task description stays
# equivalent to each -- only the output-format instruction changes, from DSPy's [[ ## groups ## ]]
# markup to a plain JSON object.
SYSTEM_PROMPT_SINGLE = (
    "Given a list of people with their 3D positions, group them into sets where each set "
    "contains people who are close to each other in space. Compute all pairwise distances "
    "between people. Choose a reasonable grouping threshold based on the distribution of "
    "these distances. People belong to the same group if their pairwise distances are below "
    "this threshold. Return only non-empty groups. Do not merge distant people into the same "
    "group. Do not hallucinate non-existent person_id.\n\n"
    "You are given a JSON array of people, each an object with keys 'person_id', 'x', 'y', 'z'.\n"
    "Respond with ONLY a JSON object of the form {\"groups\": [[person_id, ...], ...]} "
    "and no other text. All ids should appear at least once."
)

SYSTEM_PROMPT_FULL = (
    "Given detections of people with their 3D positions in a single video frame, compute groups "
    "of people who are close together. Compute pairwise distances between people and choose a "
    "reasonable grouping threshold based on the distribution of these distances. People belong to "
    "the same group if they are spatially close. Return only non-empty groups. Do not merge "
    "distant people into the same group. Do not hallucinate non-existent person_id.\n\n"
    "You are given a JSON array of the target frame's detections. Each is an object with keys "
    "'person_id', 'x', 'y', 'z', and 'movement_direction' (their net movement direction across "
    "the frames leading up to this one, or 'stationary').\n"
    "Respond with ONLY a JSON object of the form {\"groups\": [[person_id, ...], ...]} "
    "and no other text. All ids should appear at least once."
)

SYSTEM_PROMPT = SYSTEM_PROMPT_SINGLE if args.mode == 'single' else SYSTEM_PROMPT_FULL


def build_record(example):
    return {
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(example['frame_input_data'])},
            {'role': 'assistant', 'content': json.dumps({'groups': example['gt_groups']})},
        ]
    }


train_records = [build_record(e) for e in train_examples]
val_records = [build_record(e) for e in val_examples]

print(train_records[0]['messages'][1]['content'])
print(train_records[0]['messages'][2]['content'])

# Build HF Dataset objects. SFTTrainer natively understands the conversational (messages) format
# -- it applies the tokenizer's own chat template, so no manual tokenization/masking loop is
# needed here (unlike the VLM counterpart, which has to hand-roll tokenization in one of its
# variants because DSPy's local backend and SFTTrainer's conversational path don't support the
# multimodal case).
train_dataset = Dataset.from_list(train_records)
val_dataset = Dataset.from_list(val_records)
print(train_dataset, val_dataset)

# Load Qwen/Qwen2.5-7B-Instruct + standard LoRA config. LoraConfig is passed straight to
# SFTTrainer below (peft_config=), which wraps the model with get_peft_model internally -- no
# manual PEFT wrapping needed.
MODEL_NAME = 'Qwen/Qwen2.5-7B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to('cuda')

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    target_modules='all-linear',
    task_type='CAUSAL_LM',
)

# Train with trl.SFTTrainer. assistant_only_loss=True makes TRL mask the loss to the assistant
# turn only (it auto-swaps in a chat template with {% generation %} markers if the tokenizer's
# default template lacks them -- no manual labels = -100 masking loop needed).
sft_config = SFTConfig(
    output_dir=f'sft_output/qwen2.5-7b-groups-lora-nodspy{DATASET_SUFFIX}',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    bf16=True,
    assistant_only_loss=True,
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
    processing_class=tokenizer,
)

trainer.train()

# Save the LoRA adapter and a merged checkpoint.
adapter_dir = f'sft_output/qwen2.5-7b-groups-lora-nodspy{DATASET_SUFFIX}'
merged_dir = f'sft_output/qwen2.5-7b-groups-merged-nodspy{DATASET_SUFFIX}'

trainer.save_model(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size='5GB')
tokenizer.save_pretrained(merged_dir)

print(f'LoRA adapter saved to {adapter_dir}')
print(f'Merged checkpoint saved to {merged_dir}')

# Sanity-check on a held-out example. Note: unlike DSPy's ChatAdapter (which can retry/repair
# malformed structured output), there is no automatic retry here -- if the model's completion
# isn't valid JSON, json.loads raises. That's an accepted trade-off of going DSPy-free.
holdout = val_examples[0]

messages = [
    {'role': 'system', 'content': SYSTEM_PROMPT},
    {'role': 'user', 'content': json.dumps(holdout['frame_input_data'])},
]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt_text, return_tensors='pt').to(merged_model.device)

with torch.no_grad():
    generated = merged_model.generate(**inputs, max_new_tokens=512, do_sample=False)

completion = tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('raw completion:', completion)

predicted = json.loads(completion)
print('predicted groups:', predicted['groups'])
print('ground-truth groups:', holdout['gt_groups'])

# Using the result: point inference.ipynb's VLLMOfflineLM(model=...) at merged_dir, or serve it
# with `vllm serve <merged_dir>` / `vllm serve Qwen/Qwen2.5-7B-Instruct --enable-lora
# --lora-modules groups=<adapter_dir>`. Note the serving side would then need to speak this
# script's plain-JSON prompt/response convention (not DSPy's ChatAdapter markup) to match what
# the model was actually trained on -- e.g. `dspy.configure(lm=lm, adapter=dspy.JSONAdapter())`
# plus a signature instruction text matching SYSTEM_PROMPT_SINGLE/SYSTEM_PROMPT_FULL above
# (whichever matches --mode), or a direct non-DSPy call.
#
# Run with `--dataset {jrdb,egogroups}` (default jrdb) and `--mode {single,full}` (default
# single) to pick the ground-truth source and input shape; checkpoint dirs above are suffixed
# accordingly (single+jrdb keeps the original, unsuffixed names to stay compatible with anything
# hardcoding sft_output/qwen2.5-7b-groups-merged-nodspy).
#
# This file is equivalent to running sft_training_llm_nodspy.py (--mode single) or
# sft_training_llm_nodspy_full.py (--mode full) -- those two files are unchanged and still work
# standalone; this is just a single entry point covering both.
