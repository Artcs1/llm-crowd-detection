"""Isolated smoke test for the GRPO-trained checkpoint -- loads the saved merged model directly
(no GRPOTrainer/GRPOConfig involved) and prints the RAW generation text before any JSON parsing,
so we can see why generate_groups() in rl_training_llm_nodspy.py is returning None instead of
guessing from the parsed (and therefore already-lossy) output.
"""
import json
import os
from collections import Counter

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')  # free GPU, see nvidia-smi

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, log_loss
from sklearn.metrics.cluster import pair_confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer

from sft_data_utils import build_sft_examples

MERGED_DIR = 'sft_output/qwen2.5-7b-groups-grpo-merged'
PARSE_FAILURE_REWARD = -1.0

SYSTEM_PROMPT = (
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

examples = build_sft_examples(scenario_range=None, require_image=False)
scenario_ids = sorted({e['scenario_idx'] for e in examples})
val_scenarios = set(scenario_ids[-3:])
val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]


def groups_to_labels(groups, person_ids):
    label_of = {}
    for gi, group in enumerate(groups):
        for pid in group:
            label_of[int(pid)] = gi
    next_label = len(groups)
    labels = []
    for pid in person_ids:
        if pid in label_of:
            labels.append(label_of[pid])
        else:
            labels.append(next_label)
            next_label += 1
    return labels


def partition_validity_reward(completions, person_ids, **kwargs):
    rewards = []
    for completion, pids in zip(completions, person_ids):
        text = completion[0]['content']
        try:
            pred_groups = json.loads(text)['groups']
            counts = Counter(int(pid) for group in pred_groups for pid in group)
            pid_set = set(pids)
            missing = pid_set - counts.keys()
            hallucinated = counts.keys() - pid_set
            duplicated = {pid for pid, c in counts.items() if c > 1}
            violations = len(missing) + len(hallucinated) + len(duplicated)
            rewards.append(1.0 - violations / max(len(pid_set), 1))
        except Exception:
            rewards.append(PARSE_FAILURE_REWARD)
    return rewards


def pairwise_f1_reward(completions, gt_groups, person_ids, **kwargs):
    rewards = []
    for completion, gt, pids in zip(completions, gt_groups, person_ids):
        text = completion[0]['content']
        try:
            pred_groups = json.loads(text)['groups']
            pred_labels = groups_to_labels(pred_groups, pids)
            gt_labels = groups_to_labels(gt, pids)
            (_, fp), (fn, tp) = pair_confusion_matrix(gt_labels, pred_labels)
            denom = 2 * tp + fp + fn
            rewards.append(1.0 if denom == 0 else (2 * tp) / denom)
        except Exception:
            rewards.append(PARSE_FAILURE_REWARD)
    return rewards


def affinity_bce_reward(completions, gt_groups, person_ids, **kwargs):
    rewards = []
    for completion, gt, pids in zip(completions, gt_groups, person_ids):
        text = completion[0]['content']
        try:
            pred_groups = json.loads(text)['groups']
            pred_labels = np.array(groups_to_labels(pred_groups, pids))
            gt_labels = np.array(groups_to_labels(gt, pids))
            iu = np.triu_indices(len(pids), k=1)
            gt_affinity = (gt_labels[:, None] == gt_labels[None, :])[iu].astype(float)
            pred_affinity = (pred_labels[:, None] == pred_labels[None, :])[iu].astype(float)
            pred_affinity = np.clip(pred_affinity, 0.1, 0.9)
            bce = log_loss(gt_affinity, pred_affinity, labels=[0, 1])
            rewards.append(float(np.exp(-bce)))
        except Exception:
            rewards.append(PARSE_FAILURE_REWARD)
    return rewards


def make_completion(groups):
    return [{'role': 'assistant', 'content': json.dumps({'groups': groups})}]

print(f'Loading tokenizer + model from {MERGED_DIR} ...')
tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
model = AutoModelForCausalLM.from_pretrained(MERGED_DIR, torch_dtype=torch.bfloat16).to('cuda')
model.eval()

print('pad_token:', tokenizer.pad_token, 'eos_token:', tokenizer.eos_token)
print('model eos_token_id:', model.generation_config.eos_token_id)
print()

for i, ex in enumerate(val_examples[:3]):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': json.dumps(ex['frame_input_data'])},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)

    print(f'=== example {i} === prompt tokens: {inputs["input_ids"].shape[1]}, num people: {len(ex["frame_input_data"])}')

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=12000,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = generated[0][inputs['input_ids'].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)

    print(f'--- raw completion ({len(new_tokens)} new tokens) ---')
    print(repr(completion[:2000]))
    print('...' if len(completion) > 2000 else '')

    try:
        pred = json.loads(completion)['groups']
        print('--- parsed OK, keys:', ['groups'])
    except Exception as e:
        pred = None
        print('--- json.loads FAILED:', repr(e))

    pids = [int(d['person_id']) for d in ex['frame_input_data']]
    gt = ex['gt_groups']
    ari = adjusted_rand_score(groups_to_labels(gt, pids), groups_to_labels(pred, pids)) if pred is not None else None

    unique_gts = [g_ for g in gt for g_ in g]
    val = partition_validity_reward([make_completion(pred)], [unique_gts])[-1] if pred is not None else None
    f1 = pairwise_f1_reward([make_completion(pred)], [gt], [pids])[-1] if pred is not None else None
    bce = affinity_bce_reward([make_completion(pred)], [gt], [pids])[-1] if pred is not None else None

    print('gt:  ', gt)
    print('pred:', pred)
    print('ARI: ', ari)
    print('VAL: ', val)
    print('F1:  ', f1)
    print('BCE: ', bce)
    print()
