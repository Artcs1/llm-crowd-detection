# Merged counterpart of rl_training_llm_nodspy.py + rl_training_llm_nodspy_full.py -- both are the
# "fourlosses" GRPO reward variant (pairwise_f1 + affinity_bce + ari + partition_validity); the
# ari_validity and stratified reward scripts are NOT covered by this merge (out of scope, still
# separate files). The two source scripts are near-identical -- same reward functions, LoraConfig,
# GRPOConfig, and train/save/sanity-check flow -- differing only in SYSTEM_PROMPT, which
# build_sft_examples[_full] is called, DATASET_SUFFIX, and default GPU, so this file picks between
# them with --mode instead of being two separate files. rl_training_llm_nodspy.py /
# rl_training_llm_nodspy_full.py are left in place, unchanged -- this is an additional entry
# point, not a replacement.
import argparse
import json
import os
from collections import Counter


def parse_args():
    parser = argparse.ArgumentParser(
        description='GRPO fine-tune the no-DSPy SFT checkpoint (pairwise_f1/affinity_bce/ari/'
                    'validity reward) on real ground truth.'
    )
    parser.add_argument(
        '--dataset', type=str,
        choices=['jrdb', 'egogroups', 'egogroups-subset', 'egogroups-train', 'egogroups-train-subset'],
        default='jrdb',
        help="'jrdb' uses sft_data_utils.py + the jrdb-trained SFT checkpoint; 'egogroups' uses "
             "egogroups_data_utils.py + the egogroups-trained SFT checkpoint; 'egogroups-train' "
             "uses egogroups_train_data_utils.py + the egogroups-train-trained SFT checkpoint; "
             "the '-subset' variants are the same source further filtered to drop all-singleton "
             "examples (zero real/non-singleton groups), starting from the matching -subset SFT "
             "checkpoint (see sft_training_llm_nodspy_merged.py --dataset)."
    )
    parser.add_argument(
        '--mode', type=str, choices=['single', 'full'], default='single',
        help="'single' trains on a bare single-frame input (build_sft_examples, matching "
             "rl_training_llm_nodspy.py); 'full' trains on the target frame's detections enriched "
             "with a per-person 'movement_direction' label (build_sft_examples_full, matching "
             "rl_training_llm_nodspy_full.py)."
    )
    parser.add_argument(
        '--gpu', type=str, default=None,
        help="CUDA_VISIBLE_DEVICES value to pin this run to (e.g. '3'), overriding the hardcoded "
             "default GPU below. If omitted, respects an already-exported CUDA_VISIBLE_DEVICES "
             "env var, falling back to the hardcoded default otherwise. Parsed this early in the "
             "file (before torch/etc. are imported) specifically so this can take effect."
    )
    parser.add_argument(
        '--wandb', action='store_true',
        help="Report training metrics (including the reward curve) to Weights & Biases instead "
             "of nowhere (default report_to='none'). Requires wandb to already be authenticated "
             "in this environment (`wandb login` or a WANDB_API_KEY env var) -- not checked here."
    )
    parser.add_argument(
        '--wandb-project', type=str, default='llm-grpo-fourlosses',
        help="wandb project name to log to when --wandb is set. Ignored otherwise."
    )
    parser.add_argument(
        '--epochs', type=int, default=6,
        help="num_train_epochs. Default (6) matches every checkpoint saved before this flag "
             "existed, so it keeps producing the original, unsuffixed checkpoint dirs. Any other "
             "value gets a '-<N>ep' suffix appended to the output/adapter/merged dirs (on top of "
             "the usual dataset/mode suffix) so it can never overwrite a checkpoint trained with a "
             "different epoch count."
    )
    parser.add_argument(
        '--save-every-epoch', action='store_true',
        help="In addition to the final merged checkpoint saved after training completes, also "
             "save a full merged checkpoint after every epoch, to "
             "<merged_dir>-epoch<N>. Off by default -- each epoch's merged checkpoint is a full "
             "copy of the ~7B model (safetensors, same size as the final merged_dir), so this "
             "multiplies disk usage by num_train_epochs if left on for a long run. Adds real time "
             "per epoch too (merge + save + unmerge, done synchronously, blocking training)."
    )
    parser.add_argument(
        '--prompts-per-step', type=int, default=1,
        help="Number of unique prompts sampled together per optimizer step (each gets "
             "num_generations rollouts). Sets per_device_train_batch_size; "
             "gradient_accumulation_steps/num_generations are unchanged, so generation_batch_size "
             "= prompts_per_step * gradient_accumulation_steps must stay divisible by "
             "num_generations (already true for any value here given gradient_accumulation_steps=8, "
             "num_generations=8). Default 1 matches every checkpoint saved before this flag "
             "existed. Raising this increases the training micro-batch size (more simultaneous "
             "long sequences per forward/backward pass) -- watch for OOM, this config was "
             "previously tuned down specifically to avoid that."
    )
    return parser.parse_args()


args = parse_args()
print(f'dataset: {args.dataset}  mode: {args.mode}  gpu: {args.gpu or "(default)"}  '
      f'wandb: {args.wandb} ({args.wandb_project})')

# Must happen before torch is imported (see below) -- avoids the multi-GPU device_map='auto' +
# Trainer label/hidden-state device-mismatch crash. --gpu overrides the hardcoded default; if
# omitted, an already-exported CUDA_VISIBLE_DEVICES env var wins, else fall back to '0'.
if args.gpu is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# Standard way the HF Trainer's wandb integration picks up a project name; only set when --wandb
# is actually requested so a plain run never even imports/touches wandb.
if args.wandb:
    os.environ.setdefault('WANDB_PROJECT', args.wandb_project)
# Completion lengths vary a lot step to step (7-100+ people per frame), so repeated alloc/free of
# variably-sized tensors can fragment the CUDA allocator -- expandable_segments lets it grow a
# reserved block instead of hunting for a new contiguous one, which is what actually caused the
# "worked for 1 iteration, then OOM'd" pattern rather than an immediate failure. Must be set before
# torch is imported.
os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig
from sklearn.metrics import adjusted_rand_score, log_loss
from sklearn.metrics.cluster import pair_confusion_matrix
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

import egogroups_data_utils
import egogroups_train_data_utils
import sft_data_utils

# TRL's own GRPOTrainer.log() unconditionally prints a rich table of sampled completions to the
# console whenever log_completions=True, regardless of report_to -- there's no built-in flag to
# turn the console print off on its own. Patch the one module-level symbol that print gate checks;
# the wandb-logging branch below is gated separately on report_to and is unaffected. Applied
# unconditionally (not just when --wandb is set): local parquet files under
# <output_dir>/completions/ already capture every step's completions, so the console print is
# redundant either way.
import trl.trainer.grpo_trainer as _grpo_trainer_module
_grpo_trainer_module.is_rich_available = lambda: False

# Suffix applied to the starting SFT checkpoint and every saved GRPO checkpoint below, so runs
# never clobber each other or start from the wrong SFT stage. Matches
# sft_training_llm_nodspy_merged.py's own DATASET_SUFFIX convention exactly, so MODEL_NAME below
# resolves to that script's actual output path for the same --dataset/--mode.
mode_suffix = '-full' if args.mode == 'full' else ''
dataset_suffix = f'-{args.dataset}' if args.dataset != 'jrdb' else ''
DATASET_SUFFIX = f'{mode_suffix}{dataset_suffix}'

# Only touches saved-checkpoint dirs below (adapter_dir/merged_dir/output_dir), not MODEL_NAME --
# the starting SFT checkpoint doesn't depend on how many GRPO epochs this run will do.
epoch_suffix = '' if args.epochs == 6 else f'-{args.epochs}ep'
prompts_suffix = '' if args.prompts_per_step == 1 else f'-{args.prompts_per_step}p'
CKPT_SUFFIX = f'{DATASET_SUFFIX}{epoch_suffix}{prompts_suffix}'

# exclude_all_singleton is only a parameter on the egogroups_data_utils.py builders (meaningless
# for jrdb), so the jrdb branch calls its builder plain.
if args.dataset == 'jrdb':
    build_examples = sft_data_utils.build_sft_examples if args.mode == 'single' else sft_data_utils.build_sft_examples_full
    examples = build_examples(require_image=False)
elif args.dataset in ('egogroups', 'egogroups-subset'):
    build_examples = egogroups_data_utils.build_sft_examples if args.mode == 'single' else egogroups_data_utils.build_sft_examples_full
    examples = build_examples(require_image=False, exclude_all_singleton=(args.dataset == 'egogroups-subset'))
else:
    build_examples = egogroups_train_data_utils.build_sft_examples if args.mode == 'single' else egogroups_train_data_utils.build_sft_examples_full
    examples = build_examples(require_image=False, exclude_all_singleton=(args.dataset == 'egogroups-train-subset'))

scenario_ids = sorted({e['scenario_idx'] for e in examples})
val_scenarios = set(scenario_ids[-3:])  # same held-out split as the source scripts

train_examples = [e for e in examples]  # if e['scenario_idx'] not in val_scenarios]
val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]

print(f'train: {len(train_examples)} examples, val: {len(val_examples)} examples')
print(f'held-out scenarios: {sorted(val_scenarios)}')

# Hand-written prompts, copied verbatim from rl_training_llm_nodspy.py / rl_training_llm_nodspy_full.py
# (which themselves match sft_training_llm_nodspy[_full].py's SYSTEM_PROMPT -- must equal what the
# starting SFT checkpoint was trained on).
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


def build_prompt_record(example):
    person_ids = [int(d['person_id']) for d in example['frame_input_data']]
    return {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': json.dumps(example['frame_input_data'])},
        ],
        'gt_groups': example['gt_groups'],
        'person_ids': person_ids,
    }


train_dataset = Dataset.from_list([build_prompt_record(e) for e in train_examples])
val_dataset = Dataset.from_list([build_prompt_record(e) for e in val_examples])

PARSE_FAILURE_REWARD = -1.0


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


def ari_reward(completions, gt_groups, person_ids, **kwargs):
    rewards = []
    for completion, gt, pids in zip(completions, gt_groups, person_ids):
        text = completion[0]['content']
        try:
            pred_groups = json.loads(text)['groups']
            pred_labels = groups_to_labels(pred_groups, pids)
            gt_labels = groups_to_labels(gt, pids)
            rewards.append(adjusted_rand_score(gt_labels, pred_labels))
        except Exception:
            rewards.append(PARSE_FAILURE_REWARD)
    return rewards


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


def pairwise_f1_reward(completions, gt_groups, person_ids, log_extra=None, **kwargs):
    """Pair-counting F1: precision/recall over same-cluster vs. different-cluster pairwise
    decisions -- more robust to the same/different-cluster class imbalance than ARI."""
    if log_extra is not None:
        # GRPOTrainer injects this kwarg (see _calculate_rewards in trl's grpo_trainer.py) so a
        # reward function can attach an extra column to the completions table -- called from this
        # one reward function only, since calling it from more than one would duplicate/misalign
        # the column. Logged as a JSON string, not a raw list, since that's a safer table cell
        # across parquet/wandb than a nested Python object.
        log_extra('gt_groups', [json.dumps(g) for g in gt_groups])
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
    """BCE between the predicted and ground-truth NxN same-cluster/different-cluster affinity
    matrices, mapped to a (0, 1] reward via exp(-BCE). The predicted affinity is hard 0/1 (groups
    are exact partitions), so it's clipped to [0.1, 0.9] before BCE -- sklearn's own default
    epsilon is far tighter and would let a single wrong pair spike the loss to ~36."""
    rewards = []
    for completion, gt, pids in zip(completions, gt_groups, person_ids):
        text = completion[0]['content']
        try:
            pred_groups = json.loads(text)['groups']
            pred_labels = np.array(groups_to_labels(pred_groups, pids))
            gt_labels = np.array(groups_to_labels(gt, pids))
            iu = np.triu_indices(len(pids), k=1)  # unique unordered pairs
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


person_ids_test = [1, 2, 3, 4]
gt_test = [[1, 2], [3, 4]]

test_cases = {
    'perfect match':          [[1, 2], [3, 4]],
    'shuffled order':         [[4, 3], [2, 1]],
    'missing person (4)':     [[1, 2], [3]],
    'duplicated person (1)':  [[1, 2], [1, 3, 4]],
    'all singletons':         [[1], [2], [3], [4]],
}

completions_test = [make_completion(g) for g in test_cases.values()]
gt_groups_test = [gt_test] * len(test_cases)
person_ids_batch = [person_ids_test] * len(test_cases)

ari_scores = ari_reward(completions_test, gt_groups_test, person_ids_batch)
validity_scores = partition_validity_reward(completions_test, person_ids_batch)
f1_scores = pairwise_f1_reward(completions_test, gt_groups_test, person_ids_batch)
bce_scores = affinity_bce_reward(completions_test, gt_groups_test, person_ids_batch)

for name, ari, valid, f1, bce in zip(test_cases.keys(), ari_scores, validity_scores, f1_scores, bce_scores):
    print(f'{name:25s} ARI={ari:+.3f}  validity={valid:+.3f}  pairwise_F1={f1:+.3f}  affinity_BCE={bce:+.3f}')

malformed = [[{'role': 'assistant', 'content': 'not json at all'}]]
print('malformed JSON:            ARI=%+.3f  validity=%+.3f  pairwise_F1=%+.3f  affinity_BCE=%+.3f' % (
    ari_reward(malformed, [gt_test], [person_ids_test])[0],
    partition_validity_reward(malformed, [person_ids_test])[0],
    pairwise_f1_reward(malformed, [gt_test], [person_ids_test])[0],
    affinity_bce_reward(malformed, [gt_test], [person_ids_test])[0],
))


# The SFT checkpoint matching this --dataset/--mode (sft_training_llm_nodspy_merged.py's
# merged_dir for the same DATASET_SUFFIX) -- must exist on disk before this script can run.
MODEL_NAME = f'sft_output/qwen2.5-7b-groups-merged-nodspy{DATASET_SUFFIX}'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias='none',
    target_modules='all-linear',
    task_type='CAUSAL_LM',
)

model_b = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16).to('cuda')


class SaveMergedEachEpochCallback(TrainerCallback):
    """Saves a full merged checkpoint after every epoch when --save-every-epoch is set. Uses
    LoraModel.merge_adapter()/unmerge_adapter() -- accessible as <PeftModel>.base_model, NOT
    directly on PeftModel itself (verified against the installed peft==0.19.1: PeftModel has no
    merge_adapter/unmerge_adapter, only LoraModel does) -- a reversible, in-place merge of the LoRA
    delta into the frozen base Linear weights, rather than merge_and_unload() (which permanently
    discards the LoRA modules and would corrupt the trainer's model for the remaining training
    still to come).

    IMPORTANT: merge_adapter() only updates the merged *value* of base_layer.weight in place -- it
    does NOT restructure the module tree the way merge_and_unload() does. get_base_model() (a
    stable, public PeftModel API returning self.base_model.model) still returns PEFT's
    peft.tuners.lora.Linear wrapper objects, which expose a `.weight` *property* that conveniently
    forwards to `base_layer.weight` (this is what made an earlier, insufficient version of this
    check -- comparing only `.weight` values -- appear correct). But state_dict()/save_pretrained()
    enumerate real registered submodules, not properties, so they produce keys like
    "...gate_up_proj.base_layer.weight" plus leftover "...lora_A.../lora_B..." entries instead of
    the plain "...gate_up_proj.weight" vLLM/transformers expect -- confirmed live: this shipped
    once already and broke `vllm serve` on a saved epoch checkpoint with
    `KeyError: 'layers.0.mlp.gate_up_proj.base_layer.weight'`. Fix, verified end-to-end against a
    toy LoRA model (state_dict() keys clean, strict=True load succeeds, output values match):
    build a cleaned state dict (strip ".base_layer." from keys, drop lora_A/lora_B keys -- already
    folded into base_layer.weight by the merge) and load it into a freshly reloaded plain skeleton
    (constructed on CPU, not touching the live GPU model/optimizer), then save that skeleton with
    the normal save_pretrained() -- reuses transformers' own sharding/config/index.json handling
    correctly instead of hand-rolling it.

    Deliberately uses kwargs['model'] (what Trainer's CallbackHandler passes to every event, per
    transformers/trainer_callback.py's `call_event(..., model=self.model, ...)`), NOT the
    module-level model_b -- GRPOTrainer.__init__ does `model = get_peft_model(model, peft_config)`
    internally (trl/trainer/grpo_trainer.py ~line 398), which reassigns *its own local* variable to
    the PEFT-wrapped model; it never mutates the model_b object this script passed in. model_b
    therefore stays the original bare AutoModelForCausalLM for the entire script, same reason the
    final save flow below uses `trainer.model.merge_and_unload()`, not `model_b.merge_and_unload()`
    -- confirmed by reproducing the resulting AttributeError live and tracing it to exactly this."""

    def on_epoch_end(self, args, state, control, **kwargs):
        if not globals()['args'].save_every_epoch:
            return
        model = kwargs['model']
        epoch_num = int(round(state.epoch))
        epoch_dir = f'sft_output/qwen2.5-7b-groups-grpo-merged-fourlosses{CKPT_SUFFIX}-epoch{epoch_num}'

        model.base_model.merge_adapter()
        raw_sd = model.get_base_model().state_dict()
        clean_sd = {
            k.replace('.base_layer.', '.'): v.detach().cpu()
            for k, v in raw_sd.items()
            if '.lora_A.' not in k and '.lora_B.' not in k
        }
        model.base_model.unmerge_adapter()  # done with the merged values now; restore for training
                                             # ASAP, before the slow disk-bound skeleton reload below

        skeleton = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
        skeleton.load_state_dict(clean_sd, strict=True)
        skeleton.save_pretrained(epoch_dir, safe_serialization=True, max_shard_size='5GB')
        tokenizer.save_pretrained(epoch_dir)
        del skeleton, raw_sd, clean_sd
        print(f'Epoch {epoch_num} merged checkpoint saved to {epoch_dir}')


grpo_config_b = GRPOConfig(
    output_dir=f'sft_output/qwen2.5-7b-groups-grpo-lora-fourlosses{CKPT_SUFFIX}',
    use_vllm=True,
    vllm_mode='colocate',
    vllm_gpu_memory_utilization=0.25,  # lowered from 0.3 -- was OOMing (needed 27.2GiB, 26.89GiB free)
    vllm_tensor_parallel_size=1,
    num_generations=8,  # TRL's own default (was 4) -- GRPO's relative-advantage estimate per
                         # prompt is noisier with a smaller sampled group, independent of how many
                         # epochs/steps are run. gradient_accumulation_steps below is bumped to
                         # match: TRL requires effective batch size (per_device_train_batch_size *
                         # gradient_accumulation_steps) to be evenly divisible by num_generations.
    per_device_train_batch_size=args.prompts_per_step,  # default 1 (halved from 2 originally --
                         # fewer simultaneous long sequences per step). Also controls how many
                         # unique prompts are sampled together per optimizer step -- see
                         # --prompts-per-step's help text.
    gradient_accumulation_steps=8,  # doubled from 4 to keep effective batch size (8) divisible by
                                     # the new num_generations=8 -- required by TRL, not a separate
                                     # tuning choice
    gradient_checkpointing=True,  # recompute activations in backward instead of storing them --
                                   # backprop through a near-max_completion_length sequence without
                                   # this was OOMing on some (but not all) steps
    max_completion_length=4096,
    beta=0.01,
    learning_rate=1e-5,
    reward_weights=[0.4, 0.15, 0.3, 0.15],  # pairwise_f1, affinity_bce, ari, validity
    logging_steps=1,
    num_train_epochs=args.epochs,  # default 6 (doubled from the original 3 -- that comment flagged
                          # 3 as "start here, extend if eval reward is still improving" and nobody
                          # had checked; GRPO's noisier reward-based signal needs more optimization
                          # exposure than SFT's dense cross-entropy signal to shape behavior
                          # comparably). --epochs overrides this; see CKPT_SUFFIX above for how a
                          # non-default value keeps its checkpoints separate from the 6-epoch ones.
    save_strategy='epoch',
    bf16=True,
    log_completions=args.wandb,  # off entirely without --wandb -- console printing is already
                            # suppressed either way (is_rich_available patch above), so with wandb
                            # off this flag's only remaining effects (table dict build + per-step
                            # completions_*.parquet write) would serve no consumer. Matches
                            # pre-completions-visibility-feature behavior exactly when off.
    num_completions_to_print=5,  # keep the console sample small; None would print every completion
                            # (moot when log_completions=False)
    report_to='wandb' if args.wandb else 'none',
    run_name=f'{args.dataset}-{args.mode}{epoch_suffix}{prompts_suffix}' if args.wandb else None,
)

trainer_b = GRPOTrainer(
    model=model_b,
    reward_funcs=[pairwise_f1_reward, affinity_bce_reward, ari_reward, partition_validity_reward],
    args=grpo_config_b,
    train_dataset=train_dataset,
    peft_config=lora_config,
    processing_class=tokenizer,
    callbacks=[SaveMergedEachEpochCallback()],
)

trainer_b.train()

trainer = trainer_b

adapter_dir = f'sft_output/qwen2.5-7b-groups-grpo-lora-fourlosses{CKPT_SUFFIX}'
merged_dir = f'sft_output/qwen2.5-7b-groups-grpo-merged-fourlosses{CKPT_SUFFIX}'

trainer.save_model(adapter_dir)
tokenizer.save_pretrained(adapter_dir)

merged_model = trainer.model.merge_and_unload()
merged_model.eval()
merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size='5GB')
tokenizer.save_pretrained(merged_dir)

print(f'LoRA adapter saved to {adapter_dir}')
print(f'Merged checkpoint saved to {merged_dir}')

def generate_groups(model, tokenizer, frame_input_data):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': json.dumps(frame_input_data)},
    ]
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors='pt').to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=12000, do_sample=False)
    completion = tokenizer.decode(generated[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    try:
        return json.loads(completion)['groups']
    except Exception:
        return None


for ex in val_examples[:3]:
    pids = [int(d['person_id']) for d in ex['frame_input_data']]
    pred = generate_groups(merged_model, tokenizer, ex['frame_input_data'])
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

# Run with `--dataset {jrdb,egogroups,egogroups-subset}` (default jrdb) and `--mode {single,full}`
# (default single) to pick the ground-truth source and input shape, and `--gpu <id>` to pin a GPU.
# Checkpoint dirs above are suffixed accordingly (single+jrdb keeps the original, unsuffixed names
# to stay compatible with anything hardcoding sft_output/qwen2.5-7b-groups-grpo-merged-fourlosses).
#
# This file is equivalent to running rl_training_llm_nodspy.py (--mode single) or
# rl_training_llm_nodspy_full.py (--mode full) -- those two files are unchanged and still work
# standalone; this is just a single entry point covering both, now also with egogroups-subset
# support neither original script has. The ari_validity and stratified reward variants are
# untouched and out of scope for this merge.
