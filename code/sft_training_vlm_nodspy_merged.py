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
        choices=['jrdb', 'egogroups', 'egogroups-subset', 'egogroups-train', 'egogroups-train-subset',
                 'egogroups-synth-train', 'egogroups-synth-train-subset',
                 'egogroups-synth-detail-train', 'egogroups-synth-detail-train-subset', 'jrdbego', 'jrdbego-benchmark3', 'egogroups-city', 'egogroups-benchmark3', 'egogroups-k05', 'egogroups-k20',
                 'egogroups-k60', 'egogroups-k86', 'egogroups-k86v2',
                 'egogroups-k05_400', 'egogroups-k20_400', 'egogroups-k60_400', 'egogroups-k80_400'],
        default='jrdb',
        help="'jrdb' uses sft_data_utils.py (JRDB_train_fixed_gold, F1_evaluator/out/gt.pkl); "
             "'egogroups' uses egogroups_data_utils.py (gold_SEKAI_900_3, "
             "F1_evaluator/out/gt_gold_sekai_{2,22,42}.pkl); 'egogroups-train' uses "
             "egogroups_train_data_utils.py (EgoGroups_train, "
             "F1_evaluator/out/gt_group_10fps_train.pkl); the '-subset' variants are the "
             "same source further filtered to drop all-singleton examples (zero real/"
             "non-singleton groups), on top of the empty-groups filter that always applies. 'jrdbego': JRDB "
             "train (20 scenes) + EgoGroups_train (the clips outside the baselines' geo180 val split), plain union; "
             "validation on the 7 JRDB val scenes and the 180 geo180 clips, reported separately (eval_jrdb_loss, "
             "eval_ego_loss); checkpoints '-jrdbego-geo180'. 'jrdbego-benchmark3': JRDB train + benchmark3 train with the "
             "EgoGroups 3D in the JRDB robot frame (egogroups_city_data_utils.to_jrdb_style), validation on the 7 JRDB val "
             "scenes only (id_val unused); checkpoints '-jrdbego-benchmark3'. 'egogroups-city': the egogroups-city split "
             "(egogroups_city.py; clips of EgoGroups_train and EgoGroups_test, no city shared between splits), "
             "train on its 1620 train clips, validate on its 180 val clips; checkpoints '-egogroups-city'. "
             "'egogroups-benchmark3': the same loader on benchmark3 (benchmark3_{train,id_val}_clip_ids.txt, 1533 / 219 "
             "clips; README_benchmark3.md), checkpoints '-egogroups-benchmark3'. 'egogroups-k05' / -k20 / -k60 / -k86: "
             "benchmark3 with only the 300 K??_clip_ids.txt train clips (same id_val), checkpoints '-egogroups-k05' etc."
    )
    parser.add_argument(
        '--variant', type=str, choices=['p1_visual', 'p1_social', 'idsonly'], default='p1_visual',
        help="'p1_visual' sends the bbox/id-annotated image plus numeric x,y,z coordinates "
             "(matching sft_training_vlm_nodspy.py); 'idsonly' sends only the annotated image "
             "plus a bare person_ids list, no coordinates at all (matching "
             "sft_training_vlm_nodspy_visualonly.py / prompt_method='p1_visual_only'). With --mode full, "
             "idsonly sends person_id + movement_direction from the 2D box-centre motion (see --mode). "
             "'p1_social': p1_visual's inputs with the social_prompts.py system prompt (orientation, shared motion "
             "and interaction from the image, no chaining; prompt_method='p1_social' at test time); checkpoints '-social'."
    )
    parser.add_argument(
        '--mode', type=str, choices=['single', 'full'], default='single',
        help="'single' trains on a bare single-frame input (build_sft_examples, matching this "
             "script's default); 'full' trains on the target frame's detections enriched with a "
             "per-person 'movement_direction' label computed from every earlier frame in the "
             "scenario (build_sft_examples_full, matching sft_training_vlm_nodspy_full.py -- "
             "single annotated image + movement_direction as text, NOT a multi-image video like "
             "sft_training_vlm_nodspy_full_video.py). With --variant idsonly "
             "the JSON keeps only person_id + movement_direction, computed from the 2D box-centre "
             "motion (utils.get_movement_direction_2d) -- prompt_method='p1_visual_only' full at test time."
    )
    parser.add_argument(
        '--gpu', type=str, default=None,
        help="CUDA_VISIBLE_DEVICES value to pin this run to (e.g. '3'), overriding the hardcoded "
             "default GPU below. If omitted, respects an already-exported CUDA_VISIBLE_DEVICES "
             "env var, falling back to the hardcoded default otherwise. Parsed this early in the "
             "file (before torch/etc. are imported) specifically so this can take effect."
    )
    parser.add_argument(
        '--jrdb_val_split', action='store_true',
        help="Only with --dataset jrdb: train on the 20 JRDB train scenes and validate on the 7 val scenes of the "
             "baselines (sft_data_utils.JRDB_VAL_SCENES). Checkpoints get a '-jrdbtrain' suffix (RL then loads the "
             "SFT model with the same suffix). Without it, all 27 scenes are used for training."
    )
    parser.add_argument(
        '--ego_val_split', action='store_true',
        help="Only with --dataset egogroups-train[-subset]: hold out the 180 EgoGroups_train val clips shared with the "
             "baselines (egogroups_train_data_utils.VAL_SCENARIOS) and train on the other 1620. Checkpoints get a "
             "'-valsplit' suffix (RL then loads the SFT model with the same suffix). Without it, all 1800 clips are "
             "used for training (the checkpoints of the paper tables)."
    )
    parser.add_argument(
        '--lm_lora', action='store_true',
        help="LoRA on the language model's linear layers only (q/k/v/o, gate/up/down); the vision tower and the "
             "vision-to-text merger stay frozen. Default: 'all-linear' (vision tower + merger + language model). "
             "Checkpoints get an extra '-lmlora' suffix (RL --lm_lora then loads the SFT model with the same suffix)."
    )
    parser.add_argument(
        '--lm_merger_lora', action='store_true',
        help="LoRA on the language model's linear layers and on the vision-to-text merger (Qwen3-VL: also the deepstack "
             "mergers); the vision encoder blocks stay frozen. "
             "Checkpoints get an extra '-lmmergerlora' suffix (RL --lm_merger_lora loads the SFT model with the same suffix)."
    )
    parser.add_argument(
        '--no_eval', action='store_true',
        help="Skip the per-epoch validation pass (the val examples are still held out, and still used for the final "
             "one-example sanity check); saves ~8 min per epoch on 1080 val examples. Checkpoint names are unchanged.")
    parser.add_argument(
        '--tag', type=str, default='',
        help="Extra suffix appended to the checkpoint names (after '-lmlora'), e.g. '-fixedloader' to keep a rerun apart from older checkpoints of the same config. RL loads the SFT checkpoint with the same --tag.")
    parser.add_argument(
        '--base', type=str, choices=['qwen2.5', 'qwen3-4b'], default='qwen2.5',
        help="Base model (sft_data_utils.BASE_MODELS): 'qwen2.5' = Qwen/Qwen2.5-VL-7B-Instruct (default; the checkpoints of the paper "
             "tables), 'qwen3-4b' = Qwen/Qwen3-VL-4B-Instruct. Checkpoint dirs start with the base's prefix (qwen2.5-vl-7b-* / qwen3-vl-4b-*)."
    )
    return parser.parse_args()


args = parse_args()
print(f'dataset: {args.dataset}  variant: {args.variant}  mode: {args.mode}  gpu: {args.gpu or "(default)"}')


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
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer

import egogroups_city_data_utils
CITY_LIKE = {'egogroups-city': 'city', 'egogroups-benchmark3': 'benchmark3',
             **{f'egogroups-{k}': k for k in ('k05', 'k20', 'k60', 'k86', 'k86v2', 'k05_400', 'k20_400', 'k60_400', 'k80_400')}}  # --dataset -> egogroups_city.BENCHMARKS key
import egogroups_data_utils
import egogroups_synth_train_data_utils
import egogroups_train_data_utils
import sft_data_utils

# Suffix applied to every saved checkpoint dir below so runs never clobber each other. Matches
# the two source scripts' conventions exactly: 'p1_visual'+'jrdb' keeps the original, unsuffixed
# names; 'idsonly' always adds '-idsonly' before the optional '-egogroups[-subset]'. 'full' adds
# '-full' before everything else, matching sft_training_vlm_nodspy_full.py's own DATASET_SUFFIX
# convention ('-full' or '-full-<dataset>').
mode_suffix = '-full' if args.mode == 'full' else ''
variant_infix = {'idsonly': '-idsonly', 'p1_social': '-social'}.get(args.variant, '')
dataset_suffix = f'-{args.dataset}' if args.dataset != 'jrdb' else ''
if args.dataset == 'jrdb' and args.jrdb_val_split:
    dataset_suffix = '-jrdbtrain'
if args.dataset in ('egogroups-train', 'egogroups-train-subset') and args.ego_val_split:
    dataset_suffix += '-valsplit'
if args.dataset == 'jrdbego':
    dataset_suffix = '-jrdbego-geo180'
DATASET_SUFFIX = f'{mode_suffix}{variant_infix}{dataset_suffix}'
if args.lm_lora and args.lm_merger_lora:
    raise ValueError("--lm_lora and --lm_merger_lora are exclusive (LM only vs LM + merger)")
if args.lm_lora:
    DATASET_SUFFIX += '-lmlora'
if args.lm_merger_lora:
    DATASET_SUFFIX += '-lmmergerlora'
DATASET_SUFFIX += args.tag

# Load ground-truth examples and split by scenario. require_image=True so every example has a
# matching frame image on disk. Split by scenario_idx (not by individual example) so frames from
# the same scenario never appear in both train and val. exclude_all_singleton is only a parameter
# on the egogroups_data_utils.py builder (meaningless for jrdb), so the jrdb branch calls its
# builder plain; egogroups_data_utils.build_sft_examples also doesn't accept sft_data_utils.py's
# JRDB-only scenario_range kwarg, so it's omitted for that branch too.
if args.dataset == 'jrdb':
    build_examples = sft_data_utils.build_sft_examples if args.mode == 'single' else sft_data_utils.build_sft_examples_full
    examples = build_examples(require_image=True)
elif args.dataset == 'jrdbego':
    # both loaders, each example tagged with its source; EgoGroups scenario_idx shifted by JOINT_EGO_OFFSET so the two
    # datasets' scenario ids (JRDB 0-26, EgoGroups 0-1799) never collide in the scenario-level split below
    jrdb_build = sft_data_utils.build_sft_examples if args.mode == 'single' else sft_data_utils.build_sft_examples_full
    ego_build = (egogroups_train_data_utils.build_sft_examples if args.mode == 'single'
                 else egogroups_train_data_utils.build_sft_examples_full)
    JOINT_EGO_OFFSET = 10000
    examples = [dict(e, source='jrdb') for e in jrdb_build(require_image=True)]
    examples += [dict(e, source='ego', scenario_idx=e['scenario_idx'] + JOINT_EGO_OFFSET)
                 for e in ego_build(require_image=True, exclude_all_singleton=False)]
elif args.dataset == 'jrdbego-benchmark3':
    # JRDB train (native JRDB 3D) + benchmark3 train, the EgoGroups 3D rewritten in the JRDB robot frame
    # (egogroups_city_data_utils.to_jrdb_style), so JRDB test runs on JRDB_fixed_gold and the EgoGroups tests on the
    # *_jrdbstyle folders; id_val is not loaded; scenario ids shifted past JRDB's as in jrdbego
    jrdb_build = sft_data_utils.build_sft_examples if args.mode == 'single' else sft_data_utils.build_sft_examples_full
    ego_build = (egogroups_city_data_utils.build_sft_examples if args.mode == 'single'
                 else egogroups_city_data_utils.build_sft_examples_full)
    JOINT_EGO_OFFSET = 10000
    examples = [dict(e, source='jrdb') for e in jrdb_build(require_image=True)]
    examples += [dict(egogroups_city_data_utils.to_jrdb_style(e, args.mode), source='ego',
                      scenario_idx=e['scenario_idx'] + JOINT_EGO_OFFSET)
                 for e in ego_build(splits=('train',), benchmark='benchmark3', require_image=True, exclude_all_singleton=False)]
elif args.dataset in CITY_LIKE:
    # city train + val clips of both EgoGroups folders (scenario_idx = clip - 1); the city test clips are never loaded
    build_examples = (egogroups_city_data_utils.build_sft_examples if args.mode == 'single'
                      else egogroups_city_data_utils.build_sft_examples_full)
    examples = build_examples(splits=('train', 'val'), benchmark=CITY_LIKE[args.dataset], require_image=True,
                              exclude_all_singleton=False)
elif args.dataset in ('egogroups', 'egogroups-subset'):
    build_examples = egogroups_data_utils.build_sft_examples if args.mode == 'single' else egogroups_data_utils.build_sft_examples_full
    examples = build_examples(
        require_image=True, exclude_all_singleton=(args.dataset == 'egogroups-subset'),
    )
elif args.dataset in ('egogroups-train', 'egogroups-train-subset'):
    build_examples = egogroups_train_data_utils.build_sft_examples if args.mode == 'single' else egogroups_train_data_utils.build_sft_examples_full
    examples = build_examples(
        require_image=True, exclude_all_singleton=(args.dataset == 'egogroups-train-subset'),
    )
else:
    # EgoGroups_Synth_*: last frame of each 24-frame synthetic clip, exact gt_3D coordinates.
    if args.dataset.startswith('egogroups-synth-detail-train'):
        egogroups_synth_train_data_utils.set_synth_dir(egogroups_synth_train_data_utils.SYNTH_DETAIL_DIR)
    build_examples = egogroups_synth_train_data_utils.build_sft_examples if args.mode == 'single' else egogroups_synth_train_data_utils.build_sft_examples_full
    examples = build_examples(
        require_image=True, exclude_all_singleton=args.dataset.endswith('-subset'),
    )
print(f'{len(examples)} image-grounded ground-truth examples across {len({e["scenario_idx"] for e in examples})} scenarios')

scenario_ids = sorted({e['scenario_idx'] for e in examples})
val_scenarios = set(scenario_ids[-3:])  # same held-out convention as sft_training_llm_nodspy.py

train_examples = [e for e in examples]  # if e['scenario_idx'] not in val_scenarios]
val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]

if args.dataset in ('egogroups-train', 'egogroups-train-subset') and args.ego_val_split:
    # EgoGroups_train: hold out the shared 60/60/60 validation clips (never trained on), like every other method
    val_scenarios = egogroups_train_data_utils.VAL_SCENARIOS
    train_examples = [e for e in examples if e['scenario_idx'] not in val_scenarios]
    val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]

if args.dataset == 'jrdbego-benchmark3':  # validation on the 7 JRDB val scenes only (never trained on)
    val_scenarios = {e['scenario_idx'] for e in examples if e['source'] == 'jrdb' and e['sequence_name'] in sft_data_utils.JRDB_VAL_SCENES}
    train_examples = [e for e in examples if e['scenario_idx'] not in val_scenarios]
    val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]
    print(f"train: JRDB {sum(e['source'] == 'jrdb' for e in train_examples)}, EgoGroups {sum(e['source'] == 'ego' for e in train_examples)}; "
          f"val: JRDB {len(val_examples)}")

if args.dataset in CITY_LIKE:  # the city (benchmark3: id_val) val clips (never trained on)
    val_scenarios = {e['scenario_idx'] for e in examples if e['city_split'] == 'val'}
    train_examples = [e for e in examples if e['city_split'] == 'train']
    val_examples = [e for e in examples if e['city_split'] == 'val']

if args.dataset == 'jrdb' and args.jrdb_val_split:
    # JRDB: hold out the 7 val scenes of the baselines (never trained on)
    val_scenarios = {e['scenario_idx'] for e in examples if e['sequence_name'] in sft_data_utils.JRDB_VAL_SCENES}
    train_examples = [e for e in examples if e['scenario_idx'] not in val_scenarios]
    val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]

if args.dataset == 'jrdbego':
    # JRDB: the 7 val scenes of the baselines; EgoGroups: the geo180 val clips of the baselines (never trained on)
    val_scenarios = ({e['scenario_idx'] for e in examples if e['source'] == 'jrdb' and e['sequence_name'] in sft_data_utils.JRDB_VAL_SCENES}
                     | {s + JOINT_EGO_OFFSET for s in egogroups_train_data_utils.VAL_SCENARIOS_GEO180})
    train_examples = [e for e in examples if e['scenario_idx'] not in val_scenarios]
    val_examples = [e for e in examples if e['scenario_idx'] in val_scenarios]
    print(f"train: JRDB {sum(e['source'] == 'jrdb' for e in train_examples)}, EgoGroups {sum(e['source'] == 'ego' for e in train_examples)}; "
          f"val: JRDB {sum(e['source'] == 'jrdb' for e in val_examples)}, EgoGroups {sum(e['source'] == 'ego' for e in val_examples)}")

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

# --variant idsonly --mode full: SYSTEM_PROMPT_FULL without coordinates; movement_direction comes from the 2D
# box-centre motion (utils.get_movement_direction_2d), same task text as prompts.py's
# vlm_IdentifyGroups_MotionImage_idsonly (prompt_method='p1_visual_only', full setting).
SYSTEM_PROMPT_FULL_IDSONLY = (
    "Given an image of a single video frame with people annotated by bounding boxes and id labels, "
    "compute groups of people who are close together. Use only the visual positions of the labeled "
    "boxes and each person's movement direction to judge proximity -- no numeric coordinates are "
    "provided. Return only non-empty groups. Do not merge distant people into the same group. Do not "
    "hallucinate non-existent person_id.\n\n"
    "You are given an image with each person's bounding box and id label drawn on them, plus "
    "a JSON array of the people in it. Each is an object with keys 'person_id' and "
    "'movement_direction' (their net movement direction in the image across the frames leading up "
    "to this one, or 'stationary').\n"
    "Respond with ONLY a JSON object of the form {\"groups\": [[person_id, ...], ...]} "
    "and no other text. All ids should appear at least once."
)

from social_prompts import SYSTEM_PROMPT_SOCIAL, SYSTEM_PROMPT_SOCIAL_FULL  # --variant p1_social (same text as prompt_method p1_social)

if args.mode == 'full':
    SYSTEM_PROMPT = {'p1_visual': SYSTEM_PROMPT_FULL, 'p1_social': SYSTEM_PROMPT_SOCIAL_FULL}.get(args.variant, SYSTEM_PROMPT_FULL_IDSONLY)
else:
    SYSTEM_PROMPT = {'p1_visual': SYSTEM_PROMPT_P1_VISUAL, 'p1_social': SYSTEM_PROMPT_SOCIAL}.get(args.variant, SYSTEM_PROMPT_IDSONLY)


def build_user_content(example):
    """The only other thing that differs by --variant: p1_visual sends the full detections JSON
    (numeric x,y,z included), idsonly sends just the bare person_ids list (full mode: person_id +
    2D movement_direction). p1_social sends the same as p1_visual."""
    if args.variant in ('p1_visual', 'p1_social'):
        return json.dumps(example['frame_input_data'])
    if args.mode == 'full':
        return json.dumps([{'person_id': int(d['person_id']),
                            'movement_direction': example['movement_2d'].get(d['person_id'], 'stationary')}
                           for d in example['frame_input_data']])
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
if args.dataset == 'jrdbego':  # one eval loss per dataset (eval_jrdb_loss, eval_ego_loss)
    val_dataset = {src: Dataset.from_list([r for r, e in zip(val_records, val_examples) if e['source'] == src])
                   .cast_column('image', HFImage()) for src in ('jrdb', 'ego')}
print(train_dataset, val_dataset)

MODEL_NAME, CKPT_PREFIX = sft_data_utils.BASE_MODELS[args.base]['vlm']
VLM_CLASS = Qwen3VLForConditionalGeneration if args.base == 'qwen3-4b' else Qwen2_5_VLForConditionalGeneration

processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = VLM_CLASS.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16,
).to('cuda')

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias='none',
    # --lm_lora: a regex (PEFT full-matches it against module names) picking the language model's linear layers;
    # the vision tower's MLP reuses the gate/up/down_proj names, so anything under 'visual' is excluded by name
    # --lm_merger_lora: the same regex plus the merger's linear layers (sft_data_utils.LM_MERGER_LORA_TARGETS)
    target_modules=(sft_data_utils.LM_LORA_TARGETS if args.lm_lora
                    else sft_data_utils.LM_MERGER_LORA_TARGETS if args.lm_merger_lora else 'all-linear'),
    task_type='CAUSAL_LM',
)

sft_config = SFTConfig(
    output_dir=f'sft_output/{CKPT_PREFIX}-groups-lora-nodspy{DATASET_SUFFIX}',
    num_train_epochs=3,  # VL model + images -- start here, extend once you've confirmed this works
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    bf16=True,
    completion_only_loss=True,
    packing=False,
    padding_free=False,
    max_length=8192,  # SFTConfig defaults to 1024, but Qwen2.5-VL expands each image into many
                       # placeholder tokens -- these wide 3760x480 JRDB frames alone can exceed
                       # that, so a fixed cap truncates input_ids while the vision tower still
                       # produces features for the full image, desyncing the two ("current
                       # max_length is too short and causes image placeholder tokens ... to be
                       # truncated"). 4096 was still too short for JRDB (~2.3K image tokens + up to ~2.9K
                       # text) and gave NaN losses from batches whose answer was fully truncated.
    logging_steps=10,
    eval_strategy='no' if args.no_eval else 'epoch',
    save_strategy='epoch',
    report_to='none',
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=None if args.no_eval else val_dataset,
    peft_config=lora_config,
    processing_class=processor,
)

trainer.train()

adapter_dir = f'sft_output/{CKPT_PREFIX}-groups-lora-nodspy{DATASET_SUFFIX}'
merged_dir = f'sft_output/{CKPT_PREFIX}-groups-merged-nodspy{DATASET_SUFFIX}'

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
eval_model = VLM_CLASS.from_pretrained(merged_dir, torch_dtype=torch.bfloat16).to('cuda')
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
