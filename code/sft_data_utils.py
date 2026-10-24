"""Ground-truth data loader for SFT training, built on real JRDB annotations.

Cross-references `F1_evaluator/out/gt.pkl` (real human-authored social-group labels,
`{scenario_idx: {frame_key: [[track_ids...], ...]}}`, produced by
`AP_evaluator/GT_conversion_script.py`) against the per-frame detections in
`JRDB_fixed_gold/jsons_gold/<sequence>_{shard:05d}.json`, so callers get
(detections [+ image], ground-truth groups) pairs usable directly with the same
`utils.get_frame_bboxes` helper the rest of the pipeline uses.

A gt_train.pkl frame_key k is shard k of its sequence (`<sequence>_{k:05d}.json`): for all 1705
JRDB_train_fixed_gold shards the target-frame track ids equal that frame's GT track ids exactly. The
141 GT frames past a sequence's last shard have no input and are skipped. (Before 2026-10-19 the
loader took the first shard whose track ids merely contained the GT ids, which picked an earlier
shard for 940 of the 1773 labeled frames.)
"""

import glob
import json
import os
import pickle

from utils import get_frame_bboxes, get_allframes_bboxes, get_movement_direction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JRDB_GOLD_DIR = os.path.join(BASE_DIR, '..', '..', 'JRDB_train_fixed_gold')
JSONS_GOLD_DIR = os.path.join(JRDB_GOLD_DIR, 'jsons_gold')
VIDEOS_FRAMES_DIR = os.path.join(JRDB_GOLD_DIR, 'videos_frames')
GT_PKL_PATH = os.path.join(BASE_DIR, 'F1_evaluator', 'out', 'gt_train.pkl')

# JRDB validation scenes of the baselines (DVT split, same list as RTGD/rtgd_data.py JRDB_VAL); the other 20 are train.
# Used by the training scripts with --jrdb_val_split; without it they train on all 27 scenes.
JRDB_VAL_SCENES = ['clark-center-2019-02-28_1', 'gates-ai-lab-2019-02-08_0', 'huang-2-2019-01-25_0', 'meyer-green-2019-03-16_0',
                   'nvidia-aud-2019-04-18_0', 'tressider-2019-03-16_1', 'tressider-2019-04-26_2']

# LoRA targets of the VLM scripts' --lm_lora: the language model's linear layers only (PEFT full-matches a string
# target_modules as a regex). Qwen2.5-VL's vision blocks reuse gate/up/down_proj, so names under 'visual' are excluded.
LM_LORA_TARGETS = r'^(?!.*\bvisual\b).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$'
# --lm_merger_lora: the same language-model layers plus the vision-to-text merger's linear layers; the vision encoder
# blocks (visual.blocks.*) stay frozen. Qwen2.5-VL merger: visual.merger.mlp.{0,2}; Qwen3-VL: visual.merger.linear_fc{1,2}
# and the deepstack mergers visual.deepstack_merger_list.{0,1,2}.linear_fc{1,2}. Qwen2.5-VL's merger.mlp.1 is the GELU
# between the two linears, and PEFT rejects a non-Linear target, so only mlp.0 / mlp.2 are named.
LM_MERGER_LORA_TARGETS = (r'^(?:(?!.*\bvisual\b).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)'
                          r'|.*\bvisual\.(merger|deepstack_merger_list\.\d+)\.(mlp\.[02]|linear_fc\d+))$')

# --base of the four *_nodspy_merged.py training scripts: {base: {'llm'|'vlm': (HF model id, checkpoint-dir prefix)}}.
# 'qwen2.5' keeps every existing sft_output/qwen2.5{,-vl}-7b-* name; 'qwen3-4b' writes sft_output/qwen3{,-vl}-4b-*.
BASE_MODELS = {
    'qwen2.5': {'llm': ('Qwen/Qwen2.5-7B-Instruct', 'qwen2.5-7b'), 'vlm': ('Qwen/Qwen2.5-VL-7B-Instruct', 'qwen2.5-vl-7b')},
    'qwen3-4b': {'llm': ('Qwen/Qwen3-4B-Instruct-2507', 'qwen3-4b'), 'vlm': ('Qwen/Qwen3-VL-4B-Instruct', 'qwen3-vl-4b')},
}

TARGET_FRAME_ID = 15  # every jsons_gold shard is a 15-frame window; frame 15 is the last one


def load_gt():
    with open(GT_PKL_PATH, 'rb') as f:
        return pickle.load(f)


def load_scenario_names():
    """Sorted unique sequence names under jsons_gold/; index i == gt.pkl scenario_idx i
    (validated empirically: per-scenario track-id counts match almost exactly)."""
    names = set()
    for fn in os.listdir(JSONS_GOLD_DIR):
        if fn.endswith('.json'):
            names.add(fn.rsplit('_', 1)[0])
    return sorted(names)


def _load_sequence_shards(sequence_name, target_frame_id=TARGET_FRAME_ID):
    """Load every shard for a sequence once, keeping (shard_path, data, track_ids_at_target_frame)."""
    pattern = os.path.join(JSONS_GOLD_DIR, f'{sequence_name}_*.json')
    shards = []
    for shard_path in sorted(glob.glob(pattern)):
        with open(shard_path) as f:
            data = json.load(f)
        frame = next((fr for fr in data['frames'] if fr['frame_id'] == target_frame_id), None)
        if frame is None:
            continue
        track_ids = {str(d['track_id']) for d in frame['detections']}
        shards.append((shard_path, data, track_ids))
    return shards


def find_matching_shard(shards, gt_groups, sequence_name, frame_key):
    """Return (shard_path, data) of shard `<sequence_name>_{frame_key:05d}`, or (None, None) if that shard
    doesn't exist or its target-frame detections don't contain every track id in gt_groups."""
    gt_ids = {str(x) for group in gt_groups for x in group}
    wanted = os.path.join(JSONS_GOLD_DIR, f'{sequence_name}_{int(frame_key):05d}.json')
    for shard_path, data, track_ids in shards:
        if shard_path == wanted:
            return (shard_path, data) if gt_ids.issubset(track_ids) else (None, None)
    return None, None


def build_sft_examples(scenario_range=None, require_image=True, target_frame_id=TARGET_FRAME_ID, limit=None):
    """Yield dicts: scenario_idx, sequence_name, frame_key, frame_id, frame_input_data,
    personid2bbox, gt_groups (list[list[int]]), image_path (or None)."""
    gt = load_gt()
    scenario_names = load_scenario_names()
    scenario_indices = scenario_range if scenario_range is not None else sorted(gt.keys())

    examples = []
    for idx in scenario_indices:
        if idx not in gt or idx >= len(scenario_names):
            continue
        sequence_name = scenario_names[idx]
        shards = _load_sequence_shards(sequence_name, target_frame_id)

        for frame_key, gt_groups in gt[idx].items():
            if not gt_groups:
                continue

            shard_path, data = find_matching_shard(shards, gt_groups, sequence_name, frame_key)
            if shard_path is None:
                continue

            frame_input_data, personid2bbox = get_frame_bboxes(
                data, use_direction=False, depth_method='3D',
                frame_id=target_frame_id, prompt_method='p1',
            )
            if not frame_input_data:
                continue

            shard_name = os.path.basename(shard_path)[:-len('.json')]
            image_path = os.path.join(VIDEOS_FRAMES_DIR, shard_name, f'{target_frame_id:05d}.jpeg')
            if not os.path.exists(image_path):
                if require_image:
                    continue
                image_path = None

            examples.append({
                'scenario_idx': idx,
                'sequence_name': sequence_name,
                'frame_key': frame_key,
                'shard_path': shard_path,
                'frame_id': target_frame_id,
                'frame_input_data': frame_input_data,
                'personid2bbox': personid2bbox,
                'gt_groups': [[int(p) for p in group] for group in gt_groups],
                'image_path': image_path,
            })

            if limit is not None and len(examples) >= limit:
                return examples

    return examples


def build_sft_examples_full(scenario_range=None, require_image=True, target_frame_id=TARGET_FRAME_ID, limit=None):
    """Full-setting counterpart of build_sft_examples(): frame_input_data is the target frame's
    detections enriched with a 'movement_direction' label computed from every earlier frame in the
    shard (via get_allframes_bboxes + get_movement_direction) -- the same input full_inference /
    IdentifyGroups_AllFrames build for mode='llm', prompt_method='p1' in the 'full' setting
    (see utils.py's full_inference, args.setting == 'full' branch). GT matching by shard/track-id
    is unchanged from build_sft_examples(); only the input construction differs."""
    gt = load_gt()
    scenario_names = load_scenario_names()
    scenario_indices = scenario_range if scenario_range is not None else sorted(gt.keys())

    examples = []
    for idx in scenario_indices:
        if idx not in gt or idx >= len(scenario_names):
            continue
        sequence_name = scenario_names[idx]
        shards = _load_sequence_shards(sequence_name, target_frame_id)

        for frame_key, gt_groups in gt[idx].items():
            if not gt_groups:
                continue

            shard_path, data = find_matching_shard(shards, gt_groups, sequence_name, frame_key)
            if shard_path is None:
                continue

            all_frames, bboxes = get_allframes_bboxes(
                data, use_direction=False, depth_method='3D', prompt_method='p1',
            )
            if target_frame_id > len(all_frames) or not all_frames[target_frame_id - 1]:
                continue

            movement = get_movement_direction(all_frames, target_frame_id)
            frame_input_data = [
                {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
                for det in all_frames[target_frame_id - 1]
            ]
            personid2bbox = bboxes[target_frame_id - 1]

            shard_name = os.path.basename(shard_path)[:-len('.json')]
            image_path = os.path.join(VIDEOS_FRAMES_DIR, shard_name, f'{target_frame_id:05d}.jpeg')
            if not os.path.exists(image_path):
                if require_image:
                    continue
                image_path = None

            examples.append({
                'scenario_idx': idx,
                'sequence_name': sequence_name,
                'frame_key': frame_key,
                'shard_path': shard_path,
                'frame_id': target_frame_id,
                'frame_input_data': frame_input_data,
                'personid2bbox': personid2bbox,
                'gt_groups': [[int(p) for p in group] for group in gt_groups],
                'image_path': image_path,
            })

            if limit is not None and len(examples) >= limit:
                return examples

    return examples


def build_sft_examples_full_video(scenario_range=None, require_image=True, target_frame_id=TARGET_FRAME_ID, limit=None):
    """Video counterpart of build_sft_examples_full(): same GT matching and target-frame
    frame_input_data (movement_direction-enriched), but also returns a 'video_frames' list --
    one {'frame_id', 'image_path', 'personid2bbox'} entry per frame in the sampled sequence
    full_inference builds for mode='vlm_image' (utils.py): step=3 if target_frame_id in (22, 42)
    else 1, frames 1..target_frame_id, always ending with the target frame itself. For JRDB,
    target_frame_id is always 15 (never 22/42), so this always yields the shard's full 15 frames
    unsampled. Each entry's personid2bbox is that frame's own bounding boxes (not the target
    frame's), matching full_inference's per-frame annotation behavior for prompt_method='p1_visual'."""
    gt = load_gt()
    scenario_names = load_scenario_names()
    scenario_indices = scenario_range if scenario_range is not None else sorted(gt.keys())

    examples = []
    for idx in scenario_indices:
        if idx not in gt or idx >= len(scenario_names):
            continue
        sequence_name = scenario_names[idx]
        shards = _load_sequence_shards(sequence_name, target_frame_id)

        for frame_key, gt_groups in gt[idx].items():
            if not gt_groups:
                continue

            shard_path, data = find_matching_shard(shards, gt_groups, sequence_name, frame_key)
            if shard_path is None:
                continue

            all_frames, bboxes = get_allframes_bboxes(
                data, use_direction=False, depth_method='3D', prompt_method='p1',
            )
            if target_frame_id > len(all_frames) or not all_frames[target_frame_id - 1]:
                continue

            movement = get_movement_direction(all_frames, target_frame_id)
            frame_input_data = [
                {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
                for det in all_frames[target_frame_id - 1]
            ]
            personid2bbox = bboxes[target_frame_id - 1]

            shard_name = os.path.basename(shard_path)[:-len('.json')]
            image_path = os.path.join(VIDEOS_FRAMES_DIR, shard_name, f'{target_frame_id:05d}.jpeg')
            if not os.path.exists(image_path):
                if require_image:
                    continue
                image_path = None

            step = 3 if target_frame_id in (22, 42) else 1
            frame_indices = list(range(1, target_frame_id + 1, step))
            if frame_indices[-1] != target_frame_id:
                frame_indices.append(target_frame_id)

            video_frames = []
            video_missing_image = False
            for fid in frame_indices:
                fid_image_path = os.path.join(VIDEOS_FRAMES_DIR, shard_name, f'{fid:05d}.jpeg')
                if not os.path.exists(fid_image_path):
                    video_missing_image = True
                    break
                video_frames.append({
                    'frame_id': fid,
                    'image_path': fid_image_path,
                    'personid2bbox': bboxes[fid - 1],
                })
            if video_missing_image:
                if require_image:
                    continue
                video_frames = None

            examples.append({
                'scenario_idx': idx,
                'sequence_name': sequence_name,
                'frame_key': frame_key,
                'shard_path': shard_path,
                'frame_id': target_frame_id,
                'frame_input_data': frame_input_data,
                'personid2bbox': personid2bbox,
                'gt_groups': [[int(p) for p in group] for group in gt_groups],
                'image_path': image_path,
                'video_frames': video_frames,
            })

            if limit is not None and len(examples) >= limit:
                return examples

    return examples
