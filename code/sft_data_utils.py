"""Ground-truth data loader for SFT training, built on real JRDB annotations.

Cross-references `F1_evaluator/out/gt.pkl` (real human-authored social-group labels,
`{scenario_idx: {frame_key: [[track_ids...], ...]}}`, produced by
`AP_evaluator/GT_conversion_script.py`) against the per-frame detections in
`JRDB_fixed_gold/jsons_gold/<sequence>_{shard:05d}.json`, so callers get
(detections [+ image], ground-truth groups) pairs usable directly with the same
`utils.get_frame_bboxes` helper the rest of the pipeline uses.

There is no fixed formula mapping a gt.pkl frame_key to a shard file/frame_id (verified
empirically), so matching is done by comparing track-id sets instead.
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


def find_matching_shard(shards, gt_groups):
    """Return (shard_path, data) for the first shard whose target-frame detections fully
    contain every track id in gt_groups, or (None, None) if no shard matches."""
    gt_ids = {str(x) for group in gt_groups for x in group}
    for shard_path, data, track_ids in shards:
        if gt_ids.issubset(track_ids):
            return shard_path, data
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

            shard_path, data = find_matching_shard(shards, gt_groups)
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

            shard_path, data = find_matching_shard(shards, gt_groups)
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

            shard_path, data = find_matching_shard(shards, gt_groups)
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
