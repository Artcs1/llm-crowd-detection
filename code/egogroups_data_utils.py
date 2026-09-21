"""Ground-truth data loader for SFT/RL training on EgoGroups (gold_SEKAI_900_3) -- the SEKAI
counterpart of sft_data_utils.py's JRDB loader. See that file for the general design this mirrors.

Cross-references F1_evaluator/out/gt_gold_sekai_{2,22,42}.pkl (per-target-frame ground-truth
social-group labels) against gold_SEKAI_900_3/jsons_step5/clip_XXXX.json. Unlike JRDB, there's no
shard-matching needed -- each clip_XXXX.json is one complete, unsharded scenario, and scenario_idx
maps directly to clip number via idx = clip_number - 1 (same convention compute_detections.py /
compute_groupings.py already use for this dataset).

As of this writing, gt_gold_sekai_{2,22,42}.pkl are placeholder stubs (each just {'0': [[1]]}),
not real ground truth -- built anyway per explicit request, so build_sft_examples() is ready to
use as soon as those pkls are properly populated. Expect near-zero examples until then.
"""

import json
import os
import pickle
import random

from utils import get_allframes_bboxes, get_movement_direction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEKAI_GOLD_DIR = '/lp-dev/jmurrugarral/gold_SEKAI_900_3'
JSONS_GOLD_DIR = os.path.join(SEKAI_GOLD_DIR, 'jsons_step5')
VIDEOS_FRAMES_DIR = os.path.join(SEKAI_GOLD_DIR, 'videos_frames')

GT_PKL_PATHS = {
    2: os.path.join(BASE_DIR, 'F1_evaluator', 'out', 'gt_gold_sekai_2.pkl'),
    22: os.path.join(BASE_DIR, 'F1_evaluator', 'out', 'gt_gold_sekai_22.pkl'),
    42: os.path.join(BASE_DIR, 'F1_evaluator', 'out', 'gt_gold_sekai_42.pkl'),
}
TARGET_FRAME_IDS = tuple(GT_PKL_PATHS.keys())
DEPTH_METHOD = 'wilddet_3D'


def load_gt(target_frame_id):
    with open(GT_PKL_PATHS[target_frame_id], 'rb') as f:
        return pickle.load(f)


def clip_path_for_scenario(scenario_idx):
    """scenario_idx -> clip_XXXX.json path (idx = clip_number - 1)."""
    return os.path.join(JSONS_GOLD_DIR, f'clip_{scenario_idx + 1:04d}.json')


def _extract_groups(entry):
    """gt.pkl entries are normally shaped {scenario_idx: {'0': groups}} (double-nested, matching
    both gt_sekai_*.pkl and JRDB's gt.pkl convention). As of this writing, gt_gold_sekai_*.pkl is
    instead a flat single-level placeholder stub, {'0': groups} at the top level with no
    per-scenario wrapping at all -- handle both shapes so this loader keeps working once those
    pkls are replaced with properly-generated (presumably double-nested) ground truth."""
    if isinstance(entry, dict):
        return entry.get('0')
    return entry


def _get_xyz(det, depth_method=DEPTH_METHOD):
    """(x, y, z) for a detection, falling back to a random value if depth_method is missing --
    dead code for gold_SEKAI_900_3 today (wilddet_3D verified always present), kept for
    robustness / future datasets."""
    if depth_method in det:
        x, y, z = det[depth_method]
    else:
        x, y, z = random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-0.3, 0.3)
    return round(x, 4), round(y, 4), round(z, 4)


def _is_all_singleton(gt_groups):
    """True if every group has size 1 -- people are present and grouped, but there's zero real
    (non-singleton) grouping signal in the frame at all."""
    return all(len(g) == 1 for g in gt_groups)


def build_sft_examples(target_frame_ids=TARGET_FRAME_IDS, require_image=True, limit=None,
                        exclude_all_singleton=False):
    """Yield dicts: scenario_idx, clip_name, clip_path, target_frame_id, frame_id (alias),
    frame_input_data, personid2bbox, gt_groups (list[list[int]]), image_path (or None).

    exclude_all_singleton=True additionally drops examples where every group is a singleton (see
    _is_all_singleton) -- used by the 'egogroups-subset' dataset option, on top of the empty-groups
    filter below which always applies regardless of this flag."""
    examples = []
    for target_frame_id in target_frame_ids:
        gt = load_gt(target_frame_id)

        for raw_scenario_idx, entry in gt.items():
            gt_groups = _extract_groups(entry)
            if not gt_groups:
                continue
            if exclude_all_singleton and _is_all_singleton(gt_groups):
                continue
            scenario_idx = int(raw_scenario_idx)

            clip_path = clip_path_for_scenario(scenario_idx)
            if not os.path.exists(clip_path):
                continue
            with open(clip_path) as f:
                data = json.load(f)

            frame = next((fr for fr in data['frames'] if fr['frame_id'] == target_frame_id), None)
            if frame is None:
                continue

            frame_input_data = []
            personid2bbox = {}
            for det in frame['detections']:
                personid2bbox[det['track_id']] = det['bbox']
                x, y, z = _get_xyz(det)
                frame_input_data.append({'person_id': det['track_id'], 'x': x, 'y': y, 'z': z})
            if not frame_input_data:
                continue

            clip_name = os.path.basename(clip_path)[:-len('.json')]
            image_path = os.path.join(VIDEOS_FRAMES_DIR, clip_name, f'{target_frame_id:05d}.jpeg')
            if not os.path.exists(image_path):
                if require_image:
                    continue
                image_path = None

            examples.append({
                'scenario_idx': scenario_idx,
                'clip_name': clip_name,
                'clip_path': clip_path,
                'target_frame_id': target_frame_id,
                'frame_id': target_frame_id,
                'frame_input_data': frame_input_data,
                'personid2bbox': personid2bbox,
                'gt_groups': [[int(p) for p in group] for group in gt_groups],
                'image_path': image_path,
            })

            if limit is not None and len(examples) >= limit:
                return examples

    return examples


def build_sft_examples_full(target_frame_ids=TARGET_FRAME_IDS, require_image=True, limit=None,
                             exclude_all_singleton=False):
    """Full-setting counterpart of build_sft_examples(): frame_input_data is the target frame's
    detections enriched with a 'movement_direction' label computed from every earlier frame in the
    clip (via get_allframes_bboxes + get_movement_direction) -- the same input full_inference /
    IdentifyGroups_AllFrames build for mode='llm', prompt_method='p1' in the 'full' setting (see
    utils.py's full_inference, args.setting == 'full' branch). GT matching by clip/target-frame is
    unchanged from build_sft_examples(); only the input construction differs.

    exclude_all_singleton=True additionally drops examples where every group is a singleton (see
    _is_all_singleton) -- used by the 'egogroups-subset' dataset option, on top of the empty-groups
    filter below which always applies regardless of this flag.

    Note: unlike build_sft_examples()'s per-detection _get_xyz() (which falls back to a random
    (x,y,z) if DEPTH_METHOD is missing), get_allframes_bboxes() has no such fallback and will raise
    KeyError on a detection missing DEPTH_METHOD -- a non-issue today since wilddet_3D is verified
    present on every detection in gold_SEKAI_900_3 (see egogroups_data_utils module docstring/
    CLAUDE.md), but worth knowing if this loader is ever pointed at a different dataset."""
    examples = []
    for target_frame_id in target_frame_ids:
        gt = load_gt(target_frame_id)

        for raw_scenario_idx, entry in gt.items():
            gt_groups = _extract_groups(entry)
            if not gt_groups:
                continue
            if exclude_all_singleton and _is_all_singleton(gt_groups):
                continue
            scenario_idx = int(raw_scenario_idx)

            clip_path = clip_path_for_scenario(scenario_idx)
            if not os.path.exists(clip_path):
                continue
            with open(clip_path) as f:
                data = json.load(f)

            all_frames, bboxes = get_allframes_bboxes(
                data, use_direction=False, depth_method=DEPTH_METHOD, prompt_method='p1',
            )
            if target_frame_id > len(all_frames) or not all_frames[target_frame_id - 1]:
                continue

            movement = get_movement_direction(all_frames, target_frame_id)
            frame_input_data = [
                {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
                for det in all_frames[target_frame_id - 1]
            ]
            personid2bbox = bboxes[target_frame_id - 1]

            clip_name = os.path.basename(clip_path)[:-len('.json')]
            image_path = os.path.join(VIDEOS_FRAMES_DIR, clip_name, f'{target_frame_id:05d}.jpeg')
            if not os.path.exists(image_path):
                if require_image:
                    continue
                image_path = None

            examples.append({
                'scenario_idx': scenario_idx,
                'clip_name': clip_name,
                'clip_path': clip_path,
                'target_frame_id': target_frame_id,
                'frame_id': target_frame_id,
                'frame_input_data': frame_input_data,
                'personid2bbox': personid2bbox,
                'gt_groups': [[int(p) for p in group] for group in gt_groups],
                'image_path': image_path,
            })

            if limit is not None and len(examples) >= limit:
                return examples

    return examples


def build_sft_examples_full_video(target_frame_ids=TARGET_FRAME_IDS, require_image=True, limit=None,
                                   exclude_all_singleton=False):
    """Video counterpart of build_sft_examples_full(): same GT matching and target-frame
    frame_input_data (movement_direction-enriched), but also returns a 'video_frames' list -- one
    {'frame_id', 'image_path', 'personid2bbox'} entry per frame in the sampled sequence
    full_inference builds for mode='vlm_image' (utils.py): step=3 if target_frame_id in (22, 42)
    else 1, frames 1..target_frame_id, always ending with the target frame itself. Of the three
    EgoGroups target frames, only 22 and 42 get subsampled; target_frame_id=2 always yields just
    [frame 1, frame 2]. Each entry's personid2bbox is that frame's own bounding boxes (not the
    target frame's), matching full_inference's per-frame annotation behavior for
    prompt_method='p1_visual'.

    exclude_all_singleton=True additionally drops examples where every group is a singleton (see
    _is_all_singleton) -- used by the 'egogroups-subset' dataset option, on top of the empty-groups
    filter below which always applies regardless of this flag."""
    examples = []
    for target_frame_id in target_frame_ids:
        gt = load_gt(target_frame_id)

        for raw_scenario_idx, entry in gt.items():
            gt_groups = _extract_groups(entry)
            if not gt_groups:
                continue
            if exclude_all_singleton and _is_all_singleton(gt_groups):
                continue
            scenario_idx = int(raw_scenario_idx)

            clip_path = clip_path_for_scenario(scenario_idx)
            if not os.path.exists(clip_path):
                continue
            with open(clip_path) as f:
                data = json.load(f)

            all_frames, bboxes = get_allframes_bboxes(
                data, use_direction=False, depth_method=DEPTH_METHOD, prompt_method='p1',
            )
            if target_frame_id > len(all_frames) or not all_frames[target_frame_id - 1]:
                continue

            movement = get_movement_direction(all_frames, target_frame_id)
            frame_input_data = [
                {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
                for det in all_frames[target_frame_id - 1]
            ]
            personid2bbox = bboxes[target_frame_id - 1]

            clip_name = os.path.basename(clip_path)[:-len('.json')]
            image_path = os.path.join(VIDEOS_FRAMES_DIR, clip_name, f'{target_frame_id:05d}.jpeg')
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
                if fid > len(bboxes):
                    video_missing_image = True
                    break
                fid_image_path = os.path.join(VIDEOS_FRAMES_DIR, clip_name, f'{fid:05d}.jpeg')
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
                'scenario_idx': scenario_idx,
                'clip_name': clip_name,
                'clip_path': clip_path,
                'target_frame_id': target_frame_id,
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
