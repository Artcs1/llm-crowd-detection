"""Ground-truth data loader for SFT on EgoGroups_Synth_train -- sibling of
egogroups_train_data_utils.py with the same output dict schema.

Differences from EgoGroups_train:
  - 2928 clips (clip_synth_<job>_NN.json), 24 frames each; jsons_step3 is the source directory.
  - Only exact 'gt_3D' coordinates exist (no detany_3D/unidepth_3D).
  - GT is one pickle, gt_group_train.pkl, keyed {clip_idx: {str(frame_id - 1): groups}} for every
    frame; clip_idx is the position in sorted(jsons_step2/*.json) (see gt_group.py). Only the last
    frame is used as the target, mirroring EgoGroups_train.
"""

import glob
import json
import os
import pickle

from utils import get_allframes_bboxes, get_movement_direction

SYNTH_DIR = '/home/jmurrugarral/EgoGroups_Synth_train'
JSONS_DIR = os.path.join(SYNTH_DIR, 'jsons_step3')
VIDEOS_FRAMES_DIR = os.path.join(SYNTH_DIR, 'videos_frames')
GT_PKL_PATH = os.path.join(SYNTH_DIR, 'gt_group_train.pkl')
DEFAULT_DEPTH_METHOD = 'gt_3D'
FULL_WINDOW = 10  # frames used for movement_direction, matching EgoGroups_train's 10-frame clips

_clip_names = None


def load_gt():
    with open(GT_PKL_PATH, 'rb') as f:
        return pickle.load(f)


def clip_name_for(clip_idx):
    """clip_idx -> clip name (no extension), same ordering gt_group.py used to assign the index."""
    global _clip_names
    if _clip_names is None:
        _clip_names = [os.path.basename(p)[:-5]
                       for p in sorted(glob.glob(os.path.join(SYNTH_DIR, 'jsons_step2', '*.json')))]
    return _clip_names[clip_idx]


def _is_all_singleton(gt_groups):
    return all(len(g) == 1 for g in gt_groups)


def _iter_clips(exclude_all_singleton):
    """Yield (clip_idx, clip_name, clip_path, data, gt_groups) for clips with usable last-frame GT."""
    gt = load_gt()
    for clip_idx, frames_gt in gt.items():
        data_path = os.path.join(JSONS_DIR, f'{clip_name_for(clip_idx)}.json')
        if not os.path.exists(data_path):
            continue
        with open(data_path) as f:
            data = json.load(f)
        gt_groups = frames_gt.get(str(len(data['frames']) - 1))
        if not gt_groups:
            continue
        if exclude_all_singleton and _is_all_singleton(gt_groups):
            continue
        yield clip_idx, clip_name_for(clip_idx), data_path, data, gt_groups


def _image_path(clip_name, target_frame_id, require_image):
    path = os.path.join(VIDEOS_FRAMES_DIR, clip_name, f'{target_frame_id:05d}.jpeg')
    if os.path.exists(path):
        return path
    return None if not require_image else False


def build_sft_examples(depth_method=DEFAULT_DEPTH_METHOD, require_image=True, limit=None,
                       exclude_all_singleton=False):
    """Yield dicts: scenario_idx, clip_name, clip_path, target_frame_id, frame_id (alias),
    frame_input_data, personid2bbox, gt_groups (list[list[int]]), image_path (or None).
    Empty-groups clips are always dropped; exclude_all_singleton additionally drops clips where
    every group is a singleton."""
    examples = []
    for clip_idx, clip_name, clip_path, data, gt_groups in _iter_clips(exclude_all_singleton):
        frame = data['frames'][-1]
        target_frame_id = frame['frame_id']

        frame_input_data = []
        personid2bbox = {}
        for det in frame['detections']:
            personid2bbox[det['track_id']] = det['bbox']
            x, y, z = det[depth_method]
            frame_input_data.append({'person_id': det['track_id'], 'x': round(x, 4),
                                     'y': round(y, 4), 'z': round(z, 4)})
        if not frame_input_data:
            continue

        image_path = _image_path(clip_name, target_frame_id, require_image)
        if image_path is False:
            continue

        examples.append({
            'scenario_idx': clip_idx,
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
            break
    return examples


def build_sft_examples_full(depth_method=DEFAULT_DEPTH_METHOD, require_image=True, limit=None,
                            exclude_all_singleton=False):
    """Full-setting counterpart: frame_input_data is the target frame's detections enriched with
    'movement_direction', computed over the last FULL_WINDOW frames of the clip."""
    examples = []
    for clip_idx, clip_name, clip_path, data, gt_groups in _iter_clips(exclude_all_singleton):
        target_frame_id = data['frames'][-1]['frame_id']
        windowed = {**data, 'frames': data['frames'][-FULL_WINDOW:]}

        all_frames, bboxes = get_allframes_bboxes(
            windowed, use_direction=False, depth_method=depth_method, prompt_method='p1',
        )
        if not all_frames[-1]:
            continue

        movement = get_movement_direction(all_frames, len(all_frames))
        frame_input_data = [
            {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
            for det in all_frames[-1]
        ]

        image_path = _image_path(clip_name, target_frame_id, require_image)
        if image_path is False:
            continue

        examples.append({
            'scenario_idx': clip_idx,
            'clip_name': clip_name,
            'clip_path': clip_path,
            'target_frame_id': target_frame_id,
            'frame_id': target_frame_id,
            'frame_input_data': frame_input_data,
            'personid2bbox': bboxes[-1],
            'gt_groups': [[int(p) for p in group] for group in gt_groups],
            'image_path': image_path,
        })

        if limit is not None and len(examples) >= limit:
            break
    return examples
