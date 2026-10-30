"""Ground-truth data loader for SFT/RL on the egogroups-city split (egogroups_city.py): the same example schema and
builders as egogroups_train_data_utils.py, run over both EgoGroups folders -- EgoGroups_train (clips 901-2700,
gt_group_10fps_train.pkl) and EgoGroups_test (clips 1-900, gt_group_10fps_test.pkl, the same jsons_step4 /
videos_frames layout of 5 ten-frame sub-clips per clip) -- and kept to the clips of the requested city splits.

scenario_idx = clip_number - 1 for every example (one numbering for both folders, as in the city GT files), and each
example carries 'city_split' ('train' / 'val' / 'test').
benchmark='benchmark3' (or any egogroups_city.BENCHMARKS key) uses that division instead; its 'id_val' clips are 'val'.
"""

import os

import egogroups_train_data_utils as T
from egogroups_city import BENCHMARKS

TEST_DIR = os.path.join(os.path.dirname(T.EGOTRAIN_DIR), 'EgoGroups_test')
# folder -> (dataset dir, GT pickle, clip number of scenario_idx 0)
SOURCES = {
    'train': (T.EGOTRAIN_DIR, os.path.join(T.BASE_DIR, 'F1_evaluator', 'out', 'gt_group_10fps_train.pkl'), 901),
    'test': (TEST_DIR, os.path.join(T.BASE_DIR, 'F1_evaluator', 'out', 'gt_group_10fps_test.pkl'), 1),
}
SPLIT_OF = {b: {c: {'id_val': 'val'}.get(s, s) for s, clips in B.items() for c in clips} for b, B in BENCHMARKS.items()}


def _run_on(folder, fn, **kwargs):
    """Run an egogroups_train_data_utils builder on one EgoGroups folder (its module-level paths swapped in)."""
    names = ('JSONS_TRAIN_DIR', 'VIDEOS_FRAMES_DIR', 'GT_PKL_PATH', 'BASE_CLIP_OFFSET')
    saved = {n: getattr(T, n) for n in names}
    root, gt, offset = SOURCES[folder]
    T.JSONS_TRAIN_DIR, T.VIDEOS_FRAMES_DIR = os.path.join(root, 'jsons_step4'), os.path.join(root, 'videos_frames')
    T.GT_PKL_PATH, T.BASE_CLIP_OFFSET = gt, offset
    try:
        return fn(**kwargs), offset
    finally:
        for n, v in saved.items():
            setattr(T, n, v)


def _build(fn, splits, benchmark='city', **kwargs):
    examples = []
    for folder in ('test', 'train'):
        exs, offset = _run_on(folder, fn, **kwargs)
        for e in exs:
            clip = e['scenario_idx'] + offset
            split = SPLIT_OF[benchmark].get(clip)  # None: a clip outside the division (k05: the other benchmark3 train clips)
            if split in splits:
                examples.append(dict(e, scenario_idx=clip - 1, city_split=split))
    return examples


def build_sft_examples(splits=('train', 'val'), benchmark='city', **kwargs):
    return _build(T.build_sft_examples, splits, benchmark, **kwargs)


def build_sft_examples_full(splits=('train', 'val'), benchmark='city', **kwargs):
    return _build(T.build_sft_examples_full, splits, benchmark, **kwargs)


# JRDB-style 3D (joint JRDB + benchmark3 training, --dataset jrdbego-benchmark3): every detany_3D position (camera frame:
# x right, y down, z forward) rewritten in the JRDB robot frame (x forward, y left, z up), X = z, Y = -x, Z = -y + 0.89,
# exactly as scripts/make_egogroups_jrdbstyle.py does for the *_jrdbstyle test folders (a rotation plus a shift, so
# distances are unchanged). Full mode: movement_direction is recomputed from the converted frames (JRDB ground plane).
JRDB_H0 = 0.89


def _to_jrdb(d):
    return {**d, 'x': round(d['z'], 4), 'y': round(-d['x'], 4), 'z': round(-d['y'] + JRDB_H0, 4)}


def to_jrdb_style(example, mode='single'):
    if mode == 'single':
        return dict(example, frame_input_data=[_to_jrdb(d) for d in example['frame_input_data']])
    import json
    from utils import get_allframes_bboxes, get_movement_direction
    with open(example['clip_path']) as f:
        data = json.load(f)
    for fr in data['frames']:
        for p in fr['detections']:
            x, y, z = p['detany_3D']
            p['detany_3D'] = [z, -x, -y + JRDB_H0]
    all_frames, _ = get_allframes_bboxes(data, use_direction=False, depth_method='detany_3D', prompt_method='p1')
    movement = get_movement_direction(all_frames, example['target_frame_id'])
    return dict(example, frame_input_data=[{**_to_jrdb(d), 'movement_direction': movement.get(d['person_id'], 'stationary')}
                                           for d in example['frame_input_data']])
