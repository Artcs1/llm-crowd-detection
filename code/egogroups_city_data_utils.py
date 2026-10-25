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
            split = SPLIT_OF[benchmark][clip]
            if split in splits:
                examples.append(dict(e, scenario_idx=clip - 1, city_split=split))
    return examples


def build_sft_examples(splits=('train', 'val'), benchmark='city', **kwargs):
    return _build(T.build_sft_examples, splits, benchmark, **kwargs)


def build_sft_examples_full(splits=('train', 'val'), benchmark='city', **kwargs):
    return _build(T.build_sft_examples_full, splits, benchmark, **kwargs)
