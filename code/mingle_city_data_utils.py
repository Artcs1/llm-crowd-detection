"""MINGLE pairwise examples for the egogroups-city split (egogroups_city.py): mingle_train_data_utils.py's builder run
over both EgoGroups folders -- EgoGroups_train (clips 901-2700, EVP depth gold_egogroups_train) and EgoGroups_test
(clips 1-900, gold_egogroups_test; same jsons_step4 / videos_frames layout) -- keeping the city train + val clips
(the city test clips are never loaded). scenario_idx = clip_number - 1; VAL_SCENARIOS = the 180 city val clips.
"""

import os

import mingle_train_data_utils as M
from egogroups_city import CITY

DATA_ROOT = M.DATA_ROOT
SOURCES = {  # folder -> (dataset dir, depth dir, GT pickle, clip number of scenario_idx 0)
    'train': (M.EGOTRAIN_DIR, M.DEPTH_DIR, M.GT_PKL_PATH, 901),
    'test': (os.path.join(DATA_ROOT, 'EgoGroups_test'), os.path.join(DATA_ROOT, 'EVP', 'depth', 'gold_egogroups_test'),
             os.path.join(M.CODE_DIR, 'F1_evaluator', 'out', 'gt_group_10fps_test.pkl'), 1),
}
TRAINVAL = set(CITY['train']) | set(CITY['val'])
VAL_SCENARIOS = {c - 1 for c in CITY['val']}


def build_pairwise_examples(neg_per_pos=3, require_depth=True, limit=None, seed=42):
    names = ('JSONS_DIR', 'VIDEOS_FRAMES_DIR', 'DEPTH_DIR', 'GT_PKL_PATH', 'BASE_CLIP_OFFSET')
    saved = {n: getattr(M, n) for n in names}
    examples = []
    try:
        for folder in ('test', 'train'):
            root, depth, gt, offset = SOURCES[folder]
            M.JSONS_DIR, M.VIDEOS_FRAMES_DIR = os.path.join(root, 'jsons_step4'), os.path.join(root, 'videos_frames')
            M.DEPTH_DIR, M.GT_PKL_PATH, M.BASE_CLIP_OFFSET = depth, gt, offset
            for e in M.build_pairwise_examples(neg_per_pos=neg_per_pos, require_depth=require_depth, limit=limit, seed=seed):
                clip = e['scenario_idx'] + offset
                if clip in TRAINVAL:
                    examples.append(dict(e, scenario_idx=clip - 1))
    finally:
        for n, v in saved.items():
            setattr(M, n, v)
    return examples
