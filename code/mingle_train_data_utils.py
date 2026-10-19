"""Pairwise-judgment training data for a MINGLE-style ("are these two people grouped?") model,
built from EgoGroups_train's own ground truth instead of MINGLE's unreleased 100k-image dataset.

Source data (same as egogroups_train_data_utils.py / run_mingle.py's --dataset egogroups_test path):
  - GT: F1_evaluator/out/gt_group_10fps_train.pkl, {scenario_idx (0-1799): {sub_clip ('0'-'4'):
    groups}}. Track ids in these groups == jsons_step4's last-frame track_ids exactly (verified).
  - Boxes: EgoGroups_train/jsons_step4/clip_XXXX_YYYYY.json, last frame (always local frame 10).
  - Image: EgoGroups_train/videos_frames/clip_XXXX_YYYYY/00010.jpeg
  - Depth: EVP/depth/gold_egogroups_train/clip_XXXX_YYYYY/00010.png

Every exhaustive pair in a frame is 35x dominated by negatives (68,462 positive/same-group pairs vs.
2.50M exhaustive pairs across the dataset) -- so negatives are sampled per frame, not exhaustive.

Boxes/track_ids are sorted by (x1, y1) before pairing -- the same fix run_mingle.py's load_frame()
applies for EgoGroups_test: SidewalkBalletImage re-sorts its boxes internally by this same key, so
without pre-sorting here, the box a training example labels and the box the crop/depth code actually
uses at training time could be two different people.
"""

import json
import os
import pickle
import random

# Paths are resolved relative to this file (<root>/llm-crowd-detection/code/), so the same code works under
# any <root> that keeps the layout (/lp-dev/jmurrugarral, /mnt/paola-shared/jmurrugarral, ...).
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
EGOTRAIN_DIR = os.path.join(DATA_ROOT, 'EgoGroups_train')
JSONS_DIR = os.path.join(EGOTRAIN_DIR, 'jsons_step4')
VIDEOS_FRAMES_DIR = os.path.join(EGOTRAIN_DIR, 'videos_frames')
DEPTH_DIR = os.path.join(DATA_ROOT, 'EVP', 'depth', 'gold_egogroups_train')
GT_PKL_PATH = os.path.join(CODE_DIR, 'F1_evaluator', 'out', 'gt_group_10fps_train.pkl')
BASE_CLIP_OFFSET = 901
# Validation set shared with every other EgoGroups method (DVT scene_split.py, PAR, GroupTrans, RTGD/SHGD/ECCV20):
# EgoGroups_train clips are ordered by density (0901-1200 scattered, 1201-1800 moderate, 1801-2700 crowded), and the
# last 60 clips of each density (1141-1200, 1741-1800, 2641-2700) are held out -- 180 clips, 1:1:1 like EgoGroups_test.
VAL_CLIPS = [*range(1141, 1201), *range(1741, 1801), *range(2641, 2701)]
VAL_SCENARIOS = {c - BASE_CLIP_OFFSET for c in VAL_CLIPS}  # scenario_idx convention (same as egogroups_train_data_utils.py)
TARGET_FRAME = 10


def load_gt():
    with open(GT_PKL_PATH, 'rb') as f:
        return pickle.load(f)


def _box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _load_frame(clip_name):
    """(boxes, track_ids), sorted by (x1, y1), or (None, None) if the json is missing."""
    path = os.path.join(JSONS_DIR, f'{clip_name}.json')
    if not os.path.exists(path):
        return None, None
    with open(path) as f:
        dets = json.load(f)['frames'][-1]['detections']
    order = sorted(range(len(dets)), key=lambda i: (dets[i]['bbox'][0], dets[i]['bbox'][1]))
    boxes = [[int(v) for v in dets[i]['bbox']] for i in order]
    tids = [int(dets[i]['track_id']) for i in order]
    return boxes, tids


def build_pairwise_examples(neg_per_pos=3, require_depth=True, limit=None, seed=42):
    """Yield dicts: clip_name, image_path, depth_path, box1, box2, tid1, tid2, label ('Yes'/'No'),
    scenario_idx (for train/val splitting by scenario, same convention as the other loaders).

    neg_per_pos: sampled negatives per frame = min(available_negatives, neg_per_pos * n_positives),
    stochastically rounded when fractional.
    Half of the sampled negatives are the closest-by-pixel-center-distance "hard" negatives (people
    near each other but in different GT groups -- the informative case); the other half are randomly
    sampled from the rest, so the model isn't only ever shown trivially-far-apart negatives.
    """
    rng = random.Random(seed)
    gt = load_gt()
    examples = []

    for scenario_idx, sub_clips in gt.items():
        for sub_clip, groups in sub_clips.items():
            if not groups:
                continue
            clip_name = f'clip_{scenario_idx + BASE_CLIP_OFFSET:04d}_{int(sub_clip):05d}'

            image_path = os.path.join(VIDEOS_FRAMES_DIR, clip_name, f'{TARGET_FRAME:05d}.jpeg')
            depth_path = os.path.join(DEPTH_DIR, clip_name, f'{TARGET_FRAME:05d}.png')
            if not os.path.exists(image_path):
                continue
            if require_depth and not os.path.exists(depth_path):
                continue

            boxes, tids = _load_frame(clip_name)
            if boxes is None or len(boxes) < 2:
                continue
            tid2idx = {t: i for i, t in enumerate(tids)}

            positive_pairs = set()
            for group in groups:
                members = [tid2idx[int(p)] for p in group if int(p) in tid2idx]
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        i, j = sorted((members[a], members[b]))
                        positive_pairs.add((i, j))
            if not positive_pairs:
                continue

            all_pairs = [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
            negative_pairs = [p for p in all_pairs if p not in positive_pairs]

            # Fractional neg_per_pos (e.g. 1.5) is rounded stochastically per frame, so the dataset-wide
            # ratio matches it in expectation instead of every 1-positive frame rounding the same way.
            target = neg_per_pos * len(positive_pairs)
            target = int(target) + (rng.random() < target - int(target))
            n_neg_wanted = min(len(negative_pairs), target)
            if n_neg_wanted > 0 and negative_pairs:
                centers = [_box_center(b) for b in boxes]
                dist = lambda p: (centers[p[0]][0] - centers[p[1]][0]) ** 2 + \
                                  (centers[p[0]][1] - centers[p[1]][1]) ** 2
                negative_pairs.sort(key=dist)
                n_hard = n_neg_wanted // 2
                hard = negative_pairs[:n_hard]
                rest = negative_pairs[n_hard:]
                n_random = n_neg_wanted - len(hard)
                random_sample = rng.sample(rest, min(n_random, len(rest)))
                sampled_negatives = hard + random_sample
            else:
                sampled_negatives = []

            for (i, j), label in [(p, 'Yes') for p in positive_pairs] + \
                                  [(p, 'No') for p in sampled_negatives]:
                examples.append({
                    'scenario_idx': scenario_idx,
                    'clip_name': clip_name,
                    'image_path': image_path,
                    'depth_path': depth_path,
                    'box1': boxes[i],
                    'box2': boxes[j],
                    'tid1': tids[i],
                    'tid2': tids[j],
                    'label': label,
                })
                if limit is not None and len(examples) >= limit:
                    return examples

    return examples
