"""Pairwise-judgment training data for a MINGLE-style ("are these two people grouped?") model, built from JRDB train
ground truth -- the JRDB counterpart of mingle_train_data_utils.py (EgoGroups_train), with the same interface
(DATA_ROOT, VAL_SCENARIOS, build_pairwise_examples) so mingle_sft_training.py --dataset jrdb can swap it in.

Source data (same as sft_data_utils.py, the JRDB SFT loader):
  - GT: F1_evaluator/out/gt_train.pkl, {scenario_idx (0-26): {frame_key: [[track_ids...], ...]}}; scenario_idx i is the
    i-th sorted sequence name of JRDB_train_fixed_gold/jsons_gold.
  - Boxes: JRDB_train_fixed_gold/jsons_gold/<sequence>_<frame_key:05d>.json, target frame 15 (the last of the
    15-frame shard). GT frame_key k is shard k of its sequence: for all 1705 shards the target-frame track ids equal
    the GT track ids of that frame_key exactly (checked); the 141 GT frames past a sequence's last shard are skipped.
    (sft_data_utils.find_matching_shard's "first shard whose ids contain the GT ids" rule picks a different shard for
    940 of the 1773 labeled frames, so it is not used here.)
  - Image: JRDB_train_fixed_gold/videos_frames/<shard>/00015.jpeg
  - Depth: EVP/depth/gold_jrdb_train_depth_res/<shard>/00015.png (EVP/depth/run_depth_jrdb_train.py, same EVP
    preprocessing as the JRDB test depth gold_jrdb_depth_res)

Only people that appear in a GT group (singletons included) are paired, so an unlabeled detection is never used as a
"No". Negatives are sampled per frame exactly as in mingle_train_data_utils.py (half closest
by pixel-center distance, half random; fractional neg_per_pos rounded stochastically). Boxes/track ids are sorted by
(x1, y1) before pairing, the order SidewalkBalletImage uses internally.

Validation: the 7 JRDB val scenes of the baselines (sft_data_utils.JRDB_VAL_SCENES, DVT split) are held out; the other
20 are train.
"""

import glob
import json
import os
import pickle
import random

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(CODE_DIR, '..', '..'))
JRDB_TRAIN_DIR = os.path.join(DATA_ROOT, 'JRDB_train_fixed_gold')
JSONS_DIR = os.path.join(JRDB_TRAIN_DIR, 'jsons_gold')
VIDEOS_FRAMES_DIR = os.path.join(JRDB_TRAIN_DIR, 'videos_frames')
DEPTH_DIR = os.path.join(DATA_ROOT, 'EVP', 'depth', 'gold_jrdb_train_depth_res')
GT_PKL_PATH = os.path.join(CODE_DIR, 'F1_evaluator', 'out', 'gt_train.pkl')
TARGET_FRAME = 15

# Same list as sft_data_utils.JRDB_VAL_SCENES (kept literal here so this module needs nothing from utils.py / dspy).
JRDB_VAL_SCENES = ['clark-center-2019-02-28_1', 'gates-ai-lab-2019-02-08_0', 'huang-2-2019-01-25_0', 'meyer-green-2019-03-16_0',
                   'nvidia-aud-2019-04-18_0', 'tressider-2019-03-16_1', 'tressider-2019-04-26_2']
SCENARIO_NAMES = sorted({fn.rsplit('_', 1)[0] for fn in os.listdir(JSONS_DIR) if fn.endswith('.json')})
VAL_SCENARIOS = {SCENARIO_NAMES.index(s) for s in JRDB_VAL_SCENES}


def load_gt():
    with open(GT_PKL_PATH, 'rb') as f:
        return pickle.load(f)


def _box_center(box):
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def _load_sequence_shards(sequence_name):
    """[(shard_name, {track_id: box})] for every shard of a sequence, from its target frame."""
    shards = []
    for path in sorted(glob.glob(os.path.join(JSONS_DIR, f'{sequence_name}_*.json'))):
        with open(path) as f:
            data = json.load(f)
        frame = next((fr for fr in data['frames'] if fr['frame_id'] == TARGET_FRAME), None)
        if frame is None:
            continue
        tid2box = {str(d['track_id']): [int(v) for v in d['bbox']] for d in frame['detections']}
        shards.append((os.path.basename(path)[:-len('.json')], tid2box))
    return shards


def build_pairwise_examples(neg_per_pos=3, require_depth=True, limit=None, seed=42):
    """Yield dicts: clip_name (= shard name), image_path, depth_path, box1, box2, tid1, tid2, label ('Yes'/'No'),
    scenario_idx -- the same schema as mingle_train_data_utils.build_pairwise_examples."""
    rng = random.Random(seed)
    gt = load_gt()
    examples = []
    for scenario_idx in sorted(gt):
        if scenario_idx >= len(SCENARIO_NAMES):
            continue
        shards = dict(_load_sequence_shards(SCENARIO_NAMES[scenario_idx]))
        for frame_key in sorted(gt[scenario_idx], key=int):
            groups = [[str(p) for p in g] for g in gt[scenario_idx][frame_key] if g]
            if not groups:
                continue
            gt_ids = {p for g in groups for p in g}
            shard = f'{SCENARIO_NAMES[scenario_idx]}_{int(frame_key):05d}'
            tid2box = shards.get(shard)
            if tid2box is None or not gt_ids <= set(tid2box):
                continue

            image_path = os.path.join(VIDEOS_FRAMES_DIR, shard, f'{TARGET_FRAME:05d}.jpeg')
            depth_path = os.path.join(DEPTH_DIR, shard, f'{TARGET_FRAME:05d}.png')
            if not os.path.exists(image_path):
                continue
            if require_depth and not os.path.exists(depth_path):
                continue

            tids = sorted(gt_ids, key=lambda t: (tid2box[t][0], tid2box[t][1]))
            boxes = [tid2box[t] for t in tids]
            if len(boxes) < 2:
                continue
            tid2idx = {t: i for i, t in enumerate(tids)}

            positive_pairs = set()
            for group in groups:
                members = sorted(tid2idx[p] for p in group)
                for a in range(len(members)):
                    for b in range(a + 1, len(members)):
                        positive_pairs.add((members[a], members[b]))
            if not positive_pairs:
                continue

            all_pairs = [(i, j) for i in range(len(boxes)) for j in range(i + 1, len(boxes))]
            negative_pairs = [p for p in all_pairs if p not in positive_pairs]
            target = neg_per_pos * len(positive_pairs)
            target = int(target) + (rng.random() < target - int(target))
            n_neg_wanted = min(len(negative_pairs), target)
            if n_neg_wanted > 0:
                centers = [_box_center(b) for b in boxes]
                dist = lambda p: (centers[p[0]][0] - centers[p[1]][0]) ** 2 + (centers[p[0]][1] - centers[p[1]][1]) ** 2
                negative_pairs.sort(key=dist)
                n_hard = n_neg_wanted // 2
                hard, rest = negative_pairs[:n_hard], negative_pairs[n_hard:]
                sampled_negatives = hard + rng.sample(rest, min(n_neg_wanted - len(hard), len(rest)))
            else:
                sampled_negatives = []

            for (i, j), label in [(p, 'Yes') for p in sorted(positive_pairs)] + [(p, 'No') for p in sampled_negatives]:
                examples.append({
                    'scenario_idx': scenario_idx,
                    'clip_name': shard,
                    'image_path': image_path,
                    'depth_path': depth_path,
                    'box1': boxes[i],
                    'box2': boxes[j],
                    'tid1': int(tids[i]),
                    'tid2': int(tids[j]),
                    'label': label,
                })
                if limit is not None and len(examples) >= limit:
                    return examples
    return examples
