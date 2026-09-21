"""Ground-truth data loader for SFT/RL training on EgoGroups_train -- sibling of
egogroups_data_utils.py (gold_SEKAI_900_3) with the same output dict schema, but a materially
different underlying data model:

  - gold_SEKAI_900_3: 900 clips (clip_XXXX.json), one scenario each, 3 fixed target frame ids
    (2, 22, 42) per clip, GT split across 3 separate pickles (one per target frame id).
  - EgoGroups_train: 1800 base clips x 5 sub-clips (clip_XXXX_YYYYY.json, YYYYY in 00000-00004) =
    9000 files, each sub-clip its own independent 10-frame scenario (frame_ids 1-10, target frame
    always the last one, frame_id 10 -- verified constant across every sub-clip sampled). GT is a
    single pickle, F1_evaluator/out/gt_group_10fps_train.pkl, keyed {scenario_idx (0-1799):
    {sub_clip_str ('0'-'4'): groups}}.

scenario_idx -> base clip number is `idx + 901` (EgoGroups_train continues EgoGroups_test's
1-900 clip numbering); sub_clip index maps directly to the filename suffix. Verified exactly at
both ends: idx=0 -> clip_0901_00000.json and idx=1799 -> clip_2700_00000.json, GT person-ids ==
last-frame track_ids in both cases, zero mismatch.

jsons_step4 is used as the single source directory (superset of jsons_step3's fields): it has both
unidepth_3D and detany_3D, while jsons_step3 only has unidepth_3D. jsons_step5 (wilddet_3D, used by
egogroups_data_utils.py) is only 1270/9000 complete for this dataset and can't be used here.

Note: unlike gold_SEKAI_900_3, EgoGroups_train has no jsons_step2_new (track-id-alignment-fix)
stage at all (0 files) -- but this is not a concern here, since GT was verified to match
jsons_step3/4's own track_ids directly (no cross-shard/renumbering step exists in this dataset's
pipeline to misalign in the first place). If EgoGroups_train GT-matching ever looks broken, don't
assume this is the cause without re-checking -- the verified endpoints above are real evidence
against it, not a hunch.
"""

import json
import os
import pickle
import random

from utils import get_allframes_bboxes, get_movement_direction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EGOTRAIN_DIR = '/lp-dev/jmurrugarral/EgoGroups_train'
JSONS_TRAIN_DIR = os.path.join(EGOTRAIN_DIR, 'jsons_step4')
VIDEOS_FRAMES_DIR = os.path.join(EGOTRAIN_DIR, 'videos_frames')
GT_PKL_PATH = os.path.join(BASE_DIR, 'F1_evaluator', 'out', 'gt_group_10fps_train.pkl')
BASE_CLIP_OFFSET = 901  # scenario_idx = base_clip_number - BASE_CLIP_OFFSET
DEFAULT_DEPTH_METHOD = 'detany_3D'


def load_gt():
    with open(GT_PKL_PATH, 'rb') as f:
        return pickle.load(f)


def clip_path_for(scenario_idx, sub_clip):
    """(scenario_idx, sub_clip) -> clip_XXXX_YYYYY.json path."""
    base_clip_number = scenario_idx + BASE_CLIP_OFFSET
    return os.path.join(JSONS_TRAIN_DIR, f'clip_{base_clip_number:04d}_{int(sub_clip):05d}.json')


def _get_xyz(det, depth_method):
    """(x, y, z) for a detection, falling back to a random value if depth_method is missing --
    dead code in practice (unidepth_3D/detany_3D verified always present in jsons_step4), kept for
    robustness."""
    if depth_method in det:
        x, y, z = det[depth_method]
    else:
        x, y, z = random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-0.3, 0.3)
    return round(x, 4), round(y, 4), round(z, 4)


def _is_all_singleton(gt_groups):
    """True if every group has size 1 -- people are present and grouped, but there's zero real
    (non-singleton) grouping signal in the frame at all."""
    return all(len(g) == 1 for g in gt_groups)


def build_sft_examples(depth_method=DEFAULT_DEPTH_METHOD, require_image=True, limit=None,
                        exclude_all_singleton=False):
    """Yield dicts: scenario_idx, clip_name, clip_path, target_frame_id, frame_id (alias),
    frame_input_data, personid2bbox, gt_groups (list[list[int]]), image_path (or None).

    exclude_all_singleton=True additionally drops examples where every group is a singleton (see
    _is_all_singleton) -- on top of the empty-groups filter below which always applies regardless
    of this flag."""
    examples = []
    gt = load_gt()

    for scenario_idx, sub_clips in gt.items():
        for sub_clip, entry in sub_clips.items():
            gt_groups = entry
            if not gt_groups:
                continue
            if exclude_all_singleton and _is_all_singleton(gt_groups):
                continue

            clip_path = clip_path_for(scenario_idx, sub_clip)
            if not os.path.exists(clip_path):
                continue
            with open(clip_path) as f:
                data = json.load(f)

            frame = data['frames'][-1]
            target_frame_id = frame['frame_id']

            frame_input_data = []
            personid2bbox = {}
            for det in frame['detections']:
                personid2bbox[det['track_id']] = det['bbox']
                x, y, z = _get_xyz(det, depth_method)
                frame_input_data.append({'person_id': det['track_id'], 'x': x, 'y': y, 'z': z})
            if not frame_input_data:
                continue

            clip_name = f'clip_{scenario_idx + BASE_CLIP_OFFSET:04d}_{int(sub_clip):05d}'
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


def build_sft_examples_full(depth_method=DEFAULT_DEPTH_METHOD, require_image=True, limit=None,
                             exclude_all_singleton=False):
    """Full-setting counterpart of build_sft_examples(): frame_input_data is the target frame's
    detections enriched with a 'movement_direction' label computed from every earlier frame in the
    sub-clip (via get_allframes_bboxes + get_movement_direction). GT matching by
    (scenario_idx, sub_clip) is unchanged from build_sft_examples(); only the input construction
    differs.

    exclude_all_singleton=True additionally drops examples where every group is a singleton (see
    _is_all_singleton) -- on top of the empty-groups filter below which always applies regardless
    of this flag.

    Note: unlike build_sft_examples()'s per-detection _get_xyz() (which falls back to a random
    (x,y,z) if depth_method is missing), get_allframes_bboxes() has no such fallback and will raise
    KeyError on a detection missing depth_method -- a non-issue in practice since unidepth_3D/
    detany_3D are verified present on every detection in jsons_step4."""
    examples = []
    gt = load_gt()

    for scenario_idx, sub_clips in gt.items():
        for sub_clip, entry in sub_clips.items():
            gt_groups = entry
            if not gt_groups:
                continue
            if exclude_all_singleton and _is_all_singleton(gt_groups):
                continue

            clip_path = clip_path_for(scenario_idx, sub_clip)
            if not os.path.exists(clip_path):
                continue
            with open(clip_path) as f:
                data = json.load(f)

            all_frames, bboxes = get_allframes_bboxes(
                data, use_direction=False, depth_method=depth_method, prompt_method='p1',
            )
            target_frame_id = len(all_frames)
            if not all_frames[target_frame_id - 1]:
                continue

            movement = get_movement_direction(all_frames, target_frame_id)
            frame_input_data = [
                {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
                for det in all_frames[target_frame_id - 1]
            ]
            personid2bbox = bboxes[target_frame_id - 1]

            clip_name = f'clip_{scenario_idx + BASE_CLIP_OFFSET:04d}_{int(sub_clip):05d}'
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


def build_sft_examples_full_video(depth_method=DEFAULT_DEPTH_METHOD, require_image=True, limit=None,
                                   exclude_all_singleton=False):
    """Video counterpart of build_sft_examples_full(): same GT matching and target-frame
    frame_input_data (movement_direction-enriched), but also returns a 'video_frames' list -- one
    {'frame_id', 'image_path', 'personid2bbox'} entry per frame. Unlike gold_SEKAI_900_3 (which
    needed a step=3 subsampling heuristic for target frames as late as 42), every EgoGroups_train
    sub-clip has exactly 10 frames, so all of them (1 through the target frame, i.e. 1-10) are
    included with no subsampling.

    exclude_all_singleton=True additionally drops examples where every group is a singleton (see
    _is_all_singleton) -- on top of the empty-groups filter below which always applies regardless
    of this flag."""
    examples = []
    gt = load_gt()

    for scenario_idx, sub_clips in gt.items():
        for sub_clip, entry in sub_clips.items():
            gt_groups = entry
            if not gt_groups:
                continue
            if exclude_all_singleton and _is_all_singleton(gt_groups):
                continue

            clip_path = clip_path_for(scenario_idx, sub_clip)
            if not os.path.exists(clip_path):
                continue
            with open(clip_path) as f:
                data = json.load(f)

            all_frames, bboxes = get_allframes_bboxes(
                data, use_direction=False, depth_method=depth_method, prompt_method='p1',
            )
            target_frame_id = len(all_frames)
            if not all_frames[target_frame_id - 1]:
                continue

            movement = get_movement_direction(all_frames, target_frame_id)
            frame_input_data = [
                {**det, 'movement_direction': movement.get(det['person_id'], 'stationary')}
                for det in all_frames[target_frame_id - 1]
            ]
            personid2bbox = bboxes[target_frame_id - 1]

            clip_name = f'clip_{scenario_idx + BASE_CLIP_OFFSET:04d}_{int(sub_clip):05d}'
            image_path = os.path.join(VIDEOS_FRAMES_DIR, clip_name, f'{target_frame_id:05d}.jpeg')
            if not os.path.exists(image_path):
                if require_image:
                    continue
                image_path = None

            frame_indices = list(range(1, target_frame_id + 1))

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
