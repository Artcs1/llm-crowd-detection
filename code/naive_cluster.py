"""Naive, non-LLM baseline: DBSCAN clustering directly on people's 3D positions, producing the
same {'groups': [...], 'id_tobbox': {...}} prediction shape the LLM/VLM pipeline writes via
save_frame -- so it can be run through the unmodified compute_groupings.py/compute_detections.py
-> F1_evaluator/AP_evaluator, exactly like any LLM/VLM method.

DBSCAN(eps=threshold, min_samples=1) is mathematically identical to connected components on a
distance graph thresholded at `eps` (min_samples=1 means nothing is ever labeled noise) -- i.e.
exactly the single-linkage/transitive-closure algorithm prompts.py's own IdentifyGroups signature
describes in words ("compute all pairwise distances... group if below a threshold... transitively").
min_samples > 1 is available as a stricter density knob if pure single-linkage chaining turns out
to be too permissive.

2D (x,y) distance is the default rather than 3D: depth_method's z-axis (monocular depth) is the
noisiest estimated coordinate, and group proximity (F-formations) is fundamentally a ground-plane
phenomenon -- two people at different heights/poses but the same floor position are still a group.
--distance_dims 3 is available for comparison.
"""

import argparse
import json
import os

import numpy as np
from sklearn.cluster import DBSCAN

from utils import get_frame_bboxes, save_frame


def parse_args():
    parser = argparse.ArgumentParser(description="Naive DBSCAN-on-3D-position baseline")
    parser.add_argument('filename', type=str, help="Folder of scenario JSON files")
    parser.add_argument('--frame_id', type=int, default=None,
        help="Target frame id. If omitted, auto-detected per scenario as its own last frame "
             "(len(data['frames'])) -- matches EgoGroups_train/test's one-target-frame-per-file "
             "convention, so no external frame-id loop is needed for those datasets.")
    parser.add_argument('--depth_method', type=str, default='detany_3D')
    parser.add_argument('--threshold', type=float, default=1.2,
        help="DBSCAN eps, in the same units as the position field (e.g. meters).")
    parser.add_argument('--min_samples', type=int, default=1,
        help="DBSCAN min_samples. 1 (default) = pure single-linkage/connected-components; "
             "higher values require denser local support before merging two people, which "
             "reduces single-link chaining but can also emit -1 'noise' labels -- these are "
             "folded into their own singleton groups (matching how ungrouped people are always "
             "treated elsewhere in this pipeline, e.g. compute_detections.py).")
    parser.add_argument('--distance_dims', type=int, choices=[2, 3], default=2,
        help="2 (default): cluster on (x,y) only. 3: also use z (depth_method's noisiest axis).")
    parser.add_argument('--frame_path', type=str, required=True,
        help="Root folder of per-scenario video frame images (only used to build the results "
             "path / optional saved visualization, same convention as batch_fetch_groups.py).")
    parser.add_argument('--model', type=str, default='baseline/NaiveCluster',
        help="Must contain '/' -- save_frame() uses model.split('/')[1] for the results path.")
    parser.add_argument('--vlm_mode', type=str, default='llm',
        help="Not an algorithmic input -- only exists to match compute_groupings.py/"
             "compute_detections.py's required --vlm_mode path segment. Left as 'llm' (an "
             "already-valid choice in both scripts) rather than adding a new value.")
    parser.add_argument('--save_image', action='store_true')
    return parser.parse_args()


def cluster_groups(frame_input_data, threshold, min_samples, distance_dims):
    """frame_input_data: list of {'person_id','x','y','z'} dicts (get_frame_bboxes, prompt_method='p1').
    Returns list[list[person_id]]."""
    person_ids = [det['person_id'] for det in frame_input_data]
    if distance_dims == 2:
        coords = np.array([[det['x'], det['y']] for det in frame_input_data])
    else:
        coords = np.array([[det['x'], det['y'], det['z']] for det in frame_input_data])

    labels = DBSCAN(eps=threshold, min_samples=min_samples, metric='euclidean').fit_predict(coords)

    groups = {}
    next_noise_label = max(labels, default=-1) + 1
    for person_id, label in zip(person_ids, labels):
        if label == -1:
            label = next_noise_label
            next_noise_label += 1
        groups.setdefault(label, []).append(person_id)
    return list(groups.values())


def main():
    args = parse_args()
    if '/' not in args.model:
        raise ValueError(f"--model must contain '/' (e.g. 'baseline/NaiveCluster'), got: {args.model!r}")

    collected_files = [os.path.join(args.filename, f) for f in os.listdir(args.filename)
                        if os.path.isfile(os.path.join(args.filename, f))]
    collected_files.sort()

    for current_file in collected_files:
        try:
            with open(current_file, 'r') as f:
                data = json.load(f)

            fid = args.frame_id if args.frame_id is not None else len(data['frames'])

            frame_input_data, personid2bbox = get_frame_bboxes(
                data, use_direction=False, depth_method=args.depth_method,
                frame_id=fid, prompt_method='p1',
            )
            if not frame_input_data:
                continue

            groups = cluster_groups(frame_input_data, args.threshold, args.min_samples, args.distance_dims)

            output = {'groups': groups, 'id_tobbox': personid2bbox, 'frame_id': fid, 'error': None}

            save_filename = current_file.split('/')[-1][:-len('.json')]
            frame_path = f'{args.frame_path}/{save_filename}/{str(fid).zfill(5)}.jpeg'
            res_path = '../results/predictions/' + args.frame_path.rstrip('/').split('/')[-2] + '/results'

            save_frame(output, personid2bbox, res_path, save_filename, frame_path,
                       args.save_image, args.model, args.vlm_mode, args.depth_method,
                       'naive_cluster', fid)
        except Exception as e:
            print(f'Fail in: {current_file}: {e}')


if __name__ == '__main__':
    main()
