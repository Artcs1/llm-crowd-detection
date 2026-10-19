"""Convert an EgoGroups AP_evaluator detection file ("seq frame x1 y1 x2 y2 group act score", frame 10..50) into the
F1 evaluator's grouping pickle {clip_idx: {'0'..'4': [[person_id, ...], ...]}} with person_id = track_id + 1 (the ids of
gt_group_10fps_test.pkl). Boxes are mapped back to people through ~/tracking_annotations.json, which is where every
method's EgoGroups boxes come from.

    python det_txt_to_pkl.py <det.txt> <out.pkl> [--split test|train] [--hungarian]

--hungarian: for methods that predict their own boxes (VLM baseline1), match them to the annotated people of the
keyframe one-to-one by IoU (Hungarian, IoU >= 0.5, as in generate_ids_for_vlm_only.py); unmatched boxes are dropped.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("det")
ap.add_argument("out")
ap.add_argument("--split", default="test", choices=["test", "train"])
ap.add_argument("--hungarian", action="store_true")
args = ap.parse_args()

offset = 1 if args.split == "test" else 901  # clip number = seq + offset
tracking = json.load(open(os.path.expanduser("~/tracking_annotations.json")))
groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
missing = 0
if args.hungarian:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def iou(a, b):
        w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
        inter = max(0, w) * max(0, h)
        union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
        return inter / union if union > 0 else 0.0

    frames = defaultdict(list)
    for line in open(args.det):
        r = line.split()
        frames[int(r[0]), int(r[1])].append(([int(float(v)) for v in r[2:6]], r[6]))
    for (seq, fid), dets in frames.items():
        people = [p for p in tracking[str(seq + offset)].get(str(fid - 1), []) if p]
        if not people:
            missing += len(dets)
            continue
        m = np.array([[iou(b, p["bbox"]) for p in people] for b, _ in dets])
        rows, cols = linear_sum_assignment(-m)
        for i, j in zip(rows, cols):
            if m[i, j] >= 0.5:
                groups[seq][str((fid - 10) // 10)][dets[i][1]].append(people[j]["track_id"] + 1)
        missing += len(dets) - sum(m[i, j] >= 0.5 for i, j in zip(rows, cols))
for line in [] if args.hungarian else open(args.det):
    r = line.split()
    seq, fid = int(r[0]), int(r[1])
    box = [int(float(v)) for v in r[2:6]]
    people = tracking[str(seq + offset)].get(str(fid - 1), [])
    pid = next((p["track_id"] + 1 for p in people if p and [int(v) for v in p["bbox"]] == box), None)
    if pid is None:
        missing += 1
        continue
    groups[seq][str((fid - 10) // 10)][r[6]].append(pid)

pred = {seq: {k: [sorted(g) for g in gs.values()] for k, gs in frames.items()} for seq, frames in groups.items()}
pickle.dump(pred, open(args.out, "wb"))
print(f"{args.out}: {len(pred)} clips" + (f", {missing} boxes not matched" if missing else ""))
