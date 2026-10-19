"""Inverse of det_txt_to_pkl.py: write an AP_evaluator detection file from an F1 grouping pickle
{clip_idx: {'0'..'4': [[person_id, ...], ...]}}, with exactly the boxes of the ground-truth file (one row per person,
"seq frame x1 y1 x2 y2 group act score", group ids per keyframe). People missing from the pickle become singletons.

    python pkl_to_det_txt.py <pred.pkl> <out.txt> [--gt ../AP_evaluator/out/gt_group_10fps_test.txt]
"""

import argparse
import json
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("pkl")
ap.add_argument("out")
ap.add_argument("--gt", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "../AP_evaluator/out/gt_group_10fps_test.txt"))
args = ap.parse_args()

pred = pickle.load(open(args.pkl, "rb"))
tracking = json.load(open(os.path.expanduser("~/tracking_annotations.json")))
rows, missing = [], 0
gid_cache = {}
for line in open(args.gt):
    r = line.split()
    seq, fid = int(r[0]), int(r[1])
    box = [int(float(v)) for v in r[2:6]]
    people = tracking[str(seq + 1)].get(str(fid - 1), [])
    pid = next((p["track_id"] + 1 for p in people if p and [int(v) for v in p["bbox"]] == box), None)
    key = (seq, fid)
    if key not in gid_cache:
        groups = (pred.get(seq) or {}).get(str((fid - 10) // 10), []) if isinstance(pred.get(seq), dict) else []
        gid_cache[key] = ({p: i for i, g in enumerate(groups) for p in g}, [len(groups)])
    lookup, nxt = gid_cache[key]
    if pid is None or pid not in lookup:
        missing += pid is None
        gid = nxt[0]; nxt[0] += 1
    else:
        gid = lookup[pid]
    rows.append(f"{seq} {fid} {' '.join(r[2:6])} {gid} 1 1\n")
open(args.out, "w").writelines(rows)
print(f"{args.out}: {len(rows)} rows" + (f", {missing} GT boxes without a track id" if missing else ""))
