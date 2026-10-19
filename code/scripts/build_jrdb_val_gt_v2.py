"""JRDB val group GT with the JRDB test rule (v2): on every val keyframe of out/gt_group_jrdb_val.txt (7 val scenes,
frame >= 30 every 15), keep people with no_eval = false whose box lies inside the 3760x480 panorama, all group
difficulties (cluster_stat 1-3); group id = social_group cluster_ID. This is exactly the person set of the
JRDB_val_fixed_gold shards (checked below), as on JRDB test, where out/gt_group.txt and the shards agree.
The old file (cluster_stat 1-2, no_eval kept, boxes past the right edge kept) is left untouched.

    python scripts/build_jrdb_val_gt_v2.py
"""
import glob
import json
import os

ROOT = "/data/paola-shared/jmurrugarral"
AP = os.path.join(ROOT, "llm-crowd-detection/code/AP_evaluator/out")
LAB = os.path.join(ROOT, "JRDB/train_dataset_with_activity/labels/labels_2d_stitched")
SH = os.path.join(ROOT, "JRDB_val_fixed_gold/jsons_gold")
W, H = 3760, 480

scenes = sorted({os.path.basename(f).rsplit("_", 1)[0] for f in glob.glob(f"{SH}/*.json")})
keys = sorted({tuple(map(int, l.split()[:2])) for l in open(f"{AP}/gt_group_jrdb_val.txt")})
out, bad, n = [], 0, 0
for si, fr in keys:
    s = scenes[si]
    rows = []
    for p in json.load(open(f"{LAB}/{s}.json"))["labels"].get(f"{fr:06d}.jpg", []):
        x, y, w, h = p["box"]
        b = (int(x), int(y), int(x + w), int(y + h))
        if p["attributes"]["no_eval"] or b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H:
            continue
        rows.append((b, p["social_group"]["cluster_ID"]))
    shard = json.load(open(f"{SH}/{s}_{fr // 15 - 1:05d}.json"))["frames"][-1]["detections"]
    bad += {b for b, _ in rows} != {tuple(d["bbox"]) for d in shard}
    n += len(rows)
    out += [f"{si} {fr} {b[0]} {b[1]} {b[2]} {b[3]} {g} 1 0" for b, g in rows]
open(f"{AP}/gt_group_jrdb_val_v2.txt", "w").write("\n".join(out) + "\n")
print(f"{len(keys)} keyframes, {n} boxes, keyframes whose person set differs from the shard: {bad}")
