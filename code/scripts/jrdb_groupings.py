"""JRDB prediction folder -> F1 grouping pickle {seq: {str(shard number): [[person_id, ...], ...]}}, the format of
F1_evaluator/out/gt.pkl. Same rule as compute_groupings.py's JRDB branch (seq = ordinal of the scene among the sorted
predictions, people left out of every group -> singletons), but for any dataset folder (JRDB_fixed_gold,
JRDB_fixed_gold_egostyle) and any prompt method, with integer person ids (naive_cluster writes them as strings).

    python scripts/jrdb_groupings.py <predictions/.../<prompt>/15> <out.pkl>
"""
import glob
import json
import os
import pickle
import sys

files = sorted(glob.glob(os.path.join(sys.argv[1], "*")))
scenes, res = [], {}
for f in files:
    name = os.path.basename(f)
    scene, num = name.rsplit("_", 1)
    if scene not in scenes:
        scenes.append(scene)
    try:
        d = json.load(open(f"{f}/{name}.json"))
    except (OSError, json.JSONDecodeError):
        continue
    p2b, seen, groups = d["id_tobbox"], set(), []
    for g in d["groups"]:
        cur = []
        for p in g:
            if str(p) in p2b and str(p) not in seen:
                seen.add(str(p))
                cur.append(int(p))
        if cur:
            groups.append(cur)
    groups += [[int(k)] for k in p2b if k not in seen]
    res.setdefault(len(scenes) - 1, {})[str(int(num))] = groups
pickle.dump(res, open(sys.argv[2], "wb"))
print(f"{sum(len(v) for v in res.values())} keyframes of {len(files)} folders, {len(res)} scenes -> {sys.argv[2]}")
