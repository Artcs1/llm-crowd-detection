"""MINGLE Table-2 style pairwise metric: every pair of GT people on a keyframe is a binary question "same social group?"
(Yes / No). A grouping (pickle) answers Yes when both people are in the same predicted group; people missing from the
prediction are singletons, predicted ids that are not GT people are ignored. P / R / F1 are reported for the Yes and
the No class, pooled over all keyframes, on
  all:      every GT pair (No dominates),
  balanced: all Yes pairs + an equal number of No pairs sampled once (seed 0) and shared by every method.

    python pairwise_metric.py --dets ../../results/grouping_files/EgoGroups_test_10 --gt out/gt_group_10fps_test.pkl \
                              --csv ../../results/pairwise_metric_all.csv [--match DVT3,PAR,...]
"""

import argparse
import glob
import os
import pickle

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dets", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--match", default="", help="comma-separated substrings; only pickles containing one of them")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

gt = pickle.load(open(args.gt, "rb"))
FRAMES = []  # (clip, keyframe, person ids, upper-triangle index, GT Yes labels)
for c in gt:
    for k, groups in gt[c].items():
        ids = [int(p) for g in groups for p in g]
        if len(ids) < 2:
            continue
        gid = np.array([i for i, g in enumerate(groups) for _ in g])
        iu = np.triu_indices(len(ids), 1)
        FRAMES.append((c, k, ids, iu, (gid[:, None] == gid[None, :])[iu]))

# Balanced subset: all Yes pairs + as many No pairs, drawn once.
rng = np.random.default_rng(args.seed)
n_yes = sum(int(f[4].sum()) for f in FRAMES)
no_idx = np.concatenate([np.flatnonzero(~f[4]) + off for f, off in
                         zip(FRAMES, np.cumsum([0] + [len(f[4]) for f in FRAMES[:-1]]))])
keep = np.zeros(sum(len(f[4]) for f in FRAMES), bool)
keep[rng.choice(no_idx, size=min(n_yes, len(no_idx)), replace=False)] = True
BAL, off = [], 0
for f in FRAMES:
    BAL.append(keep[off:off + len(f[4])] | f[4])
    off += len(f[4])


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def score(pred):
    cnt = {s: np.zeros(4, np.int64) for s in ("all", "bal")}  # TP, FP, FN, TN for the Yes class
    for (c, k, ids, iu, y), bal in zip(FRAMES, BAL):
        pc = pred.get(c) if isinstance(pred.get(c), dict) else {}
        lookup = {int(p): i for i, g in enumerate(pc.get(k, []) or []) for p in g}
        pid = np.array([lookup.get(p, -1 - n) for n, p in enumerate(ids)])  # missing -> own singleton
        yhat = (pid[:, None] == pid[None, :])[iu]
        for s, m in (("all", slice(None)), ("bal", bal)):
            a, b = y[m], yhat[m]
            cnt[s] += [np.sum(a & b), np.sum(~a & b), np.sum(a & ~b), np.sum(~a & ~b)]
    out = []
    for s in ("all", "bal"):
        tp, fp, fn, tn = cnt[s]
        out += list(prf(tp, fp, fn)) + list(prf(tn, fn, fp))  # Yes P R F1, No P R F1
    return out


cols = [f"{s}_{c}_{m}" for s in ("all", "bal") for c in ("Yes", "No") for m in ("P", "R", "F1")]
print(f"{len(FRAMES)} keyframes, {sum(len(f[4]) for f in FRAMES)} pairs ({n_yes} Yes); balanced {int(keep.sum())} No "
      f"+ {n_yes} Yes")
rows = []
for path in sorted(glob.glob(os.path.join(args.dets, "*.pkl"))):
    name = os.path.basename(path)
    if args.match and not any(m in name for m in args.match.split(",")):
        continue
    v = score(pickle.load(open(path, "rb")))
    rows.append((name, v))
    print(f"{name[:75]:75s} all Yes-F1 {v[2]*100:5.2f} No-F1 {v[5]*100:5.2f} | bal Yes-F1 {v[8]*100:5.2f} "
          f"No-F1 {v[11]*100:5.2f}", flush=True)

rows.sort(key=lambda r: -r[1][2])
with open(args.csv, "w") as fo:
    fo.write("name," + ",".join(cols) + "\n")
    for name, v in rows:
        fo.write(name + "," + ",".join("%.2f" % (x * 100) for x in v) + "\n")
print("wrote", args.csv)
