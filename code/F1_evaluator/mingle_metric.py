"""MINGLE group-region metric (Liu et al., AAAI 2026, Sec. "Evaluation"): every group with >= 2 people becomes the
smallest box enclosing its members' person boxes; singletons are ignored. Predicted and GT group boxes of a keyframe
are matched one-to-one, greedily by IoU (the paper does not say how it matches; greedy is our choice).
  P / R / F1: a match with IoU >= 0.5 is a TP; pooled over all keyframes (micro), or with --per_image averaged
              over keyframes (macro): P and F1 over keyframes with >= 1 GT or predicted group (a keyframe with
              no GT group but predicted groups scores P = F1 = 0), R and mIoU over keyframes with >= 1 GT group
              (undefined otherwise). --gt_frames_only restores the older variant (all four over GT keyframes).
  mIoU:       mean over GT groups of the IoU of their matched prediction (0 when unmatched).
Person boxes come from ~/tracking_annotations.json (keyframe k = frame 10k+9, person id = track_id + 1).

    python mingle_metric.py --dets ../../results/grouping_files/EgoGroups_test_10 --gt out/gt_group_10fps_test.pkl \
                            --csv ../../results/mingle_metric_all.csv [--match DVT3,PAR,...]
"""

import argparse
import glob
import json
import os
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("--dets", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--match", default="", help="comma-separated substrings; only pickles containing one of them")
ap.add_argument("--iou", type=float, default=0.5)
ap.add_argument("--per_image", action="store_true",
                help="average mIoU / P / R / F1 per keyframe instead of pooling counts over all keyframes")
ap.add_argument("--gt_frames_only", action="store_true",
                help="with --per_image: skip keyframes without GT groups (false-positive groups there are ignored)")
ap.add_argument("--density", choices=["scattered", "moderate", "crowded"],
                help="only clips of this density (the 'density' field of each clip JSON, as in AP_evaluator/JRDB_eval.py)")
ap.add_argument("--clips", default="/data/paola-shared/jmurrugarral/EgoGroups_test/jsons_step1")
args = ap.parse_args()

gt = pickle.load(open(args.gt, "rb"))
if args.density:
    gt = {c: v for c, v in gt.items()
          if json.load(open(glob.glob(os.path.join(args.clips, "clip_%04d_*.json" % (c + 1)))[0]))["density"] == args.density}
tracking = json.load(open(os.path.expanduser("~/tracking_annotations.json")))
BOX = {}  # (clip_idx, keyframe) -> {person_id: box}
for c in gt:
    for k in gt[c]:
        people = tracking[str(c + 1)].get(str(10 * int(k) + 9), [])
        BOX[c, k] = {p["track_id"] + 1: p["bbox"] for p in people if p}


def group_boxes(groups, boxes):
    out = []
    for g in groups or []:
        b = [boxes[int(p)] for p in g if int(p) in boxes]
        if len(b) >= 2:
            out.append((min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)))
    return out


def iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def score(pred):
    tp = n_pred = n_gt = 0
    iou_sum = 0.0
    per = []  # per-keyframe (mIoU, P, R, F1) for --per_image; None = undefined (no GT group)
    for c in gt:
        pc = pred.get(c) if isinstance(pred.get(c), dict) else {}
        for k in gt[c]:
            tp0, iou0 = tp, iou_sum
            G = group_boxes(gt[c][k], BOX[c, k])
            Pd = group_boxes(pc.get(k, []), BOX[c, k])
            n_gt += len(G)
            n_pred += len(Pd)
            pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(G) for j, p in enumerate(Pd)), reverse=True)
            used_g, used_p = set(), set()
            for v, i, j in pairs:
                if v <= 0:
                    break
                if i in used_g or j in used_p:
                    continue
                used_g.add(i)
                used_p.add(j)
                iou_sum += v
                tp += v >= args.iou
            if G:
                t = tp - tp0
                pi, ri = (t / len(Pd) if Pd else 0.0), t / len(G)
                per.append(((iou_sum - iou0) / len(G), pi, ri, 2 * pi * ri / (pi + ri) if pi + ri else 0.0))
            elif Pd and not args.gt_frames_only:
                per.append((None, 0.0, None, 0.0))  # only false-positive groups
    if args.per_image:
        out = []
        for i in range(4):
            v = [x[i] for x in per if x[i] is not None]
            out.append(sum(v) / len(v))
        return tuple(out) + (n_pred, n_gt)
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return iou_sum / n_gt, p, r, f, n_pred, n_gt


rows = []
for path in sorted(glob.glob(os.path.join(args.dets, "*.pkl"))):
    name = os.path.basename(path)
    if args.match and not any(m in name for m in args.match.split(",")):
        continue
    m = score(pickle.load(open(path, "rb")))
    rows.append((name, m))
    print(f"{name[:75]:75s} mIoU {m[0]*100:5.2f} P {m[1]*100:5.2f} R {m[2]*100:5.2f} F1 {m[3]*100:5.2f}", flush=True)

rows.sort(key=lambda r: -r[1][3])
with open(args.csv, "w") as fo:
    fo.write("name,mIoU,P,R,F1,pred_groups,gt_groups\n")
    for name, m in rows:
        fo.write(name + "," + ",".join("%.2f" % (v * 100) for v in m[:4]) + ",%d,%d\n" % m[4:])
print("wrote", args.csv)
