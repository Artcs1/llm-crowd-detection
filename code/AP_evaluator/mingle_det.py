"""MINGLE group-region metric (same definition as F1_evaluator/mingle_metric.py) computed directly from AP-format files
("seq frame x1 y1 x2 y2 group act score"), so it works for any dataset (JRDB val/test, EgoGroups) without person ids:
every group with >= 2 people becomes the smallest box around its members' boxes, singletons are ignored, predicted and
GT group boxes of a keyframe are matched one-to-one greedily by IoU, and P / R / F1 (IoU >= 0.5) and mIoU are pooled
over all keyframes. Only predicted people whose box is a GT box of that keyframe are kept (GT drops no-eval and
out-of-image people, as in GT_conversion_script.py), and only keyframes of the GT file are scored.

    python mingle_det.py -g out/gt_group.txt -d det1.txt [det2.txt ...] [--iou 0.5]
"""
import argparse
import os
from collections import defaultdict


def load(path):
    rows = defaultdict(lambda: defaultdict(list))  # (seq, frame) -> group -> [box]
    for line in open(path):
        r = line.split()
        if len(r) < 7:
            continue
        rows[int(r[0]), int(float(r[1]))][r[6]].append(tuple(int(float(v)) for v in r[2:6]))
    return rows


def group_boxes(groups, keep=None):
    out = []
    for boxes in groups.values():
        b = [x for x in boxes if keep is None or x in keep]
        if len(b) >= 2:
            out.append((min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b)))
    return out


def iou(a, b):
    w, h = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def score(gt, det, thr=0.5):
    tp = n_pred = n_gt = 0
    iou_sum = 0.0
    for key, groups in gt.items():
        keep = {b for bs in groups.values() for b in bs}
        G, P = group_boxes(groups), group_boxes(det.get(key, {}), keep)
        n_gt, n_pred = n_gt + len(G), n_pred + len(P)
        pairs = sorted(((iou(g, p), i, j) for i, g in enumerate(G) for j, p in enumerate(P)), reverse=True)
        ug, up = set(), set()
        for v, i, j in pairs:
            if v <= 0:
                break
            if i in ug or j in up:
                continue
            ug.add(i), up.add(j)
            iou_sum += v
            tp += v >= thr
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gt if n_gt else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return 100 * iou_sum / max(n_gt, 1), 100 * p, 100 * r, 100 * f, n_pred, n_gt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-g", "--gt", required=True)
    ap.add_argument("-d", "--dets", nargs="+", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()
    gt = load(args.gt)
    print("name\tmIoU\tP\tR\tF1\tn_pred\tn_gt")
    for d in args.dets:
        m, p, r, f, npd, ngt = score(gt, load(d), args.iou)
        print(f"{os.path.basename(d)}\t{m:.2f}\t{p:.2f}\t{r:.2f}\t{f:.2f}\t{npd}\t{ngt}")
