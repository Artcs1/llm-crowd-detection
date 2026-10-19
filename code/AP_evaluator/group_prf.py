"""Precision / recall / F1 of social groups for every detection file in a folder, against an AP_evaluator GT file.

Both files are in the JRDB_eval format ("seq frame x1 y1 x2 y2 group act score"); people are matched by
(seq, frame, box). Per keyframe, a predicted group is a true positive if it matches a GT group under the "half"
criterion used by DVT / PAR / GroupTrans: for two pairs both members must agree, otherwise |G n P| / max(|G|, |P|) > 1/2.
Reported twice: groups of >= 2 people only (singletons excluded, the DVT EgoGroups convention) and all groups.

    python group_prf.py --gt out/gt_group_10fps_test.txt --dets ../../results/detection_files/EgoGroups_test_10 [--csv out.csv]
"""

import argparse
import glob
import os
from collections import defaultdict


def read(path):
    """{(seq, frame): {group id: set(box keys)}}"""
    frames = defaultdict(lambda: defaultdict(set))
    for line in open(path):
        r = line.split()
        if len(r) < 7:
            continue
        frames[(r[0], r[1])][r[6]].add(tuple(r[2:6]))
    return frames


def prf(gt, det, min_size):
    tp = n_gt = n_pred = 0
    for key, gt_groups in gt.items():
        gts = [g for g in gt_groups.values() if len(g) >= min_size]
        prs = [g for g in det.get(key, {}).values() if len(g) >= min_size]
        n_gt += len(gts)
        n_pred += len(prs)
        for g in gts:
            for p in prs:
                if len(g) == 2 and len(p) == 2:
                    tp += g == p
                elif len(g & p) / max(len(g), len(p)) > 0.5:
                    tp += 1
    prec = tp / n_pred if n_pred else 0.0
    rec = tp / n_gt if n_gt else 0.0
    return prec, rec, 2 * prec * rec / (prec + rec + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dets", required=True)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    gt = read(args.gt)
    rows = []
    for path in sorted(glob.glob(os.path.join(args.dets, "*.txt"))):
        det = read(path)
        p2, r2, f2 = prf(gt, det, 2)
        p1, r1, f1 = prf(gt, det, 1)
        rows.append((os.path.basename(path), p2, r2, f2, p1, r1, f1))
    rows.sort(key=lambda r: -r[3])
    print(f"{'file':72s} {'P':>6s} {'R':>6s} {'F1':>6s} | {'P_all':>6s} {'R_all':>6s} {'F1_all':>6s}")
    for r in rows:
        print(f"{r[0][:72]:72s} {r[1]:6.3f} {r[2]:6.3f} {r[3]:6.3f} | {r[4]:6.3f} {r[5]:6.3f} {r[6]:6.3f}")
    if args.csv:
        with open(args.csv, "w") as f:
            f.write("name,P,R,F1,P_all,R_all,F1_all\n")
            for r in rows:
                f.write(",".join([r[0]] + ["%.4f" % v for v in r[1:]]) + "\n")


if __name__ == "__main__":
    main()
