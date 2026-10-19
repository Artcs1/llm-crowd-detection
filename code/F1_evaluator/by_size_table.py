"""Overall F1 (exactly ours_jrdb_evaluation.py: frame-wise, singletons excluded) plus the per-group-size breakdown
G1, G2, G3, G4, G5+ (Evaluation.size_breakdown, singletons included for G1) for every grouping pickle in a folder.

    python by_size_table.py --dets ../../results/grouping_files/EgoGroups_test_10 --gt out/gt_group_10fps_test.pkl \
                            --csv results/by_size_all.csv [--crit exact] [--match DVT3,PAR,...]
"""

import argparse
import contextlib
import glob
import io
import os
import pickle

from utils_eval import Evaluation

ap = argparse.ArgumentParser()
ap.add_argument("--dets", required=True)
ap.add_argument("--gt", required=True)
ap.add_argument("--csv", required=True)
ap.add_argument("--crit", default="half", choices=["half", "card", "dpmm", "all", "exact"])
ap.add_argument("--match", default="", help="comma-separated substrings; only pickles containing one of them")
args = ap.parse_args()

gt = pickle.load(open(args.gt, "rb"))
keys = list(gt.keys())
cols = ["P", "R", "F1"] + [f"{b}_{m}" for b in ["G1", "G2", "G3", "G4", "G5+", "Group", "All"] for m in ["P", "R", "F1"]]
rows = []
for path in sorted(glob.glob(os.path.join(args.dets, "*.pkl"))):
    name = os.path.basename(path)
    if args.match and not any(m in name for m in args.match.split(",")):
        continue
    pred = pickle.load(open(path, "rb"))
    pred = {k: pred[k] if k in pred else [] for k in keys}  # as ours_jrdb_evaluation.py
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            p, r, f = Evaluation(whole=False, alone=False, pre_dict=pred, GT_dict=gt, crit=args.crit)()
    except ZeroDivisionError:  # Evaluation.__call__ divides by the number of predicted groups (0 for empty pickles)
        p = r = f = 0.0
    sizes = Evaluation(whole=False, alone=True, pre_dict=pred, GT_dict=gt, crit=args.crit).size_breakdown()
    vals = [p, r, f] + [sizes[b][m] for b in ["G1", "G2", "G3", "G4", "G5+", "Group", "All"] for m in ["P", "R", "F1"]]
    rows.append((name, vals))
    print(f"{name[:70]:70s} F1 {f*100:5.2f} | " + " ".join(f"{b} {sizes[b]['F1']*100:5.2f}" for b in sizes), flush=True)

rows.sort(key=lambda r: -r[1][2])
with open(args.csv, "w") as fo:
    fo.write("name," + ",".join(cols) + "\n")
    for name, vals in rows:
        fo.write(name + "," + ",".join("%.2f" % (v * 100) for v in vals) + "\n")
print("wrote", args.csv)
