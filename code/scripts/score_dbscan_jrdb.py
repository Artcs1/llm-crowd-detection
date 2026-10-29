"""Score the DBSCAN runs of scripts/run_dbscan_jrdb.sh: AP on the JRDB val scenes (out/gt_group_jrdb_val_v2.txt, to pick
eps) and, on JRDB test, AP (+ G1..G5) vs out/gt_group.txt, F1 half / exact (ours_jrdb_evaluation.py vs F1_evaluator/out/gt.pkl,
grouping pickles from compute_groupings.py) and MINGLE (mingle_det.py: mIoU / P / R / F1). Writes ../results/dbscan_jrdb.md.

    python scripts/score_dbscan_jrdb.py      (from code/, conda env py10-dspy)
"""

import os
import pickle
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(CODE, "..", "results")
APD, F1D = os.path.join(CODE, "AP_evaluator"), os.path.join(CODE, "F1_evaluator")
EPS = ["0.2", "0.4", "0.6", "0.9", "1.2", "1.5", "1.8", "2.1", "2.4"]
NAMES = [f"DBSCAN{d}D_t{t}" for d in (2, 3) for t in EPS]
TAIL = "_llm_detany_3D_naive_cluster"


def run(cmd, cwd):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True).stdout


def ap(gt, d):
    out = run([sys.executable, "JRDB_eval.py", "-g", gt, "-d", d, "-t", "task_3", "--labelmap", "./label_map/task_3.pbtxt",
               "--output", "all"], APD)
    v = [float(x) for x in re.findall(r"^[0-9.]+$", out, re.M)]
    return v[:6] if len(v) >= 6 else None


def score(name):
    val = ap("out/gt_group_jrdb_val_v2.txt", f"{RES}/detection_files/JRDB_val_fixed_gold_15/results_{name}{TAIL}.txt")
    det = f"{RES}/detection_files/JRDB_fixed_gold_15/results_{name}{TAIL}.txt"
    test = ap("out/gt_group.txt", det)
    run([sys.executable, "compute_groupings.py", "--dataset", "JRDB_fixed_gold", "--mode", "single", "--depth_method",
         "detany_3D", "--prompt_method", "naive_cluster", "--model", name, "--vlm_mode", "llm", "--frame_id", "15"], CODE)
    pkl = f"{RES}/grouping_files/JRDB_fixed_gold_15/results_{name}{TAIL}.pkl"
    g = pickle.load(open(pkl, "rb"))  # naive_cluster's person ids are strings ('4'); gt.pkl's are ints
    pickle.dump({s: {k: [[int(i) for i in grp] for grp in v] for k, v in fr.items()} for s, fr in g.items()}, open(pkl, "wb"))
    f1 = {}
    for crit in ("half", "exact"):
        out = run([sys.executable, "ours_jrdb_evaluation.py", "--det", pkl, "--gt", "out/gt.pkl", "--o", "all", "--crit", crit], F1D)
        f1[crit] = [float(x) for x in re.findall(r"^[0-9.]+$", out, re.M)][-3:]  # P, R, F1
    m = run([sys.executable, "mingle_det.py", "-g", "out/gt_group.txt", "-d", det], APD).strip().splitlines()[-1].split()
    return val, test, f1, [float(x) for x in m[1:5]]  # mIoU, P, R, F1


with ThreadPoolExecutor(6) as ex:
    res = dict(zip(NAMES, ex.map(score, NAMES)))

pct = lambda x: f"{100 * x:.2f}"
out = ["# DBSCAN (naive_cluster.py, min_samples 1) on JRDB test", "",
       "Lidar 3D positions of the JRDB_fixed_gold shards; 2D = (x, y) ground plane, 3D = (x, y, z). eps picked by val AP "
       "(7 JRDB val scenes, gt_group_jrdb_val_v2). AP = JRDB_eval task_3 vs out/gt_group.txt; F1 = ours_jrdb_evaluation "
       "(singletons excluded); MINGLE = mingle_det.py group boxes.", "",
       "| run | val AP | AP | G1 | G2 | G3 | G4 | G5+ | F1 half (P / R / F1) | F1 exact (P / R / F1) | MINGLE mIoU | MINGLE P | MINGLE R | MINGLE F1 |",
       "|" + "---|" * 14]
for dims in ("2D", "3D"):
    names = [n for n in NAMES if n.startswith(f"DBSCAN{dims}")]
    best = max(names, key=lambda n: res[n][0][0] if res[n][0] else -1)
    for n in names:
        val, test, f1, m = res[n]
        prf = lambda c: " / ".join(pct(x) for x in f1[c])
        lab = f"**{n}** (val pick)" if n == best else n
        out.append(f"| {lab} | {pct(val[0])} | " + " | ".join(pct(x) for x in test) + f" | {prf('half')} | {prf('exact')} | "
                   + " | ".join(f"{x:.2f}" for x in m) + " |")
open(os.path.join(RES, "dbscan_jrdb.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
