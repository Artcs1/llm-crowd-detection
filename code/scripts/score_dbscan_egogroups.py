"""Score the DBSCAN runs of scripts/run_dbscan_egogroups.sh: AP on each split's val clips (to pick eps), and AP
(+ G1..G5), F1 (F1_evaluator/ours_jrdb_evaluation.py, half rule) and MINGLE F1 (F1_evaluator/mingle_metric.py) on every
test set. Writes ../results/dbscan_egogroups.md and prints it.

    python scripts/score_dbscan_egogroups.py      (from code/, conda env py10-dspy)
"""

import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from egogroups_city import BENCHMARKS  # noqa: E402

RES = os.path.join(CODE, "..", "results")
ROOT = os.path.join(CODE, "..", "..")
TMP = os.path.join(CODE, "sft_output", "logs", "dbscan_egogroups", "score")
os.makedirs(TMP, exist_ok=True)
EPS = ["0.2", "0.4", "0.6", "0.9", "1.2", "1.5", "1.8"]
NAMES = [f"DBSCAN{d}D_t{t}" for d in (2, 3) for t in EPS]
GEO180 = [int(l) for l in open(os.path.join(CODE, "egogroups_val_geo180.txt")) if l.strip()]
# split -> (val folder, val clips, val GT txt, seq shift of the val GT = clip - 1 - shift, [(test folder, GT name)])
SPLITS = {
    "EgoGroups (original)": ("EgoGroups_train", GEO180, "gt_group_10fps_train", 900, [("EgoGroups_test", "gt_group_10fps_test")]),
    "egogroups-city": ("EgoGroupsCity_Train", BENCHMARKS["city"]["val"], "gt_group_10fps_city_val", 0,
                       [("EgoGroupsCity_Test", "gt_group_10fps_city_test")]),
    "benchmark2": ("EgoGroupsBenchmark2_Train", BENCHMARKS["benchmark2"]["val"], "gt_group_10fps_benchmark2_val", 0,
                   [("EgoGroupsBenchmark2_IDTest", "gt_group_10fps_benchmark2_id_test"),
                    ("EgoGroupsBenchmark2_OODTest", "gt_group_10fps_benchmark2_ood_test")]),
    "benchmark3": ("EgoGroupsBenchmark3_Train", BENCHMARKS["benchmark3"]["id_val"], "gt_group_10fps_benchmark3_id_val", 0,
                   [("EgoGroupsOODCity_Test", "gt_group_10fps_benchmark3_id_test"),
                    ("EgoGroupsOODRegions_Test", "gt_group_10fps_benchmark3_ood_test")]),
}


def det(folder, name):
    return os.path.join(RES, "detection_files", f"{folder}_10", f"results_{name}_llm_detany_3D_naive_cluster.txt")


def ap(gt, d):
    out = subprocess.run([sys.executable, "JRDB_eval.py", "-g", gt, "-d", d, "-t", "task_3", "--labelmap",
                          "./label_map/task_3.pbtxt", "--output", "all"], cwd=os.path.join(CODE, "AP_evaluator"),
                         capture_output=True, text=True).stdout
    v = [float(x) for x in re.findall(r"^[0-9.]+$", out, re.M)]
    return v[:6] if len(v) >= 6 else None


def val_ap(split, name):
    folder, clips, gt, shift, _ = SPLITS[split]
    keep = {c - 1 for c in clips}
    d = os.path.join(TMP, f"val_{folder}_{name}.txt")
    with open(d, "w") as f:  # the val clips' rows only, renumbered like the val GT
        for line in open(det(folder, name)):
            a = line.split()
            if int(a[0]) in keep:
                f.write(" ".join([str(int(a[0]) - shift)] + a[1:]) + "\n")
    g = os.path.join(CODE, "AP_evaluator", "out", gt + ".txt")
    if shift:  # EgoGroups_train GT: only the val clips' rows
        gk = os.path.join(TMP, f"valgt_{folder}.txt")
        ks = {c - 901 for c in clips}
        with open(gk, "w") as f:
            f.writelines(l for l in open(g) if int(l.split()[0]) in ks)
        g = gk
    r = ap(g, d)
    return r[0] if r else None


def test_scores(folder, gt, name):
    d = det(folder, name)
    a = ap(os.path.join(CODE, "AP_evaluator", "out", gt + ".txt"), d)
    pkl = os.path.join(RES, "grouping_files", f"{folder}_10", f"results_{name}_llm_detany_3D_naive_cluster.pkl")
    os.makedirs(os.path.dirname(pkl), exist_ok=True)
    subprocess.run([sys.executable, "F1_evaluator/det_txt_to_pkl.py", d, pkl, "--split", "test"], cwd=CODE,
                   capture_output=True)
    out = subprocess.run([sys.executable, "ours_jrdb_evaluation.py", "--det", pkl, "--gt", f"out/{gt}.pkl", "--o", "all"],
                         cwd=os.path.join(CODE, "F1_evaluator"), capture_output=True, text=True).stdout
    f1 = [float(x) for x in re.findall(r"^[0-9.]+$", out, re.M)][-3:]
    return a, f1


def mingle(folder, gt):
    csv = os.path.join(TMP, f"mingle_{folder}.csv")
    clips = os.path.join(ROOT, folder, "jsons_step1")
    subprocess.run([sys.executable, "mingle_metric.py", "--dets", os.path.join(RES, "grouping_files", f"{folder}_10"),
                    "--gt", f"out/{gt}.pkl", "--csv", csv, "--match", "DBSCAN", "--clips", clips],
                   cwd=os.path.join(CODE, "F1_evaluator"), capture_output=True)
    m = {}
    for line in open(csv).read().splitlines()[1:]:
        r = line.split(",")
        name = re.search(r"results_(DBSCAN\dD_t[0-9.]+)_", r[0])
        if name:
            m[name.group(1)] = float(r[4])  # name, mIoU, P, R, F1
    return m


with ThreadPoolExecutor(8) as ex:
    val = {(s, n): ex.submit(val_ap, s, n) for s in SPLITS for n in NAMES}
    tests = {(f, n): ex.submit(test_scores, f, g, n) for s in SPLITS for f, g in SPLITS[s][4] for n in NAMES}
    val = {k: v.result() for k, v in val.items()}
    tests = {k: v.result() for k, v in tests.items()}
ming = {f: mingle(f, g) for s in SPLITS for f, g in SPLITS[s][4]}

out = ["# DBSCAN (naive_cluster.py, min_samples 1) on the EgoGroups test sets", "",
       "eps picked on each split's val clips (best val AP), separately for 2D (x, y) and 3D (x, y, z) detany_3D positions.",
       "AP = JRDB_eval task_3; F1 = ours_jrdb_evaluation (half rule, singletons excluded); MINGLE = group-box F1 (IoU 0.5).", ""]
for s, (vf, _, _, _, tl) in SPLITS.items():
    out += [f"## {s} (val: {vf})", "", "| setting | eps (val) | val AP | " + " | ".join(
        f"{f}: AP / F1 / MINGLE F1" for f, _ in tl) + " |", "|" + "---|" * (3 + len(tl))]
    for dims in ("2D", "3D"):
        names = [n for n in NAMES if n.startswith(f"DBSCAN{dims}")]
        best = max(names, key=lambda n: val[s, n] if val[s, n] is not None else -1)
        cells = []
        for f, _ in tl:
            a, f1 = tests[f, best]
            cells.append(f"{100 * a[0]:.2f} / {100 * f1[-1]:.2f} / {ming[f].get(best, float('nan')):.2f}" if a and f1 else "n/a")
        out.append(f"| DBSCAN {dims} | {best.split('_t')[1]} | {100 * val[s, best]:.2f} | " + " | ".join(cells) + " |")
    out += ["", "All eps (test AP / F1 / MINGLE F1; val AP):", "", "| run | val AP | " + " | ".join(f for f, _ in tl) + " |",
            "|" + "---|" * (2 + len(tl))]
    for n in NAMES:
        cells = []
        for f, _ in tl:
            a, f1 = tests[f, n]
            cells.append(f"{100 * a[0]:.2f} / {100 * f1[-1]:.2f} / {ming[f].get(n, float('nan')):.2f}" if a and f1 else "n/a")
        v = val[s, n]
        out.append(f"| {n} | {100 * v:.2f} | " if v is not None else f"| {n} | n/a | ")
        out[-1] += " | ".join(cells) + " |"
    out.append("")
open(os.path.join(RES, "dbscan_egogroups.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
