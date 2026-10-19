"""JRDB val (v2, test rule) + test: AP (JRDB_eval.py task_3, G1..G5+ and overall), frame-wise F1 with the half criterion
(as RTGD/rtgd_eval.py: groups of >= 2 people; a 2-person GT group needs an exact match, larger ones overlap / max size
> 0.5) and with the exact criterion (same members, as F1_evaluator --crit exact) and the MINGLE group-box metric (mingle_det.py), for every row of make_jrdb_mingle_table.py plus our older
JRDB-trained models (trained on all 27 JRDB scenes, so their val column is left empty).
People are identified by their box (unique per keyframe); only keyframes and boxes of the GT file count.

    python make_jrdb_all_metrics.py            # prints the tables, writes results/paper_table_JRDB_{AP,F1half,F1exact}_valv2.tex
"""
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from mingle_det import load, score

HERE = os.path.dirname(os.path.abspath(__file__))
DET = os.path.join(HERE, "..", "..", "results", "detection_files")
RES = os.path.join(HERE, "..", "..", "results")
SETS = [("Val", "JRDB_val_v2_15", "out/gt_group_jrdb_val_v2.txt"), ("Test", "JRDB_fixed_gold_15", "out/gt_group.txt")]


def ours(tag):
    rows = []
    for lab, m, p in [("LLM", "llm", "p1"), ("VLM", "vlm_image", "p1_visual")]:
        for full in (False, True):
            for n in ("SFT", "RL"):
                st = f"{'full_' if full else ''}Qwen2.5{'-VL' if m == 'vlm_image' else ''}-7B-Instruct-{n}-{tag}_{m}_detany_3D_{p}"
                rows.append((f"Ours {lab} {n} ({'video' if full else 'single frame'})", st))
    return rows


SECTIONS = [
    ("Trained on JRDB", [("Social-CAD", "ECCV20_JRDBTrain_ep185"), ("JRDB-Act", "JRDBAct_JRDBTrain_ep40"),
                         ("S3R2", "SHGD_JRDBTrain_ep20"), ("P-HAR", "PHAR_JRDBTrain_ep11"), ("RTGD", "RTGD_JRDBTrain_ep5"),
                         ("GroupTrans", "GroupTrans_JRDBTrain_ep199"), ("DVT3", "DVT3_JRDBpretrained_th0.9")]),
    ("Ours, trained on all 27 JRDB scenes (older checkpoints; val seen in training)", ours("JRDBTrain")),
    ("Trained on EgoGroups (zero-shot on JRDB)", [
        ("Social-CAD", "ECCV20_EgoGroupsTrain_ep40"), ("JRDB-Act", "JRDBAct_EgoGroupsTrain_ep20"),
        ("S3R2", "SHGD_EgoGroupsTrain_ep100"), ("P-HAR", "PAR_EgoGroupsTrain_wd1e-3_ep19"),
        ("RTGD", "RTGD_EgoGroupsTrain_ep40"), ("GroupTrans", "GroupTrans_EgoGroupsTrain_ep199"),
        ("DVT3", "DVT3_EgoGroupsTrain_ep1_th0.50")]),
    ("Ours, trained on EgoGroups (zero-shot on JRDB)", ours("EgoGroupTrain")),
    ("Trained on its own data", [("MINGLE", "MINGLE-1.0")]),
]
SKIP_VAL = {"Ours, trained on all 27 JRDB scenes (older checkpoints; val seen in training)"}


def f1_half(gt, det, exact=False):
    tp = n_pred = n_gt = 0
    for key, groups in gt.items():
        keep = {b for bs in groups.values() for b in bs}
        gts = [set(bs) for bs in groups.values() if len(bs) > 1]
        prs = [s for s in (set(b for b in bs if b in keep) for bs in det.get(key, {}).values()) if len(s) > 1]
        n_gt, n_pred = n_gt + len(gts), n_pred + len(prs)
        for gg in gts:
            for pg in prs:
                tp += (gg == pg) if exact or (len(gg) == 2 and len(pg) == 2) else (len(gg & pg) / max(len(gg), len(pg)) > 0.5)
    p, r = (tp / n_pred if n_pred else 0), (tp / n_gt if n_gt else 0)
    return 100 * p, 100 * r, 100 * 2 * p * r / (p + r + 1e-8)


def ap(gt_file, det_file):
    res = subprocess.run([sys.executable, "JRDB_eval.py", "-g", gt_file, "-d", det_file, "-t", "task_3", "--labelmap",
                          "./label_map/task_3.pbtxt", "--output", "all"], cwd=HERE, capture_output=True, text=True)
    vals = [float(v) for v in res.stdout.split() if re.fullmatch(r"\d+(\.\d+)?", v)][:6]
    return [100 * v for v in vals[1:6]] + [100 * vals[0]] if len(vals) == 6 else None


gts = {name: load(os.path.join(HERE, g)) for name, _, g in SETS}
jobs = {}
for sec, rows in SECTIONS:
    for m, stem in rows:
        for name, folder, g in SETS:
            f = os.path.join(DET, folder, f"results_{stem}.txt")
            if os.path.exists(f) and not (name == "Val" and sec in SKIP_VAL):
                jobs[sec, m, name] = (os.path.join(HERE, g), f)
with ThreadPoolExecutor(8) as ex:
    aps = dict(zip(jobs, ex.map(lambda j: ap(*j), jobs.values())))
vals = {}
for k, (g, f) in jobs.items():
    d = load(f)
    vals[k] = {"AP": aps[k], "F1": f1_half(gts[k[2]], d), "F1exact": f1_half(gts[k[2]], d, exact=True),
               "MINGLE": score(gts[k[2]], d)[:4]}

COLS = {"AP": ["G1", "G2", "G3", "G4", "G5+", "AP"], "F1": ["P", "R", "F1"], "F1exact": ["P", "R", "F1"],
        "MINGLE": ["mIoU", "P", "R", "F1"]}
TAG = {"AP": "AP", "F1": "F1half", "F1exact": "F1exact"}
DESC = {"AP": "AP per group size (G1..G5+) and overall",
        "F1": "frame-wise P / R / F1, half criterion, groups of two or more",
        "F1exact": "frame-wise P / R / F1, exact criterion (same members), groups of two or more"}
for metric, cols in COLS.items():
    print(f"\n### {metric}  (val = gt_group_jrdb_val_v2.txt, test = gt_group.txt)")
    print("%-34s | %s | %s" % ("", "  ".join("%6s" % ("V " + c) for c in cols), "  ".join("%6s" % ("T " + c) for c in cols)))
    tex = []
    for sec, rows in SECTIONS:
        print(f"-- {sec}")
        tex.append(rf"    \multicolumn{{{1 + 2 * len(cols)}}}{{l}}{{\textit{{{sec}}}}} \\")
        for m, _ in rows:
            cells = []
            for name, _, _ in SETS:
                v = vals.get((sec, m, name), {}).get(metric)
                cells.append(["%6.2f" % x for x in v] if v else ["%6s" % "-"] * len(cols))
            print("%-34s | %s | %s" % (m, "  ".join(cells[0]), "  ".join(cells[1])))
            tex.append("    " + " & ".join([m] + [c.strip() for c in cells[0] + cells[1]]) + r" \\")
        tex.append(r"    \midrule")
    if metric != "MINGLE":
        n = len(cols)
        out = os.path.join(RES, f"paper_table_JRDB_{TAG[metric]}_valv2.tex")
        head = [r"\begin{table}[h!]", r"    \centering", r"    \footnotesize",
                rf"    \caption{{JRDB val (no-eval and out-of-image people left out, as on test) and test, "
                rf"{DESC[metric]}.}}",
                r"    \resizebox{\linewidth}{!}{", rf"    \begin{{tabular}}{{l{'c' * 2 * n}}}", r"    \toprule",
                rf"        & \multicolumn{{{n}}}{{c}}{{\textbf{{JRDB Val}}}} & \multicolumn{{{n}}}{{c}}{{\textbf{{JRDB Test}}}} \\",
                "    Method & " + " & ".join(cols + cols) + r" \\", r"    \midrule"]
        tail = [r"    \bottomrule", r"    \end{tabular}", r"    }",
                rf"    \label{{table:jrdb_val_test_{TAG[metric]}}}", r"\end{table}"]
        open(out, "w").write("\n".join(head + tex[:-1] + tail) + "\n")
        print("wrote", os.path.abspath(out))
