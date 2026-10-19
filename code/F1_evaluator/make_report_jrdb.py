"""JRDB test (1829 keyframes, AP_evaluator/out/gt_group.txt): MINGLE group-box F1 (mingle_det.py, IoU 0.5, pooled) of
every method trained on JRDB vs trained on EgoGroups_train, side by side, as a text report.

    python make_report_jrdb.py [--out ../../results/jrdb_test_MINGLE_F1_jrdb_vs_ego.txt]
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AE = os.path.join(HERE, "..", "AP_evaluator")
sys.path.insert(0, AE)
from mingle_det import load, score  # noqa: E402

RES = os.path.join(HERE, "..", "..", "results")
DET = os.path.join(RES, "detection_files")
GT = os.path.join(AE, "out", "gt_group.txt")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(RES, "jrdb_test_MINGLE_F1_jrdb_vs_ego.txt"))
ap.add_argument("--min_cov", type=float, default=95.0, help="runs with fewer GT people (%%) are marked incomplete")
args = ap.parse_args()

T, EGO3D = "JRDB_fixed_gold_15", "JRDB_fixed_gold_egostyle_15"
BASE = [("Social-CAD", "ECCV20_JRDBTrain_ep185", "ECCV20_EgoGroupsTrain_ep40"),
        ("JRDB-Act", "JRDBAct_JRDBTrain_ep40", "JRDBAct_EgoGroupsTrain_ep20"),
        ("S3R2", "SHGD_JRDBTrain_ep20", "SHGD_EgoGroupsTrain_ep100"),
        ("P-HAR", "PHAR_JRDBTrain_ep11", "PAR_EgoGroupsTrain_wd1e-3_ep19"),
        ("RTGD", "RTGD_JRDBTrain_ep5", "RTGD_EgoGroupsTrain_ep40"),
        ("GroupTrans", "GroupTrans_JRDBTrain_ep199", "GroupTrans_EgoGroupsTrain_ep199"),
        ("DVT3", "DVT3_JRDBpretrained_th0.9", "DVT3_EgoGroupsTrain_ep1_th0.50")]
OURS = []
for lab, m, p in [("LLM", "llm", "p1"), ("VLM", "vlm_image", "p1_visual")]:
    for full in (False, True):
        for n in ("SFT", "RL"):
            st = lambda tag: f"{'full_' if full else ''}Qwen2.5{'-VL' if m == 'vlm_image' else ''}-7B-Instruct-{n}-{tag}_{m}_detany_3D_{p}"
            OURS.append((f"Ours {lab} {n} ({'video' if full else 'single frame'})", st("JRDBTrain"), st("EgoGroupTrain")))

gt = load(GT)
keep = {k: {b for bs in g.values() for b in bs} for k, g in gt.items()}
n_gt = sum(len(v) for v in keep.values())


def f1(folder, stem):
    f = os.path.join(DET, folder, f"results_{stem}.txt")
    if not os.path.exists(f):
        return "-"
    d = load(f)
    cov = 100 * sum(1 for k, g in d.items() if k in keep for bs in g.values() for b in bs if b in keep[k]) / n_gt
    return "%.2f" % score(gt, d)[3] + ("*" if cov < args.min_cov else " ")


out = ["JRDB test (1829 keyframes, %d people): MINGLE F1 (%%) -- group boxes, one-to-one greedy IoU matching, IoU >= 0.5, pooled" % n_gt,
       "JRDB-train: baselines on the 20 JRDB train scenes (epoch picked on val), DVT3 = released checkpoint (th 0.9);",
       "            ours = older checkpoints trained on all 27 JRDB train+val scenes (VLM video never trained on JRDB).",
       "EgoGroups-train: the checkpoints of the EgoGroups_test tables (zero-shot on JRDB).",
       "  JRDB 3D = people's 3D as in the JRDB shards (robot frame: x forward, y left, z up).",
       "  Ego 3D  = same keyframes, 3D converted to the EgoGroups detany_3D camera convention (scripts/make_jrdb_egostyle.py);",
       "            only our models read 3D, so only they have this column.",
       "diff = EgoGroups-train (best of the two 3D columns) - JRDB-train.  '*' = incomplete run (< %.0f%% of GT people)." % args.min_cov,
       "", "%-30s %12s %16s %16s %8s" % ("method", "JRDB-train", "Ego (JRDB 3D)", "Ego (Ego 3D)", "diff"), "-" * 86]


def row(name, a, b, c=""):
    nums = [float(x.rstrip("* ")) for x in (b, c) if x not in ("", "-") and "*" not in x]
    d = "%+.2f" % (max(nums) - float(a.rstrip("* "))) if nums and a != "-" and "*" not in a else "-"
    out.append("%-30s %12s %16s %16s %8s" % (name, a, b, c, d))


out.append("Specialized methods")
for name, sj, se in BASE:
    row(name, f1(T, sj), f1(T, se))
out.append("Ours (Qwen2.5(-VL)-7B, SFT / RL)")
for name, sj, se in OURS:
    row(name, f1(T, sj), f1(T, se), f1(EGO3D, se))
out.append("Trained on its own data")
row("MINGLE", f1(T, "MINGLE-1.0"), "-")

# in-domain: EgoGroups_train -> EgoGroups_test (as in the paper tables: mingle_metric.py on the grouping pickles, the
# stems of make_paper_ap_tables.py EGO_BASELINES / FT) vs JRDB-train -> JRDB test (above)
import csv, subprocess, tempfile  # noqa: E401,E402
EGO_TEST = {"Social-CAD": "ECCV20_EgoGroupsTrain_ep40", "JRDB-Act": "JRDBAct_EgoGroupsTrain_ep20",
            "S3R2": "SHGD_EgoGroupsTrain_ep100", "P-HAR": "PAR_EgoGroupsTrain_wd1e-3_ep19", "RTGD": "RTGD_EgoGroupsTrain_ep40",
            "GroupTrans": "GroupTrans_EgoGroupsTrain_ep199", "DVT3": "DVT3_EgoGroupsTrain_ep1_th0.50",
            "Ours LLM SFT (single frame)": "Qwen2.5-7B-Instruct-egogroups-SFT_llm_detany_3D_p1",
            "Ours LLM RL (single frame)": "Qwen2.5-7B-Instruct-egogroups-RL-ep1_llm_detany_3D_p1",
            "Ours LLM SFT (video)": "full_Qwen2.5-7B-Instruct-egogroups-SFT_llm_detany_3D_p1",
            "Ours LLM RL (video)": "full_Qwen2.5-7B-Instruct-egogroups-RL-ep1_llm_detany_3D_p1",
            "Ours VLM SFT (single frame)": "Qwen2.5-VL-7B-Instruct-egogroups-SFT_vlm_image_detany_3D_p1_visual",
            "Ours VLM RL (single frame)": "Qwen2.5-VL-7B-Instruct-RL1_vlm_image_detany_3D_p1_visual",
            "Ours VLM SFT (video)": "full_Qwen2.5-VL-7B-Instruct-SFT_vlm_image_detany_3D_p1_visual",
            "Ours VLM RL (video)": "full_Qwen2.5-VL-7B-Instruct-RL_vlm_image_detany_3D_p1_visual",
            "MINGLE": "MINGLE_vlm_image_evp_depth_mingle"}
# JRDB-trained methods on EgoGroups_test (make_paper_ap_tables.py ZS_BASELINES); ours JRDB-trained not run there yet
JRDB_ON_EGO = {"Social-CAD": "ECCV20_JRDBTrain_ep185", "JRDB-Act": "JRDBAct_JRDBTrain_ep40", "S3R2": "SHGD_JRDBTrain_ep20",
               "P-HAR": "PAR_JRDBTrain_ep11", "RTGD": "RTGD_JRDBTrain_ep5", "GroupTrans": "GroupTrans_JRDBTrain_ep199",
               "DVT3": "DVT3_JRDBpretrained_th0.9", "MINGLE": "Qwen2-VL-7B-Instruct-base-mingle_mingle_detany_3D_p1"}
for n, sj, _ in OURS:  # picked up automatically once these runs are scored on EgoGroups_test
    JRDB_ON_EGO[n] = sj
with tempfile.TemporaryDirectory() as tmp:
    subprocess.run([sys.executable, "mingle_metric.py", "--dets", os.path.join(RES, "grouping_files", "EgoGroups_test_10"),
                    "--gt", os.path.join(HERE, "out", "gt_group_10fps_test.pkl"), "--csv", os.path.join(tmp, "m.csv"),
                    "--iou", "0.5", "--match", ",".join(list(EGO_TEST.values()) + list(JRDB_ON_EGO.values()))],
                   cwd=HERE, check=True, capture_output=True)
    EG = {r["name"][len("results_"):-len(".pkl")]: float(r["F1"]) for r in csv.DictReader(open(os.path.join(tmp, "m.csv")))}
eg = lambda stem: "%.2f " % EG[stem] if stem in EG else "- "
jt = {n: f1(T, sj) for n, sj, _ in BASE + OURS}
jt["MINGLE"] = f1(T, "MINGLE-1.0")
je = {n: f1(T, se) for n, _, se in BASE + OURS}
jx = {n: f1(EGO3D, se) for n, _, se in OURS}
SEC = [("Specialized methods", [b[0] for b in BASE]), ("Ours (Qwen2.5(-VL)-7B, SFT / RL)", [o[0] for o in OURS]),
       ("MINGLE", ["MINGLE"])]
out += ["", "", "=" * 86,
        "In-domain: EgoGroups_train -> EgoGroups_test vs JRDB-train -> JRDB test, MINGLE F1 (%)",
        "EgoGroups_test (900 clips x 5 keyframes): mingle_metric.py on the grouping pickles, same numbers as the paper tables.",
        "JRDB test: as above (ours JRDB-train = older 27-scene checkpoints; VLM video never trained on JRDB).",
        "MINGLE row: EgoGroups stem MINGLE_vlm_image_evp_depth_mingle (the paper's EgoGroups-train table row), JRDB = MINGLE-1.0.",
        "=" * 86, "%-30s %18s %18s" % ("method", "Ego -> Ego test", "JRDB -> JRDB test"), "-" * 68]
for sec, names in SEC:
    out.append(sec)
    for n in names:
        out.append("%-30s %18s %18s" % (n, eg(EGO_TEST[n]), jt[n]))
out += ["", "", "=" * 120,
        "All four train -> test combinations, MINGLE F1 (%)",
        "Ego->Ego and JRDB->Ego: EgoGroups_test (mingle_metric.py on grouping pickles, as the paper tables);",
        "JRDB->JRDB and Ego->JRDB: JRDB test (mingle_det.py on AP files). Ego->JRDB [ego-3D] = 3D in the EgoGroups convention.",
        "JRDB->Ego: baselines = the paper's zero-shot table rows; ours JRDB-trained have not been run on EgoGroups_test ('-').",
        "MINGLE: released model on JRDB test (MINGLE-1.0) and in JRDB->Ego (Qwen2-VL-7B-Instruct-base-mingle, paper zero-shot row);",
        "        Ego->Ego = MINGLE_vlm_image_evp_depth_mingle (paper EgoGroups-train row). '*' = incomplete run.",
        "=" * 120,
        "%-30s | %10s %10s | %10s %10s %16s" % ("", "EgoGroups", "test", "JRDB", "test", ""),
        "%-30s | %10s %10s | %10s %10s %16s" % ("method", "Ego-train", "JRDB-train", "JRDB-train", "Ego-train", "Ego-train[ego-3D]"),
        "-" * 96]
for sec, names in SEC:
    out.append(sec)
    for n in names:
        out.append("%-30s | %10s %10s | %10s %10s %16s" % (n, eg(EGO_TEST[n]), eg(JRDB_ON_EGO[n]), jt[n],
                                                           je.get(n, "- " if n != "MINGLE" else "- "), jx.get(n, "")))
open(args.out, "w").write("\n".join(out) + "\n")
print("\n".join(out))
print("wrote", os.path.abspath(args.out))

# ---- LaTeX: the four train -> test combinations, MINGLE F1; our rows use the 3D convention of their training data
# (Ego-train -> JRDB test: JRDB_fixed_gold_egostyle; JRDB-train -> EgoGroups test: EgoGroups_test_jrdbstyle)
JS = os.path.join(RES, "grouping_files", "EgoGroups_test_jrdbstyle_10")
EGJ = {}
if os.path.isdir(JS):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([sys.executable, "mingle_metric.py", "--dets", JS, "--gt", os.path.join(HERE, "out", "gt_group_10fps_test.pkl"),
                        "--csv", os.path.join(tmp, "m.csv"), "--iou", "0.5", "--match", ",".join(sj for _, sj, _ in OURS)],
                       cwd=HERE, check=True, capture_output=True)
        EGJ = {r["name"][len("results_"):-len(".pkl")]: float(r["F1"]) for r in csv.DictReader(open(os.path.join(tmp, "m.csv")))}
num = lambda x: None if x in (None, "", "-") or "*" in str(x) else float(str(x).strip())
CITE = {"Social-CAD": "social-cad", "JRDB-Act": "JRDB-Act", "S3R2": "li2022selfsupervised", "P-HAR": "han2022panoramic",
        "RTGD": "Jahangard_IROS_2023", "GroupTrans": "10286297", "DVT3": "Yokoyama_2025_ICCV",
        "MINGLE": "liu2025minglevlmssemanticallycomplex"}
TEXROWS = []  # (section, label, [Ego->Ego, JRDB->Ego, JRDB->JRDB, Ego->JRDB])
for n, _, _ in BASE:
    TEXROWS.append(("base", f"{n}~\\cite{{{CITE[n]}}}", [num(eg(EGO_TEST[n])), num(eg(JRDB_ON_EGO[n])), num(jt[n]), num(je[n])]))
for n, sj, _ in OURS:
    lab = n.replace("Ours ", "").replace(" (single frame)", " (single)").replace("SFT", "+ SFT").replace("RL", "+ RL")
    TEXROWS.append(("ours", lab, [num(eg(EGO_TEST[n])), EGJ.get(sj), num(jt[n]), num(jx[n])]))
TEXROWS.append(("mingle", f"MINGLE~\\cite{{{CITE['MINGLE']}}}", [num(eg(EGO_TEST["MINGLE"])), num(eg(JRDB_ON_EGO["MINGLE"])),
                                                                   num(jt["MINGLE"]), None]))
best = [max((r[2][i] for r in TEXROWS if r[2][i] is not None), default=None) for i in range(4)]
cell = lambda v, i: "-" if v is None else ("\\textbf{%.2f}" % v if abs(v - best[i]) < 1e-9 else "%.2f" % v)
L = [r"\begin{table}[h!]", r"    \centering", r"    \footnotesize",
     r"    \caption{Cross-dataset generalization: MINGLE F1 for every combination of training set (\dataset{}-Train or JRDB)"
     r" and test set (\dataset{}-Test or JRDB test). F1 is computed on group boxes (the smallest box around each group of"
     r" two or more people), matched one-to-one by IoU; a match with IoU $\geq 0.5$ is correct. Our models receive the"
     r" 3D positions in the coordinate convention of their training set (\dataset{} camera frame for \dataset{}-trained"
     r" models on JRDB, JRDB robot frame for JRDB-trained models on \dataset{}); distances between people are unchanged."
     r" $\Delta$ = cross-dataset minus in-domain F1 on the same test set (training on the other dataset;"
     r" \textcolor{green!55!black}{green} = gain, \textcolor{red!70!black}{red} = loss). Mean = average of each column over"
     r" the methods of the section that have a value ($\Delta$: over the methods with both F1 values)."
     r" MINGLE: released model, except on \dataset{}-Train $\rightarrow$ \dataset{}-Test. Bold = best per column.}",
     r"    \resizebox{\linewidth}{!}{", r"    \begin{tabular}{lcccccc}", r"    \toprule",
     r"        & \multicolumn{3}{c}{\textbf{\dataset{}-Test}} & \multicolumn{3}{c}{\textbf{JRDB Test}} \\",
     r"        \cmidrule(lr){2-4} \cmidrule(lr){5-7}",
     r"    Method & \dataset{}-Train & JRDB-Train & $\Delta$ & JRDB-Train & \dataset{}-Train & $\Delta$ \\", r"    \midrule"]
TITLES = {"base": "Specialized methods", "ours": "Ours (Qwen2.5(-VL)-7B)", "mingle": "VLM pairwise"}
dl = lambda d: "-" if d is None else (  # cross-dataset - in-domain; bold green gain, bold red loss
    "\\textbf{\\textcolor{%s}{%+.2f}}" % ("green!55!black" if d >= 0 else "red!70!black", d))
diff = lambda a, b: None if a is None or b is None else b - a
avg = lambda xs: (lambda ys: sum(ys) / len(ys) if ys else None)([x for x in xs if x is not None])
for k, sec in enumerate(TITLES):
    rows = [(lab, v) for s_, lab, v in TEXROWS if s_ == sec]
    if k:
        L.append(r"    \midrule")
    L.append(rf"    \multicolumn{{7}}{{l}}{{\textit{{{TITLES[sec]}}}}} \\")
    for lab, v in rows:
        c = [cell(x, i) for i, x in enumerate(v)]
        L.append("    " + " & ".join([lab] + c[:2] + [dl(diff(v[0], v[1]))] + c[2:] + [dl(diff(v[2], v[3]))]) + r" \\")
    if len(rows) > 1:  # column means of the section
        m = [avg(v[i] for _, v in rows) for i in range(4)]
        md = [avg(diff(v[0], v[1]) for _, v in rows), avg(diff(v[2], v[3]) for _, v in rows)]
        f = lambda x: "-" if x is None else "%.2f" % x
        L += [r"    \cmidrule(lr){1-7}",
              "    " + " & ".join([r"\textit{Mean}", f(m[0]), f(m[1]), dl(md[0]), f(m[2]), f(m[3]), dl(md[1])]) + r" \\"]
L += [r"    \bottomrule", r"    \end{tabular}", r"    }", r"    \label{table:cross_dataset_MINGLE}", r"\end{table}"]
TEX = os.path.join(RES, "paper_table_cross_dataset_MINGLE.tex")
open(TEX, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("wrote", os.path.abspath(TEX))
