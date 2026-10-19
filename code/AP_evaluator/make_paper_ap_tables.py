"""Rebuild the three EgoGroups_test tables of the paper, by group size G1..G5+, for one metric:
  --metric ap      AP per size and their mean (JRDB_eval.py task_3; the paper's tables)
  --metric f1half  F1 per size, half criterion, and the mean of the five F1s (F1_evaluator size_breakdown)
  --metric f1exact F1 per size, exact members
  --metric mingle  group-box mIoU, P, R, F1 at IoU >= 0.5 (F1_evaluator/mingle_metric.py, pooled)
  --metric mingle_prf  the same without mIoU (P, R, F1)

Tables:

  1. table:results_all_AP_llm_vlm  zero-shot LLMs / VLMs with metadata (single frame vs video) + JRDB-trained baselines
  2. table:rlhf_wilddet_llm        Qwen2.5-7B base -> +SFT -> +RL, with deltas (green = gain, red = loss)
  3. table:rlhf_wilddet_llm (b)    EgoGroups-trained baselines + our SFT / RL models
  4. table:synth_<metric>          like 2, trained on synthetic data (Synth and Synth-Detail)

AP is computed from the detection files (results_<stem>.txt), F1 from the grouping pickles (results_<stem>.pkl, with
people a method leaves out added as singletons, as in by_size_table.py). Scores are cached in a JSON file, so a rerun
only evaluates new or changed files. Bold = best per column (table 1 and 3) or per LLM / VLM block (table 2), as in the paper.
Run from code/AP_evaluator in an env that runs JRDB_eval.py (e.g. py10-dspy):

    python make_paper_ap_tables.py [--metric ap|f1half|f1exact] [--out <tex>] [--jobs 16] [--recompute]
"""
import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--dets", default=os.path.join(HERE, "../../results/detection_files/EgoGroups_test_10"))
ap.add_argument("--groupings", default=os.path.join(HERE, "../../results/grouping_files/EgoGroups_test_10"))
ap.add_argument("--gt", default=os.path.join(HERE, "out/gt_group_10fps_test.txt"))
ap.add_argument("--gt_pkl", default=os.path.join(HERE, "../F1_evaluator/out/gt_group_10fps_test.pkl"))
ap.add_argument("--metric", default="ap", choices=["ap", "f1half", "f1exact", "mingle", "mingle_prf"])
ap.add_argument("--cache", default=os.path.join(HERE, "../../results/ap_cache_EgoGroups_test.json"))
ap.add_argument("--out", default=None, help="default: results/paper_tables_{AP,F1half,F1exact,MINGLE}.tex")
ap.add_argument("--jobs", type=int, default=16)
ap.add_argument("--recompute", action="store_true")
args = ap.parse_args()
TAG = {"ap": "AP", "f1half": "F1half", "f1exact": "F1exact", "mingle": "MINGLE", "mingle_prf": "MINGLE_PRF"}[args.metric]
PRF = args.metric == "mingle_prf"  # same scores (and cache) as mingle, mIoU column dropped
if PRF:
    args.metric = "mingle"
args.out = args.out or os.path.join(HERE, f"../../results/paper_tables_{TAG}.tex")
LAST = "AP" if args.metric == "ap" else "Mean"  # last column: AP = mean AP, Mean = mean of the five F1s
WHAT = ("AP scores" if args.metric == "ap" else
        "F1 scores (" + ("half" if args.metric == "f1half" else "exact") + " criterion)")
SUMMARY = "with AP showing the average performance" if args.metric == "ap" else "with Mean the average of the five F1 scores"
NC = (3 if PRF else 4) if args.metric == "mingle" else 6  # value columns per block
if args.metric == "mingle":
    EXPLAIN = ((r" P, R and F1" if PRF else r" mIoU, P, R and F1") + r" are computed on group boxes (the smallest box around each group of two or more people),"
               r" matched one-to-one by IoU; a match with IoU $\geq 0.5$ is correct (MINGLE metric).")
else:
    EXPLAIN = " $G_1$--$G_5$ represent " + WHAT + " for group sizes 1--5$^+$, " + SUMMARY + "."

# ---------------------------------------------------------------- which detection file fills which cell (None = "-")
# file stem = detection file name without "results_" and ".txt"
QWEN25 = lambda size, vlm: (f"Qwen2.5-VL-{size}-Instruct_vlm_image_detany_3D_p1_visual" if vlm
                            else f"Qwen2.5-{size}-Instruct_llm_detany_3D_p1")
full = lambda stem: stem and "full_" + stem


def V(stem):
    """VLM run tested with p1: use its p1_visual run instead (the training prompt) once that has been scored."""
    vis = stem + "_visual"
    ok = all(os.path.exists(os.path.join(d, f"results_{vis}.{e}")) for d, e in ((args.dets, "txt"), (args.groupings, "pkl")))
    return vis if ok else stem


IS_P1_VLM = lambda s: bool(s) and "_vlm_" in s and s.endswith("_p1")
DAG = lambda s: r"$^\dagger$" if IS_P1_VLM(s) else ""
P1_NOTE = r" $^\dagger$Tested with the p1 prompt; the other VLM rows use p1\_visual, the prompt they were trained with."

# baselines that see one frame; the others take several frames (clips, trajectories or keyframe sequences)
IMAGE_BASELINES = {"MINGLE", "P-HAR"}
ZS_BASELINES = [  # table 1 gray rows: JRDB-trained methods, zero-shot on EgoGroups, under Single Frame or Video by input
    ("All-singletons", None, "NaiveAllSingletons"),
    ("Social-CAD", "social-cad", "ECCV20_JRDBTrain_ep185"),
    ("JRDB-Act", "JRDB-Act", "JRDBAct_JRDBTrain_ep40"),
    ("S3R2", "li2022selfsupervised", "SHGD_JRDBTrain_ep20"),
    ("P-HAR", "han2022panoramic", "PAR_JRDBTrain_ep11"),
    ("RTGD", "Jahangard_IROS_2023", "RTGD_JRDBTrain_ep5"),
    ("GroupTrans", "10286297", "GroupTrans_JRDBTrain_ep199"),
    ("DVT3", "Yokoyama_2025_ICCV", "DVT3_JRDBpretrained_th0.9"),
    ("MINGLE", "liu2025minglevlmssemanticallycomplex", "Qwen2-VL-7B-Instruct-base-mingle_mingle_detany_3D_p1"),
]
ZS_MODELS = [  # (family, [(params, [(method, single stem, video stem)])])
    ("Cosmos-Reason2", [
        ("2B", [("VLM", "Cosmos-Reason2-2B_vlm_image_detany_3D_p1_visual", "full_Cosmos-Reason2-2B_vlm_image_detany_3D_p1_visual"),
                ("LLM", "Cosmos-Reason2-2B_llm_detany_3D_p1", None)]),
        ("8B", [("VLM", None, None),
                ("LLM", "Cosmos-Reason2-8B_llm_detany_3D_p1", "full_Cosmos-Reason2-8B_llm_detany_3D_p1")]),
    ]),
    ("Qwen2.5", [(s, [("VLM", QWEN25(s, True), full(QWEN25(s, True))), ("LLM", QWEN25(s, False), full(QWEN25(s, False)))])
                 for s in ["3B", "7B", "32B", "72B"]]),
    ("Qwen3", [
        ("4B", [("VLM", "Qwen3-VL-4B-Instruct_vlm_image_detany_3D_p1_visual", "full_Qwen3-VL-4B-Instruct_vlm_image_detany_3D_p1_visual"),
                ("LLM", "Qwen3-4B-Instruct-2507_llm_detany_3D_p1", "full_Qwen3-4B-Instruct-2507_llm_detany_3D_p1")]),
        ("30B-A3B", [("VLM", "Qwen3-VL-30B-A3B-Instruct_vlm_image_detany_3D_p1_visual", "full_Qwen3-VL-30B-A3B-Instruct_vlm_image_detany_3D_p1_visual"),
                     ("LLM", "Qwen3-30B-A3B-Instruct-2507_llm_detany_3D_p1", "full_Qwen3-30B-A3B-Instruct-2507_llm_detany_3D_p1")]),
    ]),
]
FT = {  # table 2 / 3: (frames, processing) -> [Base, +SFT, +RL]
    ("Single Frame", "LLM"): [QWEN25("7B", False), "Qwen2.5-7B-Instruct-egogroups-SFT_llm_detany_3D_p1",
                              "Qwen2.5-7B-Instruct-egogroups-RL-ep1_llm_detany_3D_p1"],
    ("Single Frame", "VLM"): [QWEN25("7B", True), "Qwen2.5-VL-7B-Instruct-egogroups-SFT_vlm_image_detany_3D_p1_visual",
                              "Qwen2.5-VL-7B-Instruct-RL1_vlm_image_detany_3D_p1_visual"],
    ("Video", "LLM"): [full(QWEN25("7B", False)), "full_Qwen2.5-7B-Instruct-egogroups-SFT_llm_detany_3D_p1",
                       "full_Qwen2.5-7B-Instruct-egogroups-RL-ep1_llm_detany_3D_p1"],
    ("Video", "VLM"): [full(QWEN25("7B", True)), V("full_Qwen2.5-VL-7B-Instruct-SFT_vlm_image_detany_3D_p1"),
                       V("full_Qwen2.5-VL-7B-Instruct-RL_vlm_image_detany_3D_p1")],
}
EGO_BASELINES = [  # table 3 gray rows: methods trained on EgoGroups_train (checkpoint picked on validation)
    ("Social-CAD", "social-cad", "ECCV20_EgoGroupsTrain_ep40"),
    ("JRDB-Act", "JRDB-Act", "JRDBAct_EgoGroupsTrain_ep20"),
    ("S3R2", "li2022selfsupervised", "SHGD_EgoGroupsTrain_ep100"),  # best val
    ("P-HAR", "han2022panoramic", "PAR_EgoGroupsTrain_wd1e-3_ep19"),
    ("RTGD", "Jahangard_IROS_2023", "RTGD_EgoGroupsTrain_ep40"),  # best val (ep20 = earlier best, ep49 = last)
    ("GroupTrans", "10286297", "GroupTrans_EgoGroupsTrain_ep199"),
    ("DVT3", "Yokoyama_2025_ICCV", "DVT3_EgoGroupsTrain_ep1_th0.50"),
    ("MINGLE", "liu2025minglevlmssemanticallycomplex", "MINGLE_vlm_image_evp_depth_mingle"),
]
SYN = lambda proc, var, step: (f"Qwen2.5-VL-7B-Instruct-{step}-{var}_vlm_image_detany_3D_p1" if proc == "VLM"
                               else f"Qwen2.5-7B-Instruct-{step}-{var}_llm_detany_3D_p1")
SYNTH = {  # table 4: (frames, processing) -> (Base, [(training set, +SFT, +RL)]); VLM synth runs exist only for p1
    ("Single Frame", "LLM"): (FT[("Single Frame", "LLM")][0],
                              [(v, SYN("LLM", v, "SFT"), SYN("LLM", v, "RL")) for v in ["Synth", "Synth-Detail"]]),
    ("Single Frame", "VLM"): (FT[("Single Frame", "VLM")][0],
                              [(v, V(SYN("VLM", v, "SFT")), V(SYN("VLM", v, "RL"))) for v in ["Synth", "Synth-Detail"]]),
    ("Video", "LLM"): (FT[("Video", "LLM")][0],
                       [(v, full(SYN("LLM", v, "SFT")), full(SYN("LLM", v, "RL"))) for v in ["Synth", "Synth-Detail"]]),
    ("Video", "VLM"): (FT[("Video", "VLM")][0], [("Synth", None, None), ("Synth-Detail", None, None)]),
}
# table 5: zero-shot VLMs, one row per prompt; runs with < 95% of the 4500 keyframes are left out ("-")
PROMPTS = [("p1", r"p1"), ("p1_depthanything", r"p1 (Depth Anything)"), ("p1_bbox", r"p1\_bbox"),
           ("p1_visual", r"p1\_visual"), ("p1_visual_depthanything", r"p1\_visual (Depth Anything)"),
           ("p1_visual_only", r"p1\_visual\_only")]
PROMPT_MODELS = [("Cosmos-Reason2", [("2B", "Cosmos-Reason2-2B"), ("8B", "Cosmos-Reason2-8B")]),
                 ("Qwen2.5-VL", [(s, f"Qwen2.5-VL-{s}-Instruct") for s in ["3B", "7B", "32B", "72B"]]),
                 ("Qwen3-VL", [("4B", "Qwen3-VL-4B-Instruct"), ("30B-A3B", "Qwen3-VL-30B-A3B-Instruct")])]
MIN_FRAMES = 4275
COVERAGE = os.path.join(HERE, "../../results/prompt_coverage_EgoGroups_test.tsv")  # written when the runs are scored
cov = dict((r.split("\t")[0], int(r.split("\t")[1])) for r in open(COVERAGE).read().split("\n") if r) if os.path.exists(COVERAGE) else {}


def prompt_stem(model, prompt, video):
    """Stem of a complete run (enough keyframes, scored), else None."""
    s = f"{'full_' if video else ''}{model}_vlm_image_detany_3D_{prompt}"
    ok = cov.get("results_" + s, 0) >= MIN_FRAMES and all(
        os.path.exists(os.path.join(d, f"results_{s}.{e}")) for d, e in ((args.dets, "txt"), (args.groupings, "pkl")))
    return s if ok else None


PROMPT_ROWS = [(fam, params, p, name, prompt_stem(m, p, False), prompt_stem(m, p, True))
               for fam, sizes in PROMPT_MODELS for params, m in sizes for p, name in PROMPTS]
# table 6: zero-shot VLMs without metadata (baseline1: the model returns the group boxes itself; F1 / MINGLE match
# its boxes to the annotated people by IoU, det_txt_to_pkl.py --hungarian)
BASELINE_ROWS = [(fam, params, prompt_stem(m, "baseline1", False), prompt_stem(m, "baseline1", True))
                 for fam, sizes in PROMPT_MODELS for params, m in sizes]


# ---------------------------------------------------------------- AP (cached)
def det_path(stem):
    if args.metric == "ap":
        return os.path.join(args.dets, f"results_{stem}.txt")
    return os.path.join(args.groupings, f"results_{stem}.pkl")


def run_ap(stem):
    """[G1, G2, G3, G4, G5+, AP] in percent, from JRDB_eval.py task_3 (it prints AP, then G1..G5+)."""
    out = subprocess.run([sys.executable, "JRDB_eval.py", "-g", args.gt, "-d", det_path(stem), "-t", "task_3",
                          "--labelmap", "./label_map/task_3.pbtxt", "--output", "all"],
                         cwd=HERE, capture_output=True, text=True).stdout
    v = [float(x) * 100 for x in re.findall(r"^[0-9.]+$", out, re.M)]
    if len(v) < 6:
        raise RuntimeError(f"JRDB_eval.py gave no AP for {stem}")
    return v[1:6] + [v[0]]


def run_f1(stem, crit, groupings, gt_pkl):
    """[G1, G2, G3, G4, G5+, mean of the five] F1 in percent (same call as F1_evaluator/by_size_table.py)."""
    import contextlib, io, pickle
    sys.path.insert(0, os.path.join(HERE, "../F1_evaluator"))
    from utils_eval import Evaluation
    gt = pickle.load(open(gt_pkl, "rb"))
    pred = pickle.load(open(os.path.join(groupings, f"results_{stem}.pkl"), "rb"))
    pred = {k: pred[k] if k in pred else [] for k in gt}
    with contextlib.redirect_stdout(io.StringIO()):
        sizes = Evaluation(whole=False, alone=True, pre_dict=pred, GT_dict=gt, crit=crit).size_breakdown()
    f = [sizes[b]["F1"] * 100 for b in ["G1", "G2", "G3", "G4", "G5+"]]
    return f + [sum(f) / 5]


def run_mingle(stems):
    """{stem: [mIoU, P, R, F1]} in percent at IoU 0.5, pooled (one mingle_metric.py run over the needed pickles)."""
    import csv, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "m.csv")
        subprocess.run([sys.executable, "mingle_metric.py", "--dets", os.path.abspath(args.groupings), "--gt",
                        os.path.abspath(args.gt_pkl), "--csv", out, "--iou", "0.5", "--match", ",".join(stems)],
                       cwd=os.path.join(HERE, "../F1_evaluator"), check=True, capture_output=True)
        rows = {r["name"]: r for r in csv.DictReader(open(out))}
    return {s: [float(rows[f"results_{s}.pkl"][k]) for k in ("mIoU", "P", "R", "F1")] for s in stems}


stems = {s for _, _, s in ZS_BASELINES + EGO_BASELINES if s}
stems |= {s for _, sizes in ZS_MODELS for _, rows in sizes for _, a, b in rows for s in (a, b) if s}
stems |= {s for v in FT.values() for s in v}
stems |= {s for base, runs in SYNTH.values() for s in [base] + [x for _, a, b in runs for x in (a, b)] if s}
stems |= {s for *_, a, b in PROMPT_ROWS + BASELINE_ROWS for s in (a, b) if s}
missing = sorted(s for s in stems if not os.path.exists(det_path(s)))
if missing:
    sys.exit("missing detection files:\n  " + "\n  ".join(det_path(s) for s in missing))
cache = {} if args.recompute or not os.path.exists(args.cache) else json.load(open(args.cache))
key = lambda s: ("" if args.metric == "ap" else args.metric + ":") + f"{s}:{os.path.getmtime(det_path(s)):.0f}"
todo = sorted(s for s in stems if key(s) not in cache)
if todo:
    print(f"evaluating {len(todo)} files ({args.metric}) ...", file=sys.stderr)
    if args.metric == "ap":
        with ThreadPoolExecutor(args.jobs) as ex:
            res = list(ex.map(run_ap, todo))
    elif args.metric == "mingle":
        m = run_mingle(todo)
        res = [m[s] for s in todo]
    else:
        crit = "half" if args.metric == "f1half" else "exact"
        with ProcessPoolExecutor(args.jobs) as ex:
            res = list(ex.map(run_f1, todo, [crit] * len(todo), [args.groupings] * len(todo), [args.gt_pkl] * len(todo)))
    for s, v in zip(todo, res):
        cache[key(s)] = v
    json.dump(cache, open(args.cache, "w"), indent=1)
AP = {s: cache[key(s)][1:] if PRF else cache[key(s)] for s in stems}
get = lambda s: AP[s] if s else None


# ---------------------------------------------------------------- LaTeX helpers
def fmt(v, best=None):
    if v is None:
        return "-"
    t = f"{v:.2f}"
    return rf"\textbf{{{t}}}" if best is not None and round(v, 2) == round(best, 2) else t


def col_best(rows):
    """rows: lists of NC values or None -> per-column max."""
    vals = [r for r in rows if r]
    return [max(r[i] for r in vals) for i in range(NC)] if vals else [None] * NC


GRAY = r"\cellcolor{gray!15}"
GH = ("P & R & F1" if PRF else "mIoU & P & R & F1") if args.metric == "mingle" else "G$_1$ & G$_2$ & G$_3$ & G$_4$ & G$_5$ & " + LAST
C = lambda a, b: rf"\cmidrule(lr){{{a}-{b}}}"


def table1():
    # All-singletons uses no input, so it is shown in both blocks
    img = lambda name: name in IMAGE_BASELINES or name == "All-singletons"
    vid = lambda name: name not in IMAGE_BASELINES
    single = [get(s) for name, _, s in ZS_BASELINES if img(name)]
    single += [get(a) for _, sizes in ZS_MODELS for _, rows in sizes for _, a, _ in rows]
    video = [get(s) for name, _, s in ZS_BASELINES if vid(name)]
    video += [get(b) for _, sizes in ZS_MODELS for _, rows in sizes for _, _, b in rows]
    bs, bv = col_best(single), col_best(video)
    L = [r"\begin{table*}[h!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Zero-shot performance comparison on the \dataset{} dataset with metadata provided to the model. Model combination indicates the backbone family, number of parameters, and modality. Baselines (gray) are shown under Single Frame or Video according to their input (All-singletons uses no input and is shown under both)." + EXPLAIN + "}",
         r"    \resizebox{\linewidth}{!}{", r"    \begin{tabular}{lll" + "c" * 2 * NC + "}", r"    \toprule",
         rf"        & & & \multicolumn{{{NC}}}{{c}}{{\textbf{{Single Frame}}}}", rf"        & \multicolumn{{{NC}}}{{c}}{{\textbf{{Video}}}} \\",
         "        " + C(4, 3 + NC) + " " + C(4 + NC, 3 + 2 * NC),
         rf"    Model & Params & Method & {GH}", rf"        & {GH} \\", r"    \midrule"]
    for name, cite, s in ZS_BASELINES:
        v = get(s)
        label = name + (rf"~\cite{{{cite}}}" if cite else "")
        cells = [label, "-", "-"] + ([fmt(v[i], bs[i]) for i in range(NC)] if img(name) else ["-"] * NC) \
            + ([fmt(v[i], bv[i]) for i in range(NC)] if vid(name) else ["-"] * NC)
        L.append("    " + " & ".join(GRAY + c for c in cells) + r" \\")
    L.append(r"    \midrule")
    for fi, (family, sizes) in enumerate(ZS_MODELS):
        n = sum(len(rows) for _, rows in sizes)
        L.append(rf"        \multirow{{{n}}}{{*}}{{\rotatebox{{10}}{{\shortstack{{{family}}}}}}}")
        for si, (params, rows) in enumerate(sizes):
            for ri, (method, a, b) in enumerate(rows):
                va, vb = get(a), get(b)
                p = rf"\multirow{{{len(rows)}}}{{*}}{{{params}}}" if ri == 0 else ""
                cells = [p, method] + [fmt(va[i], bs[i]) if va else "-" for i in range(NC)] \
                    + [fmt(vb[i], bv[i]) if vb else "-" for i in range(NC)]
                L.append("        & " + " & ".join(cells) + r" \\")
            if si < len(sizes) - 1:
                L.append("        " + C(2, 3 + 2 * NC))
        L.append(r"    \midrule" if fi < len(ZS_MODELS) - 1 else r"    \bottomrule")
    L += [r"    \end{tabular}", r"    }", r"    \label{table:results_all_" + TAG + "_llm_vlm}", r"\end{table*}"]
    return "\n".join(L)


def table1b():
    """Table 1 with one block of columns: a Single Frame section (All-singletons, image baselines, zero-shot models on
    one frame), then a Video section (All-singletons, video baselines, zero-shot models on video). Bold per section."""
    L = [r"\begin{table*}[h!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Zero-shot performance comparison on the \dataset{} dataset with metadata provided to the model. Model combination indicates the backbone family, number of parameters, and modality. Baselines (gray) are shown in the Single Frame or Video section according to their input (All-singletons uses no input and is shown in both)." + EXPLAIN + "}",
         r"    \resizebox{\linewidth}{!}{", r"    \begin{tabular}{lll" + "c" * NC + "}", r"    \toprule"]
    L.insert(-2, r"    \renewcommand{\arraystretch}{0.85}")
    for fi, frames in enumerate(["Single Frame", "Video"]):
        video = frames == "Video"
        base = [(n, c, s) for n, c, s in ZS_BASELINES
                if n == "All-singletons" or (n in IMAGE_BASELINES) != video]
        pick = lambda a, b: b if video else a
        best = col_best([get(s) for *_, s in base]
                        + [get(pick(a, b)) for _, sizes in ZS_MODELS for _, rows in sizes for _, a, b in rows])
        if fi:
            L.append(r"    \midrule")
        L += [rf"    \multicolumn{{{3 + NC}}}{{c}}{{\textbf{{{frames}}}}} \\", "    " + C(1, 3 + NC),
              rf"    Model & Params & Method & {GH} \\", r"    \midrule"]
        for name, cite, s in base:
            v = get(s)
            label = name + (rf"~\cite{{{cite}}}" if cite else "")
            cells = [label, "-", "-"] + [fmt(v[i], best[i]) for i in range(NC)]
            L.append("    " + " & ".join(GRAY + c for c in cells) + r" \\")
        L.append(r"    \midrule")
        # only runs that exist in this section (no all-"-" rows)
        fams = [(family, [(params, [(m, get(pick(a, b))) for m, a, b in rows if get(pick(a, b))]) for params, rows in sizes])
                for family, sizes in ZS_MODELS]
        fams = [(family, [(p, r) for p, r in sizes if r]) for family, sizes in fams]
        for gi, (family, sizes) in enumerate(fams):
            n = sum(len(rows) for _, rows in sizes)
            L.append(rf"        \multirow{{{n}}}{{*}}{{{family}}}")
            for si, (params, rows) in enumerate(sizes):
                for ri, (method, v) in enumerate(rows):
                    p = (rf"\multirow{{{len(rows)}}}{{*}}{{{params}}}" if len(rows) > 1 else params) if ri == 0 else ""
                    L.append("        & " + " & ".join([p, method] + [fmt(v[i], best[i]) for i in range(NC)]) + r" \\")
            if gi < len(fams) - 1:
                L.append("    " + C(1, 3 + NC))
    L += [r"    \bottomrule", r"    \end{tabular}", r"    }", r"    \label{table:results_all_" + TAG + "_llm_vlm_sections}", r"\end{table*}"]
    return "\n".join(L)


def delta(v, ref):
    d = v - ref
    color = r"green!55!black" if d >= 0 else r"red!70!black"
    return rf"\,{{\scriptsize\textcolor{{{color}}}{{({d:+.2f})}}}}"


def table2():
    L = [r"\begin{table*}[b!]", r"    \centering", r"    \footnotesize",
         r"    \caption{ Train EgoGroups-Train, Test EgoGroups-Test (SFT vs.\ Base; each RL checkpoint vs.\ SFT) (\textcolor{green!55!black}{green} = gain, \textcolor{red!70!black}{red} = loss)." + ("" if args.metric == "ap" else EXPLAIN)
         + (P1_NOTE if any(IS_P1_VLM(x) for v in FT.values() for x in v) else "") + "}",
         r"    \resizebox{\linewidth}{!}{", r"    \begingroup", r"    \setlength{\aboverulesep}{0pt}",
         r"    \setlength{\belowrulesep}{0pt}", r"    \begin{tabular}{c|ll" + "c" * NC + "}", r"    \toprule",
         r"    \multirow{16}{*}{\rotatebox[origin=c]{90}{\textbf{Qwen2.5-7B}}}"]
    for fi, frames in enumerate(["Single Frame", "Video"]):
        if fi:
            L.append("    " + C(2, 3 + NC))
        L += [rf"        & \multicolumn{{{2 + NC}}}{{c}}{{\textbf{{{frames}}}}} \\", "    " + C(2, 3 + NC),
              rf"        & Processing & Variant & {GH} \\", "    " + C(2, 3 + NC)]
        for pi, proc in enumerate(["LLM", "VLM"]):
            base, sft, rl = (get(s) for s in FT[(frames, proc)])
            best = col_best([base, sft, rl])
            L.append(rf"        & \multirow{{3}}{{*}}{{{proc}}}")
            L.append("            & Base & " + " & ".join(fmt(base[i], best[i]) for i in range(NC)) + r" \\")
            for name, v, ref, st in (("+SFT", sft, base, FT[(frames, proc)][1]), ("+RL", rl, sft, FT[(frames, proc)][2])):
                L.append(f"        & & {name}{DAG(st)}")
                L += [f"            & {fmt(v[i], best[i])}{delta(v[i], ref[i])}" for i in range(NC)]
                L[-1] += r" \\"
            if pi == 0:
                L.append("    " + C(2, 3 + NC))
    L += [r"    \bottomrule", r"    \end{tabular}", r"    \endgroup", r"    }", r"    \label{table:rlhf_wilddet_llm" + ("" if args.metric == "ap" else "_" + TAG) + "}", r"\end{table*}"]
    return "\n".join(L)


def table3():
    """EgoGroups-trained baselines (gray, under Single Frame or Video by their input) and our SFT / RL models."""
    ours = [(f"{proc} + {var}", FT[("Single Frame", proc)][k], FT[("Video", proc)][k])
            for proc in ["LLM", "VLM"] for k, var in ((1, "SFT"), (2, "RL"))]
    img = lambda name: name in IMAGE_BASELINES
    bs = col_best([get(s) for name, _, s in EGO_BASELINES if img(name)] + [get(a) for _, a, _ in ours])
    bv = col_best([get(s) for name, _, s in EGO_BASELINES if not img(name)] + [get(b) for _, _, b in ours])
    vals = lambda v, best: [fmt(v[i], best[i]) for i in range(NC)] if v else ["-"] * NC
    L = [r"\begin{table*}[b!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Train EgoGroups-Train, Test EgoGroups-Test. Baselines are shown in gray, under Single Frame or Video according to their input." + EXPLAIN
         + (P1_NOTE if any(IS_P1_VLM(x) for v in FT.values() for x in v[1:]) else "") + "}",
         r"    \resizebox{\linewidth}{!}{", r"    \begingroup", r"    \setlength{\aboverulesep}{0pt}",
         r"    \setlength{\belowrulesep}{0pt}", r"    \begin{tabular}{lll" + "c" * 2 * NC + "}", r"    \toprule",
         rf"        & & & \multicolumn{{{NC}}}{{c}}{{\textbf{{Single Frame}}}}", rf"        & \multicolumn{{{NC}}}{{c}}{{\textbf{{Video}}}} \\",
         "        " + C(4, 3 + NC) + " " + C(4 + NC, 3 + 2 * NC),
         rf"    Model & Params & Method & {GH}", rf"        & {GH} \\", r"    \midrule"]
    for name, cite, s in EGO_BASELINES:
        v = get(s)
        cells = [rf"{name}~\cite{{{cite}}}", "-", "-"] + (vals(v, bs) + ["-"] * NC if img(name) else ["-"] * NC + vals(v, bv))
        L.append("    " + " & ".join(GRAY + c for c in cells) + r" \\")
    L.append(r"    \midrule")
    for i, (method, a, b) in enumerate(ours):
        lead = r"    \multirow{4}{*}{Qwen2.5} & \multirow{4}{*}{7B}" if i == 0 else "    &"
        dag = DAG(a) or DAG(b)
        L.append(lead + f"\n            & {method}{dag} & " + " & ".join(vals(get(a), bs) + vals(get(b), bv)) + r" \\")
        if i == 1:
            L.append("    " + C(3, 3 + 2 * NC))
    L += [r"    \bottomrule", r"    \end{tabular}", r"    \endgroup", r"    }", r"    \label{table:" + ("rlhf_wilddet_llm" if args.metric == "ap" else "train_egogroups_" + TAG) + "}", r"\end{table*}"]
    return "\n".join(L)


def table3b():
    """Table 3 with one block of columns, like table 2: a Single Frame section (image baselines + our single-frame
    models), then a Video section (video baselines + our video models). Bold = best per column within a section."""
    L = [r"\begin{table*}[b!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Train EgoGroups-Train, Test EgoGroups-Test. Baselines are shown in gray, in the Single Frame or Video section according to their input." + EXPLAIN
         + (P1_NOTE if any(IS_P1_VLM(x) for v in FT.values() for x in v[1:]) else "") + "}",
         r"    \resizebox{\linewidth}{!}{", r"    \begingroup", r"    \setlength{\aboverulesep}{0pt}",
         r"    \setlength{\belowrulesep}{0pt}", r"    \begin{tabular}{lll" + "c" * NC + "}", r"    \toprule"]
    for fi, frames in enumerate(["Single Frame", "Video"]):
        base = [(n, c, s) for n, c, s in EGO_BASELINES if (n in IMAGE_BASELINES) == (frames == "Single Frame")]
        ours = [(f"{proc} + {var}", FT[(frames, proc)][k]) for proc in ["LLM", "VLM"] for k, var in ((1, "SFT"), (2, "RL"))]
        best = col_best([get(s) for *_, s in base] + [get(s) for _, s in ours])
        if fi:
            L.append(r"    \midrule")
        L += [rf"    \multicolumn{{{3 + NC}}}{{c}}{{\textbf{{{frames}}}}} \\", "    " + C(1, 3 + NC),
              rf"    Model & Params & Method & {GH} \\", "    " + C(1, 3 + NC)]
        for name, cite, s in base:
            v = get(s)
            cells = [rf"{name}~\cite{{{cite}}}", "-", "-"] + ([fmt(v[i], best[i]) for i in range(NC)] if v else ["-"] * NC)
            L.append("    " + " & ".join(GRAY + c for c in cells) + r" \\")
        L.append("    " + C(1, 3 + NC))
        for i, (method, s) in enumerate(ours):
            lead = r"    \multirow{4}{*}{Qwen2.5} & \multirow{4}{*}{7B}" if i == 0 else "    &"
            L.append(lead + f"\n            & {method}{DAG(s)} & " + " & ".join(fmt(get(s)[k], best[k]) for k in range(NC)) + r" \\")
    L += [r"    \bottomrule", r"    \end{tabular}", r"    \endgroup", r"    }", r"    \label{table:train_egogroups_sections_" + TAG + "}", r"\end{table*}"]
    return "\n".join(L)


def table4():
    """Like table 2, for synthetic training data: Base, then +SFT / +RL per training set (SFT vs Base, RL vs its SFT)."""
    dag = lambda proc, s: DAG(s)
    n_rows = 4 + sum(1 + 2 * len(runs) for _, runs in SYNTH.values())  # 2 header rows per frame setting + data rows
    L = [r"\begin{table*}[b!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Train EgoGroups-Synth, Test EgoGroups-Test (each SFT vs.\ Base; each RL vs.\ its SFT) (\textcolor{green!55!black}{green} = gain, \textcolor{red!70!black}{red} = loss)."
         + EXPLAIN + (P1_NOTE if any(IS_P1_VLM(x) for _, runs in SYNTH.values() for _, a, b in runs for x in (a, b)) else "") + "}",
         r"    \resizebox{\linewidth}{!}{", r"    \begingroup", r"    \setlength{\aboverulesep}{0pt}",
         r"    \setlength{\belowrulesep}{0pt}", r"    \begin{tabular}{c|ll" + "c" * NC + "}", r"    \toprule",
         rf"    \multirow{{{n_rows}}}{{*}}{{\rotatebox[origin=c]{{90}}{{\textbf{{Qwen2.5-7B}}}}}}"]
    for fi, frames in enumerate(["Single Frame", "Video"]):
        if fi:
            L.append("    " + C(2, 3 + NC))
        L += [rf"        & \multicolumn{{{2 + NC}}}{{c}}{{\textbf{{{frames}}}}} \\", "    " + C(2, 3 + NC),
              rf"        & Processing & Variant & {GH} \\", "    " + C(2, 3 + NC)]
        for pi, proc in enumerate(["LLM", "VLM"]):
            base_s, runs = SYNTH[(frames, proc)]
            base = get(base_s)
            best = col_best([base] + [get(x) for _, a, b in runs for x in (a, b)])
            L.append(rf"        & \multirow{{{1 + 2 * len(runs)}}}{{*}}{{{proc}}}")
            L.append("            & Base & " + " & ".join(fmt(base[i], best[i]) for i in range(NC)) + r" \\")
            for var, a, b in runs:
                for name, s, ref in ((f"+SFT ({var})", a, base), (f"+RL ({var})", b, get(a))):
                    v = get(s)
                    L.append(f"        & & {name}{dag(proc, s)}")
                    if v is None:
                        L[-1] += " & " + " & ".join(["-"] * NC) + r" \\"
                        continue
                    L += [f"            & {fmt(v[i], best[i])}" + (delta(v[i], ref[i]) if ref else "") for i in range(NC)]
                    L[-1] += r" \\"
            if pi == 0:
                L.append("    " + C(2, 3 + NC))
    L += [r"    \bottomrule", r"    \end{tabular}", r"    \endgroup", r"    }", r"    \label{table:synth_" + TAG + "}", r"\end{table*}"]
    return "\n".join(L)


def table5():
    """Zero-shot VLMs, one row per prompt (single frame vs video). Bold = best prompt per model and column."""
    L = [r"\begin{table*}[h!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Zero-shot VLMs on the \dataset{} dataset with different prompts. p1: frame and person metadata (ids, boxes, 3D positions from DetAny3D);"
         r" (Depth Anything): the same with 3D positions from Depth Anything; p1\_bbox: bare frame plus a separate list of boxes;"
         r" p1\_visual: boxes and ids drawn on the frame, plus 3D positions; p1\_visual\_only: drawn frame and person ids, no coordinates."
         r" Bold = best prompt per model. Runs with fewer than 95\% of the 4,500 keyframes are left out (-)." + EXPLAIN + "}",
         r"    \resizebox{\linewidth}{!}{", r"    \begin{tabular}{lll" + "c" * 2 * NC + "}", r"    \toprule",
         rf"        & & & \multicolumn{{{NC}}}{{c}}{{\textbf{{Single Frame}}}}", rf"        & \multicolumn{{{NC}}}{{c}}{{\textbf{{Video}}}} \\",
         "        " + C(4, 3 + NC) + " " + C(4 + NC, 3 + 2 * NC),
         rf"    Model & Params & Prompt & {GH}", rf"        & {GH} \\", r"    \midrule"]
    for fi, (family, sizes) in enumerate(PROMPT_MODELS):
        L.append(rf"        \multirow{{{len(sizes) * len(PROMPTS)}}}{{*}}{{{family}}}")
        for si, (params, _) in enumerate(sizes):
            rows = [r for r in PROMPT_ROWS if r[0] == family and r[1] == params]
            bs, bv = col_best([get(r[4]) for r in rows]), col_best([get(r[5]) for r in rows])
            for ri, (_, _, _, name, a, b) in enumerate(rows):
                va, vb = get(a), get(b)
                p = rf"\multirow{{{len(rows)}}}{{*}}{{{params}}}" if ri == 0 else ""
                cells = [p, name] + [fmt(va[i], bs[i]) if va else "-" for i in range(NC)] \
                    + [fmt(vb[i], bv[i]) if vb else "-" for i in range(NC)]
                L.append("        & " + " & ".join(cells) + r" \\")
            if si < len(sizes) - 1:
                L.append("        " + C(2, 3 + 2 * NC))
        L.append(r"    \midrule" if fi < len(PROMPT_MODELS) - 1 else r"    \bottomrule")
    L += [r"    \end{tabular}", r"    }", r"    \label{table:prompts_" + TAG + "}", r"\end{table*}"]
    return "\n".join(L)


def table6():
    """Zero-shot VLMs without metadata (baseline1). Bold = best per column."""
    L = [r"\begin{table*}[h!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Zero-shot performance on the \dataset{} dataset \textbf{without} metadata provided to the model (\textbf{VLM baseline}):"
         r" the model gets only the frame(s) and returns the boxes of the people in each group."
         r" Model combination indicates the backbone family and number of parameters."
         r" Runs with fewer than 95\% of the 4,500 keyframes are left out (-)." + EXPLAIN + "}",
         r"    \resizebox{0.99\linewidth}{!}{", r"    \begin{tabular}{ll" + "c" * 2 * NC + "}", r"    \toprule",
         r"        & \multirow{2}{*}{Params}", rf"        & \multicolumn{{{NC}}}{{c}}{{\textbf{{Single Frame}}}}",
         rf"        & \multicolumn{{{NC}}}{{c}}{{\textbf{{Video}}}} \\", "        " + C(3, 2 + NC) + " " + C(3 + NC, 2 + 2 * NC),
         rf"    Model & & {GH} & {GH} \\", r"    \midrule"]
    bs, bv = col_best([get(r[2]) for r in BASELINE_ROWS]), col_best([get(r[3]) for r in BASELINE_ROWS])
    for fi, (family, sizes) in enumerate(PROMPT_MODELS):
        rows = [r for r in BASELINE_ROWS if r[0] == family]
        for ri, (_, params, a, b) in enumerate(rows):
            va, vb = get(a), get(b)
            m = rf"\multirow{{{len(rows)}}}{{*}}{{\rotatebox{{10}}{{\shortstack{{{family}}}}}}}" if ri == 0 else ""
            cells = [m, params] + [fmt(va[i], bs[i]) if va else "-" for i in range(NC)] \
                + [fmt(vb[i], bv[i]) if vb else "-" for i in range(NC)]
            L.append("        " + " & ".join(cells) + r" \\")
        L.append(r"    \midrule" if fi < len(PROMPT_MODELS) - 1 else r"    \bottomrule")
    L += [r"    \end{tabular}", r"    }", r"    \label{table:results_baselines_" + TAG + "}", r"\end{table*}"]
    return "\n".join(L)


RUN_STATS = os.path.join(HERE, "../../results/baseline1_run_stats_EgoGroups_test.tsv")  # scripts/baseline1_run_stats.py


def table7():
    """Run statistics of the baseline1 runs of table 6 (same for every metric)."""
    if not os.path.exists(RUN_STATS):
        return "% table 7: run code/scripts/baseline1_run_stats.py first"
    st = {r[0]: r[1:] for r in (l.split("\t") for l in open(RUN_STATS).read().split("\n")[1:] if l)}
    H = r"Saved & Missing & Last clip & 5/5 & Empty & Med. & Max & $>$50"
    L = [r"\begin{table*}[h!]", r"    \centering", r"    \footnotesize",
         r"    \caption{Zero-shot VLM baseline runs of Table~\ref{table:results_baselines_" + TAG + r"} on the 4,500 keyframes (900 clips $\times$ 5) of \dataset{}."
         r" Saved: keyframes with an answer (a failed call, e.g. an output cut at the token limit that cannot be parsed, saves nothing); Missing: 4,500 $-$ Saved;"
         r" Last clip: last clip reached (a run that stopped early ends before 900); 5/5: clips with all 5 keyframes;"
         r" Empty: answers with no group; Med./Max: boxes per non-empty answer; $>$50: answers with more than 50 boxes."
         r" Runs with fewer than 95\% of the keyframes are left out of Table~\ref{table:results_baselines_" + TAG + "}.}",
         r"    \resizebox{0.99\linewidth}{!}{", r"    \begin{tabular}{ll" + "c" * 16 + "}", r"    \toprule",
         r"        & \multirow{2}{*}{Params}", r"        & \multicolumn{8}{c}{\textbf{Single Frame}}",
         r"        & \multicolumn{8}{c}{\textbf{Video}} \\", "        " + C(3, 10) + " " + C(11, 18),
         rf"    Model & & {H} & {H} \\", r"    \midrule"]
    for fi, (family, sizes) in enumerate(PROMPT_MODELS):
        for si, (params, m) in enumerate(sizes):
            cells = [rf"\multirow{{{len(sizes)}}}{{*}}{{\rotatebox{{10}}{{\shortstack{{{family}}}}}}}" if si == 0 else "", params]
            for pre in ("results", "results_full"):
                r = st.get(f"{pre}_{m}")
                if r is None:
                    cells += ["-"] * 8
                    continue
                n, last, full, empty, med, mx, o50 = r
                cells += [f"{int(n):,}", f"{4500 - int(n):,}", last[5:9]] + [f"{int(x):,}" for x in (full, empty, med, mx, o50)]
            L.append("        " + " & ".join(cells) + r" \\")
        L.append(r"    \midrule" if fi < len(PROMPT_MODELS) - 1 else r"    \bottomrule")
    L += [r"    \end{tabular}", r"    }", r"    \label{table:baseline_runs_" + TAG + "}", r"\end{table*}"]
    return "\n".join(L)


tex = "\n\n".join([table1(), table1b(), table2(), table3(), table3b(), table4(), table5(), table6(), table7()]) + "\n"
open(args.out, "w").write(tex)
print(tex)
print("wrote", args.out, file=sys.stderr)
