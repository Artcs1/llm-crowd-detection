"""JRDB val + test table with the MINGLE group-region metric (mingle_det.py: mIoU / P / R / F1 at IoU 0.5, pooled).

Only methods that never trained on the 7 JRDB val scenes: the baselines trained on the 20 JRDB train scenes (epoch
picked on val), DVT3's released JRDB checkpoint (same split) and MINGLE (trained on its own street-level data).
Val = AP_evaluator/out/gt_group_jrdb_val.txt (365 keyframes), test = out/gt_group.txt (1829 keyframes).
Detection files: results/detection_files/JRDB_val_15 and JRDB_fixed_gold_15; a missing file is shown as "-".

    python make_jrdb_mingle_table.py [--out ../../results/paper_table_JRDB_MINGLE.tex]
"""
import argparse
import os

from mingle_det import load, score

HERE = os.path.dirname(os.path.abspath(__file__))
DET = os.path.join(HERE, "..", "..", "results", "detection_files")
ap = argparse.ArgumentParser()
ap.add_argument("--val", choices=["v1", "v2"], default="v2",
                help="v2: val GT / detections with the JRDB test rule (gt_group_jrdb_val_v2.txt, JRDB_val_v2_15: no_eval and "
                     "out-of-image people left out, all group difficulties); v1: the baselines' original val GT")
ap.add_argument("--out", default=None)
args = ap.parse_args()
args.out = args.out or os.path.join(HERE, "..", "..", "results",
                                    "paper_table_JRDB_MINGLE%s.tex" % ("_valv2" if args.val == "v2" else ""))

SECTIONS = [  # (section title, [(name, cite key, val file stem, test file stem)])
    ("Trained on JRDB", [
        ("Social-CAD", "social-cad", "ECCV20_JRDBTrain_ep185", "ECCV20_JRDBTrain_ep185"),
        ("JRDB-Act", "JRDB-Act", "JRDBAct_JRDBTrain_ep40", "JRDBAct_JRDBTrain_ep40"),
        ("S3R2", "li2022selfsupervised", "SHGD_JRDBTrain_ep20", "SHGD_JRDBTrain_ep20"),
        ("P-HAR", "han2022panoramic", "PHAR_JRDBTrain_ep11", "PHAR_JRDBTrain_ep11"),
        ("RTGD", "Jahangard_IROS_2023", "RTGD_JRDBTrain_ep5", "RTGD_JRDBTrain_ep5"),
        ("GroupTrans", "10286297", "GroupTrans_JRDBTrain_ep199", "GroupTrans_JRDBTrain_ep199"),
        ("DVT3", "Yokoyama_2025_ICCV", "DVT3_JRDBpretrained_th0.9", "DVT3_JRDBpretrained_th0.9"),
    ]),
    ("Trained on \\dataset{} (zero-shot on JRDB)", [  # the checkpoints of the EgoGroups tables
        ("Social-CAD", "social-cad", "ECCV20_EgoGroupsTrain_ep40", "ECCV20_EgoGroupsTrain_ep40"),
        ("JRDB-Act", "JRDB-Act", "JRDBAct_EgoGroupsTrain_ep20", "JRDBAct_EgoGroupsTrain_ep20"),
        ("S3R2", "li2022selfsupervised", "SHGD_EgoGroupsTrain_ep100", "SHGD_EgoGroupsTrain_ep100"),
        ("P-HAR", "han2022panoramic", "PAR_EgoGroupsTrain_wd1e-3_ep19", "PAR_EgoGroupsTrain_wd1e-3_ep19"),
        ("RTGD", "Jahangard_IROS_2023", "RTGD_EgoGroupsTrain_ep40", "RTGD_EgoGroupsTrain_ep40"),
        ("GroupTrans", "10286297", "GroupTrans_EgoGroupsTrain_ep199", "GroupTrans_EgoGroupsTrain_ep199"),
        ("DVT3", "Yokoyama_2025_ICCV", "DVT3_EgoGroupsTrain_ep1_th0.50", "DVT3_EgoGroupsTrain_ep1_th0.50"),
    ]),
    ("Ours, trained on \\dataset{} (zero-shot on JRDB)", [  # Qwen2.5(-VL)-7B SFT / RL on EgoGroups_train
        ("Ours LLM SFT (single frame)", None, "Qwen2.5-7B-Instruct-SFT-EgoGroupTrain_llm_detany_3D_p1", "Qwen2.5-7B-Instruct-SFT-EgoGroupTrain_llm_detany_3D_p1"),
        ("Ours LLM RL (single frame)", None, "Qwen2.5-7B-Instruct-RL-EgoGroupTrain_llm_detany_3D_p1", "Qwen2.5-7B-Instruct-RL-EgoGroupTrain_llm_detany_3D_p1"),
        ("Ours LLM SFT (video)", None, "full_Qwen2.5-7B-Instruct-SFT-EgoGroupTrain_llm_detany_3D_p1", "full_Qwen2.5-7B-Instruct-SFT-EgoGroupTrain_llm_detany_3D_p1"),
        ("Ours LLM RL (video)", None, "full_Qwen2.5-7B-Instruct-RL-EgoGroupTrain_llm_detany_3D_p1", "full_Qwen2.5-7B-Instruct-RL-EgoGroupTrain_llm_detany_3D_p1"),
        ("Ours VLM SFT (single frame)", None, "Qwen2.5-VL-7B-Instruct-SFT-EgoGroupTrain_vlm_image_detany_3D_p1_visual", "Qwen2.5-VL-7B-Instruct-SFT-EgoGroupTrain_vlm_image_detany_3D_p1_visual"),
        ("Ours VLM RL (single frame)", None, "Qwen2.5-VL-7B-Instruct-RL-EgoGroupTrain_vlm_image_detany_3D_p1_visual", "Qwen2.5-VL-7B-Instruct-RL-EgoGroupTrain_vlm_image_detany_3D_p1_visual"),
        ("Ours VLM SFT (video)", None, "full_Qwen2.5-VL-7B-Instruct-SFT-EgoGroupTrain_vlm_image_detany_3D_p1_visual", "full_Qwen2.5-VL-7B-Instruct-SFT-EgoGroupTrain_vlm_image_detany_3D_p1_visual"),
        ("Ours VLM RL (video)", None, "full_Qwen2.5-VL-7B-Instruct-RL-EgoGroupTrain_vlm_image_detany_3D_p1_visual", "full_Qwen2.5-VL-7B-Instruct-RL-EgoGroupTrain_vlm_image_detany_3D_p1_visual"),
    ]),
    ("Trained on its own data", [
        ("MINGLE", "liu2025minglevlmssemanticallycomplex", "MINGLE-1.0", "MINGLE-1.0"),
    ]),
]
SETS = [("Val", "JRDB_val_v2_15", "out/gt_group_jrdb_val_v2.txt") if args.val == "v2" else
        ("Val", "JRDB_val_15", "out/gt_group_jrdb_val.txt"), ("Test", "JRDB_fixed_gold_15", "out/gt_group.txt")]

gts = {name: load(os.path.join(HERE, gt)) for name, _, gt in SETS}
vals = {}
for sec, methods in SECTIONS:
    for m, _, sv, st in methods:
        for (name, folder, _), stem in zip(SETS, (sv, st)):
            f = os.path.join(DET, folder, f"results_{stem}.txt")
            vals[sec, m, name] = score(gts[name], load(f))[:4] if os.path.exists(f) else None

best = {}  # per section and column
for sec, methods in SECTIONS:
    for name, _, _ in SETS:
        for i in range(4):
            col = [vals[sec, m, name][i] for m, *_ in methods if vals[sec, m, name]]
            best[sec, name, i] = max(col) if col else None

fmt = lambda v, b: ("\\textbf{%.2f}" % v) if b is not None and abs(v - b) < 1e-9 else "%.2f" % v
L = [r"\begin{table}[h!]", r"    \centering", r"    \footnotesize",
     r"    \caption{Methods trained on JRDB (20 train scenes, checkpoint picked on the 7 val scenes) or on \dataset{}, evaluated on the JRDB val"
     r" and test sets. mIoU, P, R and F1 are computed on group boxes (the smallest box around each group of two or more"
     r" people), matched one-to-one by IoU; a match with IoU $\geq 0.5$ is correct (MINGLE metric). DVT3: released JRDB"
     r" checkpoint; MINGLE: released model (trained on its own street-level data)."
     + (r" On val, as on test, people marked no-eval or outside the image are left out and all group difficulties are kept."
        if args.val == "v2" else "") + r" Bold = best per section.}",
     r"    \resizebox{\linewidth}{!}{", r"    \begin{tabular}{lcccccccc}", r"    \toprule",
     r"        & \multicolumn{4}{c}{\textbf{JRDB Val}} & \multicolumn{4}{c}{\textbf{JRDB Test}} \\",
     r"        \cmidrule(lr){2-5} \cmidrule(lr){6-9}",
     r"    Method & mIoU & P & R & F1 & mIoU & P & R & F1 \\", r"    \midrule"]
for k, (sec, methods) in enumerate(SECTIONS):
    if k:
        L.append(r"    \midrule")
    L.append(rf"    \multicolumn{{9}}{{l}}{{\textit{{{sec}}}}} \\")
    for m, cite, _, _ in methods:
        cells = [f"{m}~\\cite{{{cite}}}" if cite else m]
        for name, _, _ in SETS:
            v = vals[sec, m, name]
            cells += [fmt(v[i], best[sec, name, i]) for i in range(4)] if v else ["-"] * 4
        L.append("    " + " & ".join(cells) + r" \\")
L += [r"    \bottomrule", r"    \end{tabular}", r"    }", r"    \label{table:jrdb_val_test_MINGLE}", r"\end{table}"]
tex = "\n".join(L) + "\n"
open(args.out, "w").write(tex)
print(tex)
print("wrote", os.path.abspath(args.out))
