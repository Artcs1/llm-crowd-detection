"""Collect AP (task_3: AP + G1..G5+ AP) and F1 (half and exact: All, Alone=G1, Group=G2+, G2..G5+) for every method in
EgoGroups_test_10 into one text report. AP numbers come from a folder of '<name> AP G1 G2 G3 G4 G5' lines (one .ap file
per method), F1 numbers from by_size_all.csv / by_size_exact_all.csv (by_size_table.py).

    python make_report.py --ap_dir <dir of .ap files> --out ../../results/all_methods_AP_F1.txt
"""
import argparse, csv, glob, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--ap_dir", required=True)
ap.add_argument("--res", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../results"))
ap.add_argument("--out", required=True)
args = ap.parse_args()

AP = {}
for f in glob.glob(os.path.join(args.ap_dir, "*.ap")):
    r = open(f).read().split()
    AP[r[0][len("results_"):]] = [float(v) * 100 for v in r[1:7]]
DET = os.path.join(args.res, "detection_files/EgoGroups_test_10")
COV = {n: 100 * sum(1 for _ in open(os.path.join(DET, "results_" + n + ".txt"))) / 76000 for n in AP}
def load(p):
    return {r["name"][len("results_"):-len(".pkl")] if r["name"].endswith(".pkl") else r["name"].replace("results_", ""): r
            for r in csv.DictReader(open(p))}
F1 = {"half": load(os.path.join(args.res, "by_size_all.csv")), "exact": load(os.path.join(args.res, "by_size_exact_all.csv"))}
pct = float  # by_size_table.py already writes percentages

CATS = [
    ("Baselines (no learning)", lambda n: n.startswith("Naive")),
    ("Specialized methods trained on JRDB (cross-dataset)", lambda n: "JRDBTrain" in n or "JRDBpretrained" in n),
    ("Specialized methods trained on EgoGroups_train", lambda n: "EgoGroupsTrain" in n),
    ("MINGLE", lambda n: n.startswith("MINGLE")),
    ("Fine-tuned Qwen (SFT / RL)", lambda n: re.search(r"(SFT|RL\d|RL-ep|RL-Synth)", n) is not None),
    ("Zero-shot LLMs / VLMs (single = target frame only)", lambda n: not n.startswith("full_")),
    ("Zero-shot LLMs / VLMs (full = with movement / video)", lambda n: True),
]
names = sorted(set(AP) | set(F1["half"]))
groups, used = [], set()
for title, pred in CATS:
    sel = [n for n in names if n not in used and pred(n)]
    used.update(sel); groups.append((title, sel))
W = max(len(n) for n in names) + 2

def line(cells, w=7):
    return "".join(c.rjust(w) for c in cells)

out = []
P = out.append
P("EgoGroups_test (900 clips, 5 keyframes each, 76,000 people) -- all methods")
P("AP  = AP_evaluator JRDB_eval.py task_3 (mean of the 5 size-class APs; wrongly grouped people are dropped, not FPs)")
P("F1  = F1_evaluator frame-wise group F1. 'half': a predicted group matches a GT group if equal (pairs) or |inter|/max > 1/2;")
P("      'exact': identical members. All = every group incl. singletons; Alone = G1 (people alone); Group = G2+ pooled")
P("      (= the original F1 number); Mean = (Alone + Group)/2. People missing from a prediction count as predicted singletons.")
P("Notes: DVT3_EgoGroupsTrain_epN_thT = new-split DVT, stage-2 epoch N (1 = best val, 30 = last), threshold T (0.9 = paper default, 0.50 = val sweep).")
P("       PAR_EgoGroupsTrain_epN uses the original weight decay 1e-2 (relation weights decay to 0 by epoch ~16; ep11 = best val);")
P("       PAR_EgoGroupsTrain_wd1e-3_epN uses weight decay 1e-3 (ep19 = best val, ep30 = last). RTGD_EgoGroupsTrain: ep40 = best val, ep49 = last (ep20 = earlier best).")
P("       cov% = share of the 76,000 GT people present in the AP detection file (LLM outputs that failed to parse drop")
P("       people; AP counts them as misses, while F1 counts them as predicted singletons).")
P("       Qwen2.5-7B-Instruct_llm_wilddet_3D_p1 has an EMPTY prediction file, so it scores exactly like NaiveAllSingletons.")
P("       NaiveCluster3D_* and full_Cosmos-Reason2-2B_llm AP files were rebuilt from their pickles (old files had wrong boxes;")
P("       kept in detection_files/EgoGroups_test_10/BROKEN_boxes_backup/).")
P("       '-' = not available (empty or unparseable prediction file).")
P("")
for crit in ["AP", "half", "exact"]:
    if crit == "AP":
        P("=" * 100); P("1. AP (%)"); P("=" * 100)
        hdr = line(["AP", "G1", "G2", "G3", "G4", "G5+", "cov%"])
    else:
        P(""); P("=" * 100); P(f"{2 if crit == 'half' else 3}. F1 (%), {crit} criterion"); P("=" * 100)
        hdr = line(["All", "Alone", "Group", "Mean", "G2", "G3", "G4", "G5+", "Grp_P", "Grp_R"])
    for title, sel in groups:
        if not sel: continue
        P(""); P(f"-- {title}"); P("method".ljust(W) + hdr)
        for n in sel:
            if crit == "AP":
                v = AP.get(n); P(n.ljust(W) + (line(["%.2f" % x for x in v] + ["%.1f" % COV[n]]) if v else line(["-"] * 7)))
            else:
                r = F1[crit].get(n)
                if not r: P(n.ljust(W) + line(["-"] * 10)); continue
                g1, gr = pct(r["G1_F1"]), pct(r["Group_F1"])
                vals = [pct(r["All_F1"]), g1, gr, (g1 + gr) / 2] + [pct(r[b + "_F1"]) for b in ["G2", "G3", "G4", "G5+"]] + \
                       [pct(r["Group_P"]), pct(r["Group_R"])]
                P(n.ljust(W) + line(["%.2f" % x for x in vals]))
open(args.out, "w").write("\n".join(out) + "\n")
print("wrote", args.out, len(names), "methods")
