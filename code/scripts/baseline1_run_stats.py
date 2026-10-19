"""Per-run statistics of the zero-shot VLM baseline1 runs on EgoGroups_test (read by make_paper_ap_tables.py, table 7):
saved keyframes (a failed call saves nothing), clips with all 5 keyframes, empty answers, boxes per answer.

    python scripts/baseline1_run_stats.py   (from code/)
"""
import json
import os
from collections import Counter

ROOT = "../results/predictions/EgoGroups_test"
OUT = "../results/baseline1_run_stats_EgoGroups_test.tsv"
rows = []
for pre in ["results", "results_full"]:
    for m in sorted(os.listdir(f"{ROOT}/{pre}")):
        d = f"{ROOT}/{pre}/{m}/vlm_image/detany_3D/baseline1/10"
        if not os.path.isdir(d):
            continue
        names = sorted(n for n in os.listdir(d) if n.startswith("clip_"))
        clips = Counter(n[:9] for n in names)
        nbox, empty = [], 0
        for n in names:
            g = json.load(open(f"{d}/{n}/{n}.json")).get("groups") or []
            k = sum(len(x) for x in g)
            empty += k == 0
            if k:
                nbox.append(k)
        nbox.sort()
        rows.append([f"{pre}_{m}", len(names), names[-1] if names else "-", sum(v == 5 for v in clips.values()), empty,
                     nbox[len(nbox) // 2] if nbox else 0, nbox[-1] if nbox else 0, sum(k > 50 for k in nbox)])
        print(*rows[-1])
with open(OUT, "w") as f:
    f.write("run\tframes\tlast\tfull_clips\tempty\tmedian_boxes\tmax_boxes\tover50\n")
    f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
print("wrote", OUT)
