"""egogroups-city: train / val / test division of all 2,700 EgoGroups clips (clip numbers 1-2700) with no city and no
source video shared between splits (user-provided lists, egogroups_city_{train,val,test}.txt; 1620 / 180 / 900 clips).

The split mixes the two EgoGroups folders: clips 1-900 live in EgoGroups_test (GT rows of out/gt_group_10fps_test.*,
seq = clip - 1) and clips 901-2700 in EgoGroups_train (gt_group_10fps_train.*, seq = clip - 901). The city files use one
numbering for all clips, seq = clip - 1:

    python egogroups_city.py gt                     # AP_evaluator/out + F1_evaluator/out gt_group_10fps_city_{train,val,test}.{txt,pkl}
    python egogroups_city.py gt --benchmark benchmark2   # gt_group_10fps_benchmark2_{train,val,id_test,ood_test}.{txt,pkl}
    python egogroups_city.py gt --benchmark benchmark3   # gt_group_10fps_benchmark3_{train,id_val,id_test,ood_test}.{txt,pkl}
    python egogroups_city.py merge --split test --test_part A.txt --train_part B.txt --out C.txt
        # a method's detection file for EgoGroups_test (seq = clip - 1) + one for EgoGroups_train clips (seq = clip - 901)
        # -> the rows of the city split's clips, seq = clip - 1 (same for .pkl files: {seq: {'0'..'4': groups}})

benchmark2 (BENCHMARKS['benchmark2']): a second user-provided division (laaf*_clip_ids.txt, 1359 / 194 / 388 / 759 clips),
no city or source video shared; ood_test = the only clips of African, Eastern Europe and Latin America. Same numbering.
benchmark3: a third one (benchmark3_*_clip_ids.txt, 1533 / 219 / 438 / 510), no city or source video shared; id_test =
"OOD_city" (unseen cities of seen regions), ood_test = "OOD_regions" (the only African and Latin America clips).
"""

import argparse
import os
import pickle

CODE = os.path.dirname(os.path.abspath(__file__))
SPLITS = ("train", "val", "test")
CITY = {s: [int(l) for l in open(os.path.join(CODE, f"egogroups_city_{s}.txt")) if l.strip()] for s in SPLITS}
SUFFIX = "_city"  # run folders / output names of egogroups-city runs
BENCHMARKS = {
    "city": CITY,
    "benchmark2": {s: [int(l) for l in open(os.path.join(CODE, f)) if l.strip()] for s, f in (
        ("train", "laaf4_train_clip_ids.txt"), ("val", "laaf_id_val_clip_ids.txt"),
        ("id_test", "laaf_id_test_clip_ids.txt"), ("ood_test", "laaf_ood_test_clip_ids.txt"))},
    "benchmark3": {s: [int(l) for l in open(os.path.join(CODE, f"benchmark3_{s}_clip_ids.txt")) if l.strip()]
                   for s in ("train", "id_val", "id_test", "ood_test")},
}


def folder(clip):
    """('test', seq in gt_group_10fps_test) or ('train', seq in gt_group_10fps_train) of a clip number."""
    return ("test", clip - 1) if clip <= 900 else ("train", clip - 901)


def by_folder(split):
    """{'test': [clip numbers in EgoGroups_test], 'train': [clip numbers in EgoGroups_train]} of a city split."""
    out = {"test": [], "train": []}
    for c in CITY[split]:
        out[folder(c)[0]].append(c)
    return out


def gt_path(split, ext="txt", benchmark="city"):
    sub = "AP_evaluator" if ext == "txt" else "F1_evaluator"
    return os.path.join(CODE, sub, "out", f"gt_group_10fps_{benchmark}_{split}.{ext}")


def _merge_txt(parts, clips, out):
    """parts = {'test': file with seq = clip - 1, 'train': file with seq = clip - 901}; rows of `clips`, seq = clip - 1."""
    clips = set(clips)
    rows = []
    for name, path in parts.items():
        offset = 1 if name == "test" else 901
        for line in open(path):
            r = line.split()
            if r and int(r[0]) + offset in clips:
                rows.append((int(r[0]) + offset - 1, int(r[1]), " ".join([str(int(r[0]) + offset - 1)] + r[1:])))
    rows.sort(key=lambda x: (x[0], x[1]))
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        f.writelines(r[2] + "\n" for r in rows)
    return len({r[0] for r in rows})


def _merge_pkl(parts, clips, out):
    clips = set(clips)
    res = {}
    for name, path in parts.items():
        offset = 1 if name == "test" else 901
        for seq, v in pickle.load(open(path, "rb")).items():
            if int(seq) + offset in clips:
                res[int(seq) + offset - 1] = v
    with open(out, "wb") as f:
        pickle.dump(dict(sorted(res.items())), f)
    return len(res)


def merge(split, test_part, train_part, out):
    parts = {k: v for k, v in (("test", test_part), ("train", train_part)) if v}
    fn = _merge_pkl if out.endswith(".pkl") else _merge_txt
    n = fn(parts, CITY[split], out)
    need = {folder(c)[0] for c in CITY[split]}
    assert need <= set(parts), f"{split} has clips in EgoGroups_{sorted(need - set(parts))}, which no part covers"
    return n


def build_gt(benchmark="city"):
    for s, clips in BENCHMARKS[benchmark].items():
        for ext, sub in (("txt", "AP_evaluator"), ("pkl", "F1_evaluator")):
            parts = {k: os.path.join(CODE, sub, "out", f"gt_group_10fps_{k}.{ext}") for k in ("test", "train")}
            n = (_merge_pkl if ext == "pkl" else _merge_txt)(parts, clips, gt_path(s, ext, benchmark))
            # clips with nobody at a keyframe have no rows in the txt files (15 of EgoGroups_test's 1-900)
            print(gt_path(s, ext, benchmark), n, "clips of", len(clips))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gt", "merge"])
    ap.add_argument("--split", choices=SPLITS, default="test")
    ap.add_argument("--test_part", help="detection file of the method on EgoGroups_test (seq = clip - 1)")
    ap.add_argument("--train_part", help="detection file of the method on EgoGroups_train clips (seq = clip - 901)")
    ap.add_argument("--out")
    ap.add_argument("--benchmark", choices=sorted(BENCHMARKS), default="city", help="gt: which division")
    a = ap.parse_args()
    if a.cmd == "gt":
        build_gt(a.benchmark)
    else:
        print(a.out, merge(a.split, a.test_part, a.train_part, a.out), "clips")
