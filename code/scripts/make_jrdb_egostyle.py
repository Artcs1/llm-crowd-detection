"""JRDB shards with EgoGroups-style 3D input, for the EgoGroups-trained models. The JRDB "3D" field (robot frame: x
forward = panorama centre, y left, z up) is rewritten in the camera frame of EgoGroups' detany_3D (x right, y down,
z forward):  x = -Y,  y = 0.89 - Z (0.89 = median EgoGroups y),  z = X - min_X + 1 with min_X the smallest X in the
shard (all its frames), so every depth is positive (people behind the robot included) and the nearest person is 1 m
ahead. A proper rotation plus a shift: every distance between people is unchanged. The field keeps its name "3D"
(utils.py reads it for dataset JRDB_gold). videos_frames links to the source; the source folders are untouched.

    python scripts/make_jrdb_egostyle.py   # -> ../../JRDB_fixed_gold_egostyle, ../../JRDB_val_fixed_gold_egostyle
"""
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

ROOT = "/data/paola-shared/jmurrugarral"
H0 = 0.89
PAIRS = [("JRDB_fixed_gold", "JRDB_fixed_gold_egostyle"), ("JRDB_val_fixed_gold", "JRDB_val_fixed_gold_egostyle")]


def convert(job):
    f, dst = job
    d = json.load(open(f))
    xs = [p["3D"][0] for fr in d["frames"] for p in fr["detections"] if p.get("3D")]
    min_x = min(xs) if xs else 0.0
    for fr in d["frames"]:
        for p in fr["detections"]:
            if p.get("3D"):
                X, Y, Z = p["3D"]
                p["3D"] = [-Y, H0 - Z, X - min_x + 1.0]
    d["note"] = "3D in the EgoGroups detany_3D camera frame (x right, y down, z forward), see make_jrdb_egostyle.py"
    json.dump(d, open(os.path.join(dst, os.path.basename(f)), "w"))


if __name__ == "__main__":
    for src, dst in PAIRS:
        out = f"{ROOT}/{dst}/jsons_gold"
        os.makedirs(out, exist_ok=True)
        link = f"{ROOT}/{dst}/videos_frames"
        if not os.path.lexists(link):
            os.symlink(os.path.relpath(os.path.realpath(f"{ROOT}/{src}/videos_frames"), f"{ROOT}/{dst}"), link)
        files = sorted(glob.glob(f"{ROOT}/{src}/jsons_gold/*.json"))
        with ProcessPoolExecutor(16) as ex:
            list(ex.map(convert, [(f, out) for f in files], chunksize=20))
        print(src, len(files), "files ->", out)
