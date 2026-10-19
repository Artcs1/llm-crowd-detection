"""EgoGroups_test with JRDB-style 3D input: every detany_3D position (camera frame: x right, y down, z forward) is
rewritten in the JRDB robot frame (x forward, y left, z up):  X = z,  Y = -x,  Z = -y + 0.89  (0.89 = median EgoGroups
y, so the median height is ~0 as in JRDB). A proper rotation plus a shift: every distance between people is unchanged.
Writes ../../EgoGroups_test_jrdbstyle/jsons_step6 (videos_frames links to EgoGroups_test); EgoGroups_test is untouched.

    python scripts/make_egogroups_jrdbstyle.py
"""
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

ROOT = "/data/paola-shared/jmurrugarral"
SRC, DST = f"{ROOT}/EgoGroups_test/jsons_step6", f"{ROOT}/EgoGroups_test_jrdbstyle/jsons_step6"
H0 = 0.89


def convert(f):
    d = json.load(open(f))
    for fr in d["frames"]:
        for p in fr["detections"]:
            x, y, z = p["detany_3D"]
            p["detany_3D"] = [z, -x, -y + H0]
    d["note"] = "detany_3D in the JRDB frame (x forward, y left, z up), see make_egogroups_jrdbstyle.py"
    json.dump(d, open(os.path.join(DST, os.path.basename(f)), "w"))


if __name__ == "__main__":
    os.makedirs(DST, exist_ok=True)
    link = f"{ROOT}/EgoGroups_test_jrdbstyle/videos_frames"
    if not os.path.exists(link):
        os.symlink("../EgoGroups_test/videos_frames", link)
    files = sorted(glob.glob(f"{SRC}/*.json"))
    with ProcessPoolExecutor(16) as ex:
        list(ex.map(convert, files, chunksize=20))
    print(len(files), "files ->", DST)
