"""Dataset folders of the egogroups-city split (egogroups_city.py), in the layout of EgoGroups_train / EgoGroups_test:

  <root>/EgoGroupsCity_Train   the 1620 city train + 180 city val clips (val = egogroups_city_val.txt, as for
                               EgoGroups_train + its geo180 val list)
  <root>/EgoGroupsCity_Test    the 900 city test clips
  <root>/EVP/depth/gold_egogroupscity_{train,test}   their EVP depth maps

Every entry is a relative link into EgoGroups_test (clips 1-900) / EgoGroups_train (901-2700) and EVP/depth/
gold_egogroups_{test,train}; clip names (clip_XXXX_YYYYY, 5 sub-clips per clip) keep their original numbers, so
annotations, depth and the city GT (gt_group_10fps_city_*, seq = clip - 1) line up. Sub-folders: jsons_step1..6 +
videos_frames. EgoGroups_train has jsons_step5 for only part of its clips and no jsons_step6: its clips' jsons_step6
entries link to their jsons_step4 file (step6 = step4 + the depthanything / wilddet fields, same boxes, track ids and
detany_3D values), missing jsons_step5 files are left missing.

    python scripts/make_egogroupscity_folders.py [--root <folder holding EgoGroups_train / EgoGroups_test>] [--no_depth]

--benchmark benchmark2 builds the same for benchmark2 (egogroups_city.BENCHMARKS): <root>/EgoGroupsBenchmark2_Train
(train + val, val = laaf_id_val_clip_ids.txt), <root>/EgoGroupsBenchmark2_IDTest, <root>/EgoGroupsBenchmark2_OODTest and
EVP/depth/gold_egogroupsbenchmark2_{train,idtest,oodtest}. --benchmark benchmark3: <root>/EgoGroupsBenchmark3_Train
(train + id_val), <root>/EgoGroupsOODCity_Test (id_test) and <root>/EgoGroupsOODRegions_Test (ood_test).

<root> defaults to the folder that contains this repo (llm-crowd-detection/..). The depth links are skipped (with a
message) when <root>/EVP/depth/gold_egogroups_{test,train} do not exist, or with --no_depth; only MINGLE uses them.
Re-running is safe: existing links are kept, missing ones are added. See README_egogroups_city.md.
"""

import argparse
import os
import sys

CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CODE)
from egogroups_city import BENCHMARKS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--root', default=os.path.abspath(os.path.join(CODE, '..', '..')),
                help='folder holding EgoGroups_train and EgoGroups_test (default: the parent of llm-crowd-detection)')
ap.add_argument('--no_depth', action='store_true', help='do not link the EVP depth maps')
ap.add_argument('--benchmark', choices=sorted(BENCHMARKS), default='city')
args = ap.parse_args()
ROOT = os.path.abspath(args.root)
for name in ('EgoGroups_train', 'EgoGroups_test'):
    if not os.path.isdir(os.path.join(ROOT, name)):
        sys.exit(f'{os.path.join(ROOT, name)} not found: pass --root <folder holding EgoGroups_train and EgoGroups_test>')
DEPTH_SRC = {k: os.path.join(ROOT, 'EVP', 'depth', f'gold_egogroups_{k}') for k in ('test', 'train')}
LINK_DEPTH = not args.no_depth and all(os.path.isdir(d) for d in DEPTH_SRC.values())
if not args.no_depth and not LINK_DEPTH:
    print('EVP depth maps not found (', ', '.join(DEPTH_SRC.values()), '): skipping the depth links')
SUBS = [f'jsons_step{i}' for i in range(1, 7)] + ['videos_frames']
B = BENCHMARKS[args.benchmark]
if args.benchmark == 'city':
    FOLDERS = {'EgoGroupsCity_Train': B['train'] + B['val'], 'EgoGroupsCity_Test': B['test']}
elif args.benchmark == 'benchmark3':  # id_test = OOD_city, ood_test = OOD_regions
    FOLDERS = {'EgoGroupsBenchmark3_Train': B['train'] + B['id_val'], 'EgoGroupsOODCity_Test': B['id_test'],
               'EgoGroupsOODRegions_Test': B['ood_test']}
else:
    FOLDERS = {'EgoGroupsBenchmark2_Train': B['train'] + B['val'], 'EgoGroupsBenchmark2_IDTest': B['id_test'],
               'EgoGroupsBenchmark2_OODTest': B['ood_test']}


def link(target, name):
    if not os.path.lexists(name):
        os.symlink(os.path.relpath(target, os.path.dirname(name)), name)


for folder, clips in FOLDERS.items():
    counts = {}
    for sub in SUBS:
        os.makedirs(os.path.join(ROOT, folder, sub), exist_ok=True)
        for c in clips:
            src = os.path.join(ROOT, 'EgoGroups_test' if c <= 900 else 'EgoGroups_train')
            for k in range(5):
                name = f'clip_{c:04d}_{k:05d}' + ('' if sub == 'videos_frames' else '.json')
                target = os.path.join(src, sub, name)
                if sub == 'jsons_step6' and c > 900:
                    target = os.path.join(src, 'jsons_step4', name)
                if os.path.exists(target):
                    link(target, os.path.join(ROOT, folder, sub, name))
                elif sub != 'jsons_step5':
                    raise FileNotFoundError(target)
        counts[sub] = len(os.listdir(os.path.join(ROOT, folder, sub)))
    if not LINK_DEPTH:
        print(folder, len(clips), 'clips', counts)
        continue
    depth = os.path.join(ROOT, 'EVP', 'depth', 'gold_' + folder.lower())  # gold_egogroupscity_train, ...
    os.makedirs(depth, exist_ok=True)
    for c in clips:
        src = DEPTH_SRC['test' if c <= 900 else 'train']
        for k in range(5):
            name = f'clip_{c:04d}_{k:05d}'
            if not os.path.isdir(os.path.join(src, name)):
                raise FileNotFoundError(os.path.join(src, name))
            link(os.path.join(src, name), os.path.join(depth, name))
    counts['depth'] = len(os.listdir(depth))
    print(folder, len(clips), 'clips', counts)
