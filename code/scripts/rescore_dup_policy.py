"""Re-score video reward-ablation predictions with a chosen rule for a person listed in several groups.

compute_detections.py (finegrained) keeps a person's FIRST group; the GRPO rewards' groups_to_labels
(rl_training_vlm_nodspy_merged.py) keep the LAST. This rebuilds the same detection rows as compute_detections.py
(ids not in id_tobbox dropped, people left out -> singletons, group ids in order) under --policy first|last, scores them
with AP_evaluator/mingle_det.py, and reports how many keyframes have a duplicated / missing / made-up id.
--policy first must reproduce the official scores (sanity check).

  python scripts/rescore_dup_policy.py --policy last [--models Qwen3-VL-4B-Instruct-RL-B3-Social-RW-f1-Full ...]
Detection files: ../results/detection_files/<split>_10/dup_<policy>/<same name as compute_detections.py>
"""
import argparse
import glob
import json
import os
import subprocess
from collections import Counter
from multiprocessing import Pool

SPLITS = {'EgoGroupsOODCity_Test': 'benchmark3_id_test', 'EgoGroupsOODRegions_Test': 'benchmark3_ood_test'}
RUNS = ['v', 'v-bce', 'v-bce-f1', 'v-bce-ari', 'v-f1-ari', 'v-f1', 'v-ari', 'f1', 'v-bce-f1-ari']
ap = argparse.ArgumentParser()
ap.add_argument('--policy', choices=['first', 'last'], required=True)
ap.add_argument('--models', nargs='*', default=[f'Qwen3-VL-4B-Instruct-RL-B3-Social-RW-{r}-Full' for r in RUNS])
ap.add_argument('--procs', type=int, default=18)
args = ap.parse_args()


def rows_for(data, idx, int_img, policy):
    groups, id2box = data['groups'], data['id_tobbox']
    if policy == 'last':  # each person's group = the LAST group listing them (groups_to_labels)
        owner = {str(p): gi for gi, g in enumerate(groups) for p in g}
        groups = [[p for p in g if owner[str(p)] == gi] for gi, g in enumerate(groups)]
    out, seen, gid = [], set(), 1
    for g in groups:  # from here exactly compute_detections.py finegrained (first occurrence wins)
        flag = False
        for p in g:
            if str(p) in id2box and str(p) not in seen:
                flag = True
                seen.add(str(p))
                out.append((idx, int_img, *map(int, id2box[str(p)]), gid))
        gid += flag
    for k, v in id2box.items():
        if k not in seen:
            out.append((idx, int_img, *map(int, v), gid))
            gid += 1
    return out


def job(task):
    model, split = task
    pred = f'../results/predictions/{split}/results_full/{model}/vlm_image/detany_3D/p1_social/10/*'
    det = f'../results/detection_files/{split}_10/dup_{args.policy}/results_full_{model}_vlm_image_detany_3D_p1_social.txt'
    os.makedirs(os.path.dirname(det), exist_ok=True)
    st = Counter()
    lines = []
    for d in sorted(glob.glob(pred)):
        name = os.path.basename(d)
        try:
            data = json.load(open(f'{d}/{name}.json'))
        except (OSError, json.JSONDecodeError):
            continue
        s = name.split('_')
        idx, int_img = int(s[1]) - 1, (int(s[-1]) + 1) * 10
        ids = set(data['id_tobbox'])
        c = Counter(str(p) for g in data['groups'] for p in g)
        st['frames'] += 1
        st['dup'] += any(n > 1 for n in c.values())
        st['dup_people'] += sum(1 for n in c.values() if n > 1)
        st['missing'] += bool(ids - c.keys())
        st['madeup'] += bool(c.keys() - ids)
        lines += [' '.join(map(str, (*r, 1, 1))) for r in rows_for(data, idx, int_img, args.policy)]
    with open(det, 'w') as f:
        f.write('\r\n'.join(lines) + '\r\n')
    res = subprocess.run(['python3', 'mingle_det.py', '-g', f'out/gt_group_10fps_{SPLITS[split]}.txt', '-d', '../' + det],
                         cwd='AP_evaluator', capture_output=True, text=True).stdout
    last = [l for l in res.splitlines() if 'warn' not in l.lower()][-1].split('\t')
    return model, split, last[1:5], st


if __name__ == '__main__':
    tasks = [(m, s) for m in args.models for s in SPLITS]
    with Pool(min(args.procs, len(tasks))) as pool:
        out = pool.map(job, tasks)
    print('model\tsplit\tmIoU\tP\tR\tF1\tframes\tdup_frames\tdup_people\tmissing_frames\tmadeup_frames')
    for model, split, sc, st in out:
        print('\t'.join([model, split, *sc, *(str(st[k]) for k in ('frames', 'dup', 'dup_people', 'missing', 'madeup'))]))
