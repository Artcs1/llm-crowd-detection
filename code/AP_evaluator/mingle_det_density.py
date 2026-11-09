"""MINGLE P / R / F1 (mingle_det.py) of benchmark3 test detections, overall and per density level (scattered / moderate /
crowded, `density` of annotations jsons_step1/<sub-clip>.json): the GT is restricted to the keyframes of that level (seq =
clip - 1, frame = (sub-clip + 1) * 10) and scored as usual, so the overall numbers equal mingle_det.py's.
Scores every .txt of results/detection_files/<set>_10 (or the given files) on both sets; TSV rows:
set, file, density, mIoU, P, R, F1, n_pred, n_gt.

    python mingle_det_density.py [--out density_scores.tsv] [det file names ...]
"""
import argparse
import glob
import json
import os

from mingle_det import load, score

SETS = {'EgoGroupsOODCity_Test': 'benchmark3_id_test', 'EgoGroupsOODRegions_Test': 'benchmark3_ood_test'}
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..')
DET = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'results', 'detection_files')
LEVELS = ['scattered', 'moderate', 'crowded']


def densities(dataset):
    out = {}
    for p in glob.glob(f'{ROOT}/{dataset}/jsons_step1/clip_*.json'):
        num, sub = os.path.basename(p)[:-5].split('_')[1:]
        out[int(num) - 1, (int(sub) + 1) * 10] = json.load(open(p))['density']
    return out


ap = argparse.ArgumentParser()
ap.add_argument('files', nargs='*')
ap.add_argument('--out', default='density_scores.tsv')
args = ap.parse_args()
with open(args.out, 'w') as fh:
    fh.write('set\tfile\tdensity\tmIoU\tP\tR\tF1\tn_pred\tn_gt\n')
    for dataset, gt_name in SETS.items():
        gt = load(f'out/gt_group_10fps_{gt_name}.txt')
        dens = densities(dataset)
        missing = [k for k in gt if k not in dens]
        assert not missing, f'{dataset}: {len(missing)} GT keyframes without density, e.g. {missing[:3]}'
        subsets = {'all': gt, **{d: {k: v for k, v in gt.items() if dens[k] == d} for d in LEVELS}}
        files = args.files or sorted(os.path.basename(p) for p in glob.glob(f'{DET}/{dataset}_10/*.txt'))
        for f in files:
            path = f'{DET}/{dataset}_10/{f}'
            if not os.path.exists(path):
                continue
            det = load(path)
            for d, g in subsets.items():
                m, p, r, f1, npd, ngt = score(g, det)
                fh.write(f'{dataset}\t{f}\t{d}\t{m:.2f}\t{p:.2f}\t{r:.2f}\t{f1:.2f}\t{npd}\t{ngt}\n')
        print(dataset, {d: len(g) for d, g in subsets.items()}, 'keyframes', flush=True)
