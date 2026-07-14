import subprocess
import glob
import argparse
import pandas as pd
import shlex
import math
import os
from datetime import datetime

task = "task_3"
labelmap = "./label_map/task_3.pbtxt"

REGION_CODES = {"AF", "AN", "CA", "EU", "GE", "LA", "LE", "ME", "NE", "SA", "O"}
DENSITIES = ("scattered", "moderate", "crowded")

parser = argparse.ArgumentParser(description="Batch evaluator JRDB results")
parser.add_argument('--groundtruth', nargs='+', required=True,
                    help="List of ground truth files (one per dataset path)")
parser.add_argument('--dataset_path', nargs='+', required=True,
                    help="List of dataset paths (one per ground truth file)")
parser.add_argument('--output', default=None,
                    help="Output CSV name (without extension). Defaults to a timestamp.")
args = parser.parse_args()

if len(args.groundtruth) != len(args.dataset_path):
    raise ValueError(f"Mismatch: {len(args.groundtruth)} groundtruth(s) vs {len(args.dataset_path)} dataset_path(s). They must be equal.")

output_csv = (args.output + '.csv') if args.output else ('evaluation_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv')


def run_jrdb_eval(groundtruth, det, mode):
    """Runs JRDB_eval.py once. Returns the parsed [G1, G2, G3, G4, G5, AP] list (raw 0-1 scale,
    unrounded) on success, or None on failure/unparseable output."""
    cmd = [
        "python3", "JRDB_eval.py",
        "-g", groundtruth,
        "-d", det,
        "-t", task,
        "--labelmap", labelmap,
        "--output", mode,
    ]
    print("Running:", shlex.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"⚠️  Failed on {det} [{mode}]:\n{result.stderr}")
        return None

    lines = [l for l in result.stdout.strip().split('\n') if l.strip()]
    if len(lines) < 6:
        print(f"⚠️  Unexpected output for {det} [{mode}], skipping.")
        return None

    try:
        return [float(lines[1]), float(lines[2]), float(lines[3]), float(lines[4]), float(lines[5]), float(lines[0])]
    except ValueError as e:
        print(f"⚠️  Could not parse metrics for {det} [{mode}]: {e}")
        return None


def zero_fill(values):
    """Replaces a missing (whole sub-call failed) result, or any individual NaN entry within an
    otherwise-valid result, with 0.0 -- a thin region/density bucket (e.g. too few scenes to
    compute a stable AP) is treated as zero rather than dropped or left unusable."""
    if values is None:
        return [0.0] * 6
    return [0.0 if (v is None or math.isnan(v)) else v for v in values]


def combine_density_buckets(groundtruth, det, region_code):
    """Runs the 3 density-restricted sub-evaluations for one region (region ∩ scattered/moderate/
    crowded) and averages them with equal weight, so the combined region AP isn't skewed by
    however that region's scenes happen to split across density levels."""
    bucket_results = [
        zero_fill(run_jrdb_eval(groundtruth, det, f"{region_code}_{density}"))
        for density in DENSITIES
    ]
    return [sum(vals) / 3 for vals in zip(*bucket_results)]


all_runs = {}

for groundtruth, dataset_path in zip(args.groundtruth, args.dataset_path):
    detection_files = glob.glob(dataset_path + '/*')
    detection_files.sort()

    for det in detection_files:

        print(f"Evaluating: {det}  [gt: {groundtruth}]")

        if args.output in REGION_CODES:
            raw_metrics = combine_density_buckets(groundtruth, det, args.output)
        else:
            raw_metrics = run_jrdb_eval(groundtruth, det, args.output)
            if raw_metrics is None:
                continue

        metrics = [round(v * 100, 2) for v in raw_metrics]

        name = det.split('/')[-1]
        all_runs.setdefault(name, []).append(metrics)

columns = ['name', 'G1', 'G2', 'G3', 'G4', 'G5', 'AP']
averaged = []
for name, runs in all_runs.items():
    df_runs = pd.DataFrame(runs, columns=columns[1:])
    avg = df_runs.mean().round(2).tolist()
    averaged.append([name] + avg)

os.makedirs("results", exist_ok=True)

df = pd.DataFrame(averaged, columns=columns)
df.to_csv(f'results/{output_csv}', index=False)

print(f"✅ All evaluations done — results saved to {output_csv}")
