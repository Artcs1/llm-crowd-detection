import subprocess
import glob
import argparse
import pandas as pd
import os
from datetime import datetime

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

all_runs = {}

for groundtruth, dataset_path in zip(args.groundtruth, args.dataset_path):
    detection_files = glob.glob(dataset_path + '/*')
    detection_files.sort()

    for det in detection_files:
        print(f"Evaluating: {det}  [gt: {groundtruth}]")
        result = subprocess.run(
            [
                "python3", "ours_jrdb_evaluation.py",
                "--det", det,
                "--gt", groundtruth,
                "--o", args.output 
            ],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"⚠️  Failed on {det}:\n{result.stderr}")
            continue

        performance = result.stdout.split('\n')

        if len(performance) < 3:
            print(f"⚠️  Unexpected output for {det}, skipping.")
            continue

        try:
            metrics = [
                round(float(performance[-4]) * 100, 2),  # Precision
                round(float(performance[-3]) * 100, 2),  # Recall
                round(float(performance[-2]) * 100, 2),  # AP
            ]
        except ValueError as e:
            print(f"⚠️  Could not parse metrics for {det}: {e}")
            print(f"Output was: {performance}")
            continue

        name = det.split('/')[-1]
        all_runs.setdefault(name, []).append(metrics)

columns = ['name', 'Precision', 'Recall', 'AP']
averaged = []
for name, runs in all_runs.items():
    df_runs = pd.DataFrame(runs, columns=columns[1:])
    avg = df_runs.mean().round(2).tolist()
    averaged.append([name] + avg)

os.makedirs("results",exist_ok=True)
df = pd.DataFrame(averaged, columns=columns)
df.to_csv(f'results/{output_csv}', index=False)

print(f"✅ All evaluations done — results saved to {output_csv}")
