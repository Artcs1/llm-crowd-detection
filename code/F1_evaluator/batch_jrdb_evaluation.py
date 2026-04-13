import subprocess
import csv
import glob
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Batch evaluator JRDB results")
parser.add_argument('dataset_path')
args = parser.parse_args()

detection_files = glob.glob(args.dataset_path + '/*')
detection_files.sort()
datas = []

for det in detection_files:
    print(f"Evaluating: {det}")

    result = subprocess.run(
        [
            "python3", "ours_jrdb_evaluation.py",
            "--det", det,
            "--gt", "gt_sekai_2.pkl"
        ],
        capture_output=True,
        text=True
    )

    print(result)
    performance = result.stdout.split('\n')
    print(performance)

    data = [det.split('/')[-1]]
    print(data)

    if len(performance) < 3:
        print(f"WARNING: Unexpected output for {det}, skipping.")
        continue

    try:
        data.append(round(float(performance[-4]) * 100, 2))  # Precision
        data.append(round(float(performance[-3]) * 100, 2))  # Recall
        data.append(round(float(performance[-2]) * 100, 2))  # AP
    except ValueError as e:
        print(f"WARNING: Could not parse output for {det}: {e}")
        print(f"Output was: {performance}")
        continue

    datas.append(data)
    #print(datas)

columns = ['name', 'Precision', 'Recall', 'AP']
output_csv = args.dataset_path.split('/')[-1] + '.txt'
print(output_csv)
df = pd.DataFrame(datas, columns=columns)
df.to_csv(output_csv, index=False)
    

print("✅ All evaluations done — results saved to "+output_csv)

