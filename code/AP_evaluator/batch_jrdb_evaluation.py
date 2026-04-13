import subprocess
import csv
import glob
import argparse
import pandas as pd

#groundtruth = "out/gt_group.txt" # CHANGE
#groundtruth = 'gt_sekai_ours_2.txt'
groundtruth = 'out/gt_sekai_ours_22.txt'
#groundtruth = 'gt_sekai_ours_42.txt'
task = "task_3"
labelmap = "./label_map/task_3.pbtxt"

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
            "python3", "JRDB_eval.py",
            "-g", groundtruth,
            "-d", det, #os.path.expanduser(f"~/Desktop/llm-crowd-detection/code/detection_files/JRDB/{det}"),
            "-t", task,
            "--labelmap", labelmap
        ],
        capture_output=True,
        text=True
    )

    # Parse stdout line by line
    #metrics = {"file": det}
    #for line in result.stdout.splitlines():
    #    if ":" in line:
    #        key, value = line.split(":", 1)
    #        metrics[key.strip()] = value.strip()

    performance = result.stdout.split('\n')
    print(performance)

    data = [det.split('/')[-1]]
    data.append(round(float(performance[1])*100,2))
    data.append(round(float(performance[2])*100,2))
    data.append(round(float(performance[3])*100,2))
    data.append(round(float(performance[4])*100,2))
    data.append(round(float(performance[5])*100,2))
    data.append(round(float(performance[0])*100,2))

    datas.append(data)

columns = ['name', 'G1', 'G2', 'G3', 'G4', 'G5', 'AP']
output_csv = args.dataset_path.split('/')[-1] + '.txt'
df = pd.DataFrame(datas, columns=columns)
df.to_csv(output_csv, index=False)
    

print("✅ All evaluations done — results saved to "+output_csv)
