from collections import defaultdict
from scipy.optimize import linear_sum_assignment
import numpy as np
import pickle
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Hungarian matching and save results to PKL")
parser.add_argument("--pr_file", type=str, required=True,help="Prediction file path")
parser.add_argument("--gt_file", type=str, required=True,help="Ground truth file path")
args = parser.parse_args()

pr_file = args.pr_file
gt_file = args.gt_file

def build_id_to_dets(lines):
    id_to_dets = defaultdict(list)
    map_id = defaultdict(int)

    for line in lines:
        line = line.rstrip("\n")
        values = line.split(" ")
        values = [int(val) for val in values]

        key = values[0]
        group = values[6]

        map_id[key] += 1
        box_id = map_id[key]

        item = [
            values[2],  # x
            values[3],  # y
            values[4],  # w
            values[5],  # h
            group,
            box_id
        ]

        id_to_dets[key].append(item)

    return id_to_dets

def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union = areaA + areaB - inter

    return inter / union if union > 0 else 0
    
with open(pr_file,"r") as f:
    lines = f.readlines()
    
prediction = build_id_to_dets(lines)

with open(gt_file, "r") as f:
    lines = f.readlines()
    
ground_truth = build_id_to_dets(lines)

keys = list(ground_truth.keys())

results  = {}


for key in keys:

    if key not in results:
        results[key] = {}
    
    pred = prediction[key]
    gt = ground_truth[key]

    pred_bboxes = [det[:4] for det in pred]
    gt_bboxes = [det[:4] for det in gt]

    # Cost matrix
    cost_matrix = np.array([
        [1 - iou(a, b) for b in pred_bboxes]
        for a in gt_bboxes
    ])
    
    # Hungarian matching
    rows, cols = linear_sum_assignment(cost_matrix)

    detected_groups = defaultdict(list)
    for r, c in zip(rows, cols):
        if 1 - cost_matrix[r, c] >= 0.5:
            detected_groups[pred[c][4]].append(r)

    
    #print(detected_groups)

    results[key]['0'] = list(detected_groups.values()) 
    #print(results[key]['0'])
    #print(pred_bboxes)
    #print(gt_bboxes)
    
    #print(pred)
    #print(gt)
    #input()

print(results)

pkl_file = Path(pr_file).with_suffix(".pkl")
with open(pkl_file, "wb") as f:
    pickle.dump(results, f)
print(f"Saved results to: {pkl_file}")

#print(prediction)
