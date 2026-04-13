import argparse
import pickle
from utils_eval import Load_GT, Evaluation, Predict_graph

parser = argparse.ArgumentParser(description="Evaluate prediction pickle file")
parser.add_argument("--det", type=str, required=True, help="Path to the detected pickle file")
parser.add_argument("--gt", type=str, required=True, help="Path to the GT pickle file")
args = parser.parse_args()

with open(args.det, "rb") as f:
    pred = pickle.load(f)

with open(args.gt, "rb") as f:
    gt = pickle.load(f)


Evaluater = Evaluation(whole=False, alone=False, pre_dict=pred, GT_dict=gt)
pre_d, rec_d, f1_d = Evaluater()

print(pre_d)
print(rec_d)
print(f1_d)
