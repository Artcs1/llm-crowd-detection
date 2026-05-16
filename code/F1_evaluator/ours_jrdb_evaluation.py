import argparse
import pickle
from utils_eval import Load_GT, Evaluation, Predict_graph

parser = argparse.ArgumentParser(description="Evaluate prediction pickle file")
parser.add_argument("--det", type=str, required=True, help="Path to the detected pickle file")
parser.add_argument("--gt", type=str, required=True, help="Path to the GT pickle file")
parser.add_argument("--o", type=str, required=True, help="Partition result")
args = parser.parse_args()

with open(args.det, "rb") as f:
    pred = pickle.load(f)

with open(args.gt, "rb") as f:
    gt = pickle.load(f)

seq_len = [item for item in gt.keys()]

if args.o == 'scattered':
    pred = {seq_len[i]: pred[seq_len[i]] if seq_len[i] in pred else [] for i in range(0, len(seq_len), 3)}
    gt = {seq_len[i]:gt[seq_len[i]] for i in range(0,len(seq_len),3) }
elif args.o == 'moderate':
    pred = {seq_len[i]: pred[seq_len[i]] if seq_len[i] in pred else [] for i in range(1, len(seq_len), 3)}
    gt = {seq_len[i]:gt[seq_len[i]] for i in range(1,len(seq_len),3) }
elif args.o == 'crowded':
    pred = {seq_len[i]: pred[seq_len[i]] if seq_len[i] in pred else [] for i in range(2, len(seq_len), 3)}
    gt = {seq_len[i]:gt[seq_len[i]] for i in range(2,len(seq_len),3) }

Evaluater = Evaluation(whole=False, alone=False, pre_dict=pred, GT_dict=gt)
pre_d, rec_d, f1_d = Evaluater()

print(pre_d)
print(rec_d)
print(f1_d)
