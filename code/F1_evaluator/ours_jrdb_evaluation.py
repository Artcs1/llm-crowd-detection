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


import json

country_mapping = {
    "USA": "United States",
    "UAE": "United Arab Emirates",
    "Korea": "South Korea",
    "Czechia": "Czech Republic",
}

region_modes = {
    "AF": "African",
    "AN": "Anglo",
    "CA": "Confucian Asia",
    "EU": "Eastern Europe",
    "GE": "Germanic Europe",
    "LA": "Latin America",
    "LE": "Latin Europe",
    "ME": "Middle East",
    "NE": "Nordic Europe",
    "SA": "South-East Asia",
    "O": "Other",
}

source_dataset = "/lp-dev/jmurrugarral/SEKAI_900_3/"

seq_keys = list(gt.keys())

# -------------------------------------------------
# Select indices
# -------------------------------------------------
if args.o == "scattered":
    selected_idx = range(0, len(seq_keys), 3)

elif args.o == "moderate":
    selected_idx = range(1, len(seq_keys), 3)

elif args.o == "crowded":
    selected_idx = range(2, len(seq_keys), 3)

# -------------------------------------------------
# Region-based modes
# -------------------------------------------------
elif args.o in region_modes:

    target_region = region_modes[args.o]

    selected_idx = []

    for i, key in enumerate(seq_keys):

        json_file = (
            f"{source_dataset}/jsons_step1/{key}.json"
        )

        with open(json_file, "r") as f:
            data = json.load(f)

        country = country_mapping.get(
            data["country"],
            data["country"]
        )

        globe = country_to_region_globe[country]

        if globe == target_region:
            selected_idx.append(i)

# -------------------------------------------------
# Default: all
# -------------------------------------------------
else:
    selected_idx = range(len(seq_keys))

# -------------------------------------------------
# Filter pred and gt
# -------------------------------------------------
pred = {
    seq_keys[i]: pred[seq_keys[i]]
    if seq_keys[i] in pred else []
    for i in selected_idx
}

gt = {
    seq_keys[i]: gt[seq_keys[i]]
    for i in selected_idx
}

Evaluater = Evaluation(
    whole=False,
    alone=False,
    pre_dict=pred,
    GT_dict=gt
)

pre_d, rec_d, f1_d = Evaluater()

print(pre_d)
print(rec_d)
print(f1_d)
