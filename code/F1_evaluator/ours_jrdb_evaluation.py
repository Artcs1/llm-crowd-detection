import argparse
import glob
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

# Copied from AP_evaluator/JRDB_eval.py's country_to_region_globe -- was referenced here before
# but never defined/imported anywhere in this file (a live NameError the moment a region mode
# was ever exercised).
country_to_region_globe = {
    "United Kingdom": "Anglo", "Belgium": "Germanic Europe", "Ireland": "Anglo",
    "France": "Latin Europe", "Germany": "Germanic Europe", "Portugal": "Latin Europe",
    "Spain": "Latin Europe", "Italy": "Latin Europe", "Switzerland": "Latin Europe",
    "Netherlands": "Germanic Europe", "Austria": "Germanic Europe", "Malta": "Latin Europe",
    "Denmark": "Nordic Europe", "Norway": "Nordic Europe", "Sweden": "Nordic Europe",
    "Finland": "Nordic Europe", "Iceland": "Nordic Europe", "Greece": "Eastern Europe",
    "Poland": "Eastern Europe", "Czech Republic": "Eastern Europe", "Slovakia": "Eastern Europe",
    "Hungary": "Eastern Europe", "Romania": "Latin Europe", "Belarus": "Eastern Europe",
    "Serbia": "Eastern Europe", "Moldova": "Latin Europe", "Estonia": "Nordic Europe",
    "Latvia": "Nordic Europe", "United States": "Anglo", "Canada": "Anglo", "Bermuda": "Other",
    "Mexico": "Latin America", "Brazil": "Latin America", "Chile": "Latin America",
    "Peru": "Latin America", "Uruguay": "Latin America", "Paraguay": "Latin America",
    "Colombia": "Latin America", "Costa Rica": "Latin America", "Venezuela": "Latin America",
    "Jamaica": "African", "Barbados": "African", "Antigua and Barbuda": "African",
    "Saint Lucia": "Other", "Kenya": "African", "Malawi": "African", "South African": "African",
    "Nigeria": "African", "Ghana": "African", "Senegal": "African", "Benin": "African",
    "Madagascar": "African", "Ethiopia": "African", "Eritrea": "African", "Mali": "African",
    "Gambia": "African", "United Arab Emirates": "Middle East", "Jordan": "Middle East",
    "Syria": "Middle East", "Iraq": "Middle East", "Lebanon": "Middle East", "Egypt": "Middle East",
    "Morocco": "Middle East", "Kuwait": "Middle East", "Israel": "Latin Europe",
    "Turkey": "Middle East", "Afghanistan": "South-East Asia", "India": "South-East Asia",
    "Pakistan": "South-East Asia", "Bangladesh": "South-East Asia", "Sri Lanka": "South-East Asia",
    "Nepal": "South-East Asia", "China": "Confucian Asia", "Japan": "Confucian Asia",
    "South Korea": "Confucian Asia", "Taiwan": "Confucian Asia", "Thailand": "South-East Asia",
    "Vietnam": "Confucian Asia", "Indonesia": "South-East Asia", "Singapore": "Confucian Asia",
    "Brunei": "South-East Asia", "Kazakhstan": "Eastern Europe", "Kyrgyzstan": "Eastern Europe",
    "Tajikistan": "South-East Asia", "Azerbaijan": "Middle East", "Georgia": "Eastern Europe",
    "Australia": "Anglo", "New Zealand": "Anglo", "Samoa": "South-East Asia", "Holy See": "Other",
    "South Africa": "African",
}

# EgoGroups_test (clip_<base>_<sub>.json naming) -- NOT gold_SEKAI_900_3 (clip_<n>.json, no
# sub-clip suffix). This was previously pointed at gold_SEKAI_900_3, which doesn't have files
# matching our GT pickle's clip_XXXX_YYYYY keys at all (confirmed: that lookup would 404).
source_dataset = "/lp-dev/jmurrugarral/EgoGroups_test/"

seq_keys = list(gt.keys())

DENSITY_MODES = {"scattered", "moderate", "crowded"}


def _load_clip_json(scenario_idx):
    # keys in gt/pred are int scenario_idx (0-indexed); EgoGroups_test clip files are
    # clip_<base 1-indexed>_<sub-clip>.json -- any sub-clip works, density/region are identical
    # across all 5 (verified). Same convention as AP_evaluator/JRDB_eval.py's _load_clip_json.
    json_file = glob.glob(f"{source_dataset}/jsons_step1/clip_{scenario_idx + 1:04d}_*.json")[0]
    with open(json_file, "r") as f:
        return json.load(f)


# -------------------------------------------------
# Select indices
# -------------------------------------------------
# Real per-clip density (each clip JSON's own 'density' field) instead of an artificial
# positional split -- matches the fix already applied and verified in AP_evaluator/JRDB_eval.py.
if args.o in DENSITY_MODES:
    selected_idx = [i for i, key in enumerate(seq_keys) if _load_clip_json(key)["density"] == args.o]

# -------------------------------------------------
# Region-based modes
# -------------------------------------------------
elif args.o in region_modes:

    target_region = region_modes[args.o]

    selected_idx = []

    for i, key in enumerate(seq_keys):

        data = _load_clip_json(key)

        if "globe_region" in data:
            globe = data["globe_region"]
        else:
            country = country_mapping.get(data["country"], data["country"])
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
    whole=False,  # gt/pred are now correctly {scenario_idx: {sub_clip: groups}} (compute_groupings.py's
                  # collapsing bug is fixed), so each GT_dict[i] is itself a per-sub-clip dict -- whole=False
                  # is what iterates that nested structure correctly (whole=True would treat GT_dict[i]'s
                  # own keys ('0'-'4', each length 1) as if they were groups and filter every one out as a
                  # false "singleton", leaving nothing to evaluate).
    alone=False,#True,
    pre_dict=pred,
    GT_dict=gt
)

pre_d, rec_d, f1_d = Evaluater()

print(pre_d)
print(rec_d)
print(f1_d)
