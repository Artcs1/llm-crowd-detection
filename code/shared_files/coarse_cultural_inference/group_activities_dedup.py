"""
Group-level cultural activities with canonical deduplication
------------------------------------------------------------
Standalone post-processing step that lets you link a *group's bounding box* to
that group's activities, with the activity labels canonicalised through the
same global map produced by `activities_vocab.py`.

For each model it reads:
  * the raw merged inference `annotations.json` (source of per-group bboxes), and
  * the global `canonical_map` saved by `activities_vocab.py`,
and writes `annotations_ca.json` alongside them: a copy of `annotations.json`
with one field added to every group's `cultural_output`: `group_activity_dedup`
(the canonicalised, deduplicated activity labels). Since `bbox` already lives
on the group dict right next to `cultural_output`, this gives a direct link
from a group's bounding box to its deduplicated activities.

`annotations.json` itself is never modified.

Pipeline order:
  preprocess_cultural.py -> merge_inference_results.py (annotations.json)
  -> activities_vocab.py (canonical_map)
  -> group_activities_dedup.py (this script)
"""

import os
import json
import argparse

import pandas as pd
from tqdm.auto import tqdm

# Reuse the exact cleaning + data paths from the vocab step so the group-level
# canonicalisation matches the frame-level one.
from activities_vocab import DATA_PATHS, clean_activities


def add_group_activity_dedup(data, canonical_map):
    """Add `group_activity_dedup` to every group's `cultural_output`, in place.

    For each group: lowercase its raw `group_activity` labels (None -> []),
    clean them the same way the vocabulary build does, then map each term
    through the global canonical_map. Stored as a sorted list of unique
    canonical labels.
    """
    for ann in data["annotations"]:
        for g in ann["groups"]:
            co = g.setdefault("cultural_output", {})
            raw = co.get("group_activity") or []
            raw_lower = [a.lower() for a in raw]

            dedup = sorted({
                canonical_map.get(term, term) for term in clean_activities(raw_lower)
            })
            co["group_activity_dedup"] = dedup

    return data


def process_model(model_dir, canonical_map, ca_out_name):
    """Load one model's annotations.json, add group_activity_dedup, and write
    the result as annotations_ca.json in the same directory."""
    annotations_path = os.path.join(model_dir, "annotations.json")
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"annotations.json not found: {annotations_path}")

    with open(annotations_path, "r") as f:
        data = json.load(f)

    add_group_activity_dedup(data, canonical_map)

    ca_out_path = os.path.join(model_dir, ca_out_name)
    with open(ca_out_path, "w") as f:
        json.dump(data, f, indent=4)

    return ca_out_path


def main():
    parser = argparse.ArgumentParser(
        description="Add canonicalised group-level activities into each model's "
                     "annotations.json, saved as annotations_ca.json."
    )
    parser.add_argument(
        "--mode", choices=["allVLM", "indVLM"], default="allVLM",
        help="Selects which vocab-results pickle to read the canonical_map from.",
    )
    parser.add_argument(
        "--vocab-results", default=None,
        help="Path to the vocab-results pickle from activities_vocab.py "
             "(default: <mode>_activity_vocab_deduplication_results.pkl).",
    )
    parser.add_argument(
        "--ca-out-name", default="annotations_ca.json",
        help="Output filename written into each model dir.",
    )
    args = parser.parse_args()

    if args.vocab_results is None:
        args.vocab_results = f"{args.mode}_activity_vocab_deduplication_results.pkl"

    if not os.path.exists(args.vocab_results):
        raise FileNotFoundError(
            f"Vocab-results pickle not found: {args.vocab_results}\n"
            f"Run activities_vocab.py --mode {args.mode} first (it writes the canonical_map)."
        )

    results = pd.read_pickle(args.vocab_results)
    canonical_map = results["canonical_map"]
    print(f"Loaded canonical_map with {len(canonical_map)} terms from {args.vocab_results}")

    # Model directories = the dirs holding each model's pd_annotations.pkl.
    model_dirs = {key: os.path.dirname(path) for key, path in DATA_PATHS.items()}

    for key, model_dir in tqdm(model_dirs.items(), desc="Models"):
        ca_out_path = process_model(model_dir, canonical_map, args.ca_out_name)
        print(f"  [{key}] wrote {ca_out_path}")


if __name__ == "__main__":
    main()

