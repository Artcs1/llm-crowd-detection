"""
Group-level cultural activities with canonical deduplication
------------------------------------------------------------
Standalone post-processing step that lets you link a *group's bounding box* to
that group's activities, with the activity labels canonicalised through the
same global map produced by `activities_vocab.py`.

It does NOT change the frame-level outputs. For each model it reads:
  * the raw merged inference `annotations.json` (source of per-group bboxes),
  * the frame-level `pd_annotation_dedup.pkl` (used as the row base), and
  * the global `canonical_map` saved by `activities_vocab.py`,
and writes a new `pd_groups_dedup.pkl` alongside them.

`pd_groups_dedup.pkl` is the frame-level dedup dataframe with four new
list-columns appended, index-aligned by group so `group_bboxes[i]` corresponds
to group `i`'s activities:

  group_ids            : list[int]              — groupId per group (0-indexed)
  group_bboxes         : list[[x1,y1,x2,y2]]    — raw pixel xyxy (1920x1080 space)
  group_activity       : list[list[str]]        — per group, lowercased raw labels
  group_activity_dedup : list[list[str]]        — per group, canonicalised & unique

Frames with zero groups get empty lists in all four columns.

Pipeline order:
  preprocess_cultural.py -> merge_inference_results.py (annotations.json)
  -> activities_vocab.py (canonical_map + pd_annotation_dedup.pkl)
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


def build_annotation_lookup(annotations_path):
    """Map (clip, frame) -> annotation dict for one model's annotations.json.

    clip  = videoFolder.split('/')[-2]      e.g. "clip_0001"
    frame = videoInfo['annotationFrame'] + 1 (matches the frame-level dataframe)
    """
    with open(annotations_path, "r") as f:
        data = json.load(f)

    lut = {}
    for ann in data["annotations"]:
        clip = ann["videoFolder"].split("/")[-2]
        frame = ann["videoInfo"]["annotationFrame"] + 1
        lut[(clip, frame)] = ann
    return lut


def group_columns_for_annotation(ann, canonical_map):
    """Return the four aligned per-group lists for a single annotation.

    Groups are taken in file order (groupId is 0-indexed and sequential, so
    this preserves the groupId ordering). Missing/None activity lists become [].
    """
    group_ids, group_bboxes, group_activity, group_activity_dedup = [], [], [], []

    for g in ann["groups"]:
        co = g.get("cultural_output", {}) or {}
        raw = co.get("group_activity") or []
        raw_lower = [a.lower() for a in raw]

        # Same cleaning as the vocabulary build, then map through the global
        # canonical map; store as sorted unique canonical labels.
        dedup = sorted({
            canonical_map.get(term, term) for term in clean_activities(raw_lower)
        })

        group_ids.append(g["groupId"])
        group_bboxes.append(g["bbox"])
        group_activity.append(raw_lower)
        group_activity_dedup.append(dedup)

    return group_ids, group_bboxes, group_activity, group_activity_dedup


def process_model(model_dir, canonical_map, base_name, out_name):
    """Attach group-level columns to one model's frame-level dedup dataframe."""
    base_path = os.path.join(model_dir, base_name)
    annotations_path = os.path.join(model_dir, "annotations.json")

    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Base dataframe not found: {base_path}\n"
            f"Run activities_vocab.py first to produce '{base_name}'."
        )
    if not os.path.exists(annotations_path):
        raise FileNotFoundError(f"annotations.json not found: {annotations_path}")

    df = pd.read_pickle(base_path).copy()  # base is read-only; work on a copy
    lut = build_annotation_lookup(annotations_path)

    ids_col, bbox_col, act_col, dedup_col = [], [], [], []
    unmatched = 0
    for clip, frame, n in zip(df["clip"], df["frame"], df["num_groups"]):
        ann = lut.get((clip, frame))
        if ann is None:
            unmatched += 1
            ids_col.append([]); bbox_col.append([]); act_col.append([]); dedup_col.append([])
            continue
        gi, gb, ga, gd = group_columns_for_annotation(ann, canonical_map)
        # Sanity: per-group lists must agree with the frame's num_groups.
        assert len(gi) == n, f"{clip} frame {frame}: {len(gi)} groups vs num_groups={n}"
        ids_col.append(gi); bbox_col.append(gb); act_col.append(ga); dedup_col.append(gd)

    df["group_ids"] = ids_col
    df["group_bboxes"] = bbox_col
    df["group_activity"] = act_col
    df["group_activity_dedup"] = dedup_col

    out_path = os.path.join(model_dir, out_name)
    df.to_pickle(out_path)

    if unmatched:
        print(f"    WARNING: {unmatched} rows had no matching annotation (empty group lists).")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Attach group-level bboxes + canonicalised activities to each model's dedup dataframe."
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
        "--base-name", default="pd_annotation_dedup.pkl",
        help="Frame-level dataframe filename used as the row base, per model dir.",
    )
    parser.add_argument(
        "--out-name", default="pd_groups_dedup.pkl",
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
        out_path = process_model(model_dir, canonical_map, args.base_name, args.out_name)
        print(f"  [{key}] wrote {out_path}")


if __name__ == "__main__":
    main()
