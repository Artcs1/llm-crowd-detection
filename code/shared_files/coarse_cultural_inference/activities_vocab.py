from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os


"""
Consistent Semantic Deduplication Across Multiple Vocabularies
--------------------------------------------------------------
Guarantees that the same term always maps to the same canonical form,
regardless of which vocabulary it appears in.

This variant consumes the `pd_annotations.pkl` outputs produced by
`preprocess_cultural.py` (one per model, under
`<data_dir>/results_cultural/<model_name>/pd_annotations.pkl`).

Relative to the CVPR-era `cultural_<model>_final.pkl` schema, the only
structural differences that matter here are:
  * the activities column is named `activity` (singular), not `activities`;
  * each cell is a `list`, not a `set` (clean_activities handles both).
The merge key `file`, plus `globe_region` and `country`, are unchanged.

Strategy
--------
1. Union all vocabularies into a single global set of unique terms.
2. Sort the global set before deduplication so canonical selection is
   deterministic — sets are unordered, so without sorting the "winner"
   of each cluster would vary across Python runs.
3. Run deduplication once on the sorted global list → produces a global
   canonical map  {any_term: canonical_term}.
4. Apply the global map to each vocabulary individually.

This ensures "automobile" always maps to "car" (or whichever canonical
was chosen) across every vocabulary, never sometimes to "car" and
sometimes to "vehicle".

Modes
-----
allVLM : Union activities from all models per image first, then group by
         region. One vocabulary per region combining all models.
indVLM : Keep each model's activities separate. One vocabulary per
         (region, model) pair — labels include a model suffix.
"""


#MODEL_KEYS = ["dpsk", "gmni", "llva", "qw7b", "qw72b", "qw30b"]
MODEL_KEYS = ["llva", "qw7b", "qw72b", "qw30b"]

# Outputs of preprocess_cultural.py:
#   <data_dir>/results_cultural/<model_name>/pd_annotations.pkl
_RESULTS_DIR = "/lustre/nvwulf/projects/CascanteBonillaGroup-nvwulf/pchitale/data/eccv/results_cultural"

_RESULTS_DIR = "/lp-dev/jmurrugarral/llm-crowd-detection/code/annotations/results_cultural"

DATA_PATHS = {
    "dpsk":  f"{_RESULTS_DIR}/deepseek-vl2/pd_annotations.pkl",
    #"gmni":  f"{_RESULTS_DIR}/gemini/pd_annotations.pkl",
    "llva":  f"{_RESULTS_DIR}/llava-v1.6-mistral-7b-hf/pd_annotations.pkl",
    "qw7b":  f"{_RESULTS_DIR}/Qwen2.5-VL-7B-Instruct/pd_annotations.pkl",
    "qw72b": f"{_RESULTS_DIR}/Qwen2.5-VL-72B-Instruct/pd_annotations.pkl",
    "qw30b": f"{_RESULTS_DIR}/Qwen3-VL-30B-A3B-Instruct/pd_annotations.pkl",
}

# Column in pd_annotations.pkl holding the per-clip activity labels.
ACTIVITY_COL = "activity"


# ---------------------------------------------------------------------------
# Step 1 — build a global canonical map from the union of all terms
# ---------------------------------------------------------------------------

def build_global_canonical_map(
    vocabularies: list[set[str]],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, str]:
    """
    Union all vocabularies and deduplicate globally.

    Vocabularies are sets, so we sort the union before deduplication to
    guarantee that canonical selection is deterministic across runs.

    Returns
    -------
    canonical_map : dict[str, str]
        Maps *every* term (including canonicals) to its canonical form.
        e.g. {"automobile": "car", "vehicle": "car", "car": "car", ...}
    """
    global_vocab: list[str] = sorted(set().union(*vocabularies), key=lambda t: (len(t), t))

    print(f"Global vocabulary: {len(global_vocab)} unique terms across {len(vocabularies)} vocabularies.")

    print(f"\nLoading model '{model_name}'…")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(global_vocab)} terms…")
    embeddings = model.encode(global_vocab, convert_to_numpy=True, show_progress_bar=True)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / np.where(norms == 0, 1, norms)
    sim = cosine_similarity(embeddings_norm)

    assignment: dict[int, int] = {}
    for i in tqdm(range(len(global_vocab)), desc="Deduplicating terms"):
        if i in assignment:
            continue
        assignment[i] = i
        for j in range(i + 1, len(global_vocab)):
            if j not in assignment and sim[i, j] >= threshold:
                assignment[j] = i
                print(
                    f"  '{global_vocab[j]}' → '{global_vocab[i]}' "
                    f"(similarity={sim[i, j]:.3f})"
                )

    canonical_map: dict[str, str] = {
        global_vocab[idx]: global_vocab[canonical_idx]
        for idx, canonical_idx in assignment.items()
    }
    return canonical_map


# ---------------------------------------------------------------------------
# Step 2 — apply the global map to each vocabulary
# ---------------------------------------------------------------------------

def apply_canonical_map(
    vocabularies: list[set[str]],
    canonical_map: dict[str, str],
) -> tuple[list[set[str]], dict[str, list[str]]]:
    """
    Replace every term in every vocabulary with its canonical form.

    Returns
    -------
    deduped_vocabs : list[set[str]]
        Each vocabulary as a set of canonical terms (duplicates collapsed).

    global_duplicates : dict[str, list[str]]
        Canonical term → all terms that were merged into it (globally),
        sorted for readability.
    """
    global_duplicates: dict[str, list[str]] = {}
    for term, canonical in canonical_map.items():
        if term != canonical:
            global_duplicates.setdefault(canonical, []).append(term)
    global_duplicates = {k: sorted(v) for k, v in global_duplicates.items()}

    deduped_vocabs: list[set[str]] = [
        {canonical_map.get(term, term) for term in vocab}
        for vocab in vocabularies
    ]

    return deduped_vocabs, global_duplicates


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def deduplicate_vocabularies(
    vocabularies: list[set[str]],
    threshold: float = 0.85,
    model_name: str = "all-MiniLM-L6-v2",
) -> tuple[list[set[str]], dict[str, list[str]], dict[str, str]]:
    """
    End-to-end deduplication across multiple vocabularies.

    Returns
    -------
    deduped_vocabs      : deduplicated version of each input vocabulary (as sets)
    global_duplicates   : canonical → [semantic duplicates merged into it]
    canonical_map       : every term → its canonical form (full lookup table)
    """
    canonical_map = build_global_canonical_map(vocabularies, threshold, model_name)
    deduped_vocabs, global_duplicates = apply_canonical_map(vocabularies, canonical_map)
    return deduped_vocabs, global_duplicates, canonical_map


def clean_activities(activity_list):
    s = set(activity_list) if activity_list is not None else set()
    s.discard('')
    s = {term for term in s if not any('一' <= char <= '鿿' for char in term)}
    return s


# ---------------------------------------------------------------------------
# Write the deduplicated labels back into each model's pd_annotations.pkl
# ---------------------------------------------------------------------------

def write_back_deduped_columns(
    canonical_map: dict[str, str],
    dedup_col: str = "activity_dedup",
) -> None:
    """
    For every model in DATA_PATHS, write a copy of its pd_annotations.pkl with
    an added column of canonicalised activity labels. The original file is
    left untouched; the copy is saved alongside it as `pd_annotation_dedup.pkl`.

    The new column is derived per row from the original `activity` list:
    the same cleaning used to build the vocabulary is applied (drop empties
    and CJK terms), then each surviving term is mapped through the global
    canonical_map. The result is stored as a sorted list of unique canonical
    labels.
    """
    for key, path in DATA_PATHS.items():
        df = pd.read_pickle(path)  # fresh load — do NOT reuse the cleaned copies
        df[dedup_col] = df[ACTIVITY_COL].apply(
            lambda lst: sorted({
                canonical_map.get(term, term) for term in clean_activities(lst)
            })
        )
        out_path = os.path.join(os.path.dirname(path), "pd_annotation_dedup.pkl")
        df.to_pickle(out_path)
        print(f"  [{key}] wrote '{dedup_col}' to {out_path}")


# ---------------------------------------------------------------------------
# Vocabulary builders: allVLM and indVLM
# ---------------------------------------------------------------------------

def build_allVLM_vocabularies(merged_df: pd.DataFrame) -> tuple[list[set[str]], list[str]]:
    """
    One vocabulary per region, unioning activities across all models.
    Labels: region name.
    """
    merged_df = merged_df.copy()
    merged_df['activities_all'] = merged_df.apply(
        lambda row: set.union(*[
            row[f'activities_{k}'] if isinstance(row[f'activities_{k}'], set) else set()
            for k in MODEL_KEYS
        ]),
        axis=1,
    )
    by_region = (
        merged_df.groupby('globe_region')
        .agg({'activities_all': lambda x: set.union(*x)})
        .reset_index()
    )
    vocabularies = by_region['activities_all'].tolist()
    labels = by_region['globe_region'].tolist()
    return vocabularies, labels


def build_indVLM_vocabularies(merged_df: pd.DataFrame) -> tuple[list[set[str]], list[str]]:
    """
    One vocabulary per (region, model) pair, keeping model activities separate.
    Labels: "<region>_<model_key>".
    """
    vocabularies, labels = [], []
    for key in MODEL_KEYS:
        col = f'activities_{key}'
        by_region = (
            merged_df.groupby('globe_region')
            .agg({col: lambda x: set.union(*x)})
            .reset_index()
        )
        vocabularies.extend(by_region[col].tolist())
        labels.extend([f"{region}_{key}" for region in by_region['globe_region']])
    return vocabularies, labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Activity vocabulary deduplication.")
    parser.add_argument(
        "--mode",
        choices=["allVLM", "indVLM"],
        default="allVLM",
        help="allVLM: one vocab per region (all models unioned). "
             "indVLM: one vocab per (region, model) pair.",
    )
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--output", default=None, help="Output pickle path (default: <mode>_activity_vocab_deduplication_results.pkl)")
    parser.add_argument(
        "--dedup-col",
        default="activity_dedup",
        help="Name of the deduplicated-activity column written into each pd_annotation_dedup.pkl.",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = f"{args.mode}_activity_vocab_deduplication_results.pkl"

    dfs = {key: pd.read_pickle(path) for key, path in DATA_PATHS.items()}
    for key, df in dfs.items():
        df[ACTIVITY_COL] = df[ACTIVITY_COL].apply(clean_activities)

    # A clip appears once per sampled frame (e.g. frames 2/22/42), so `file`
    # alone is NOT unique — the join key must be (file, frame) to avoid a
    # many-to-many merge blow-up.
    merged_df = dfs[MODEL_KEYS[-1]][['file', 'frame']].copy()
    for key, df in dfs.items():
        merged_df = merged_df.merge(
            df[['file', 'frame', ACTIVITY_COL]], on=['file', 'frame'], how='inner'
        ).rename(columns={ACTIVITY_COL: f'activities_{key}'})
    merged_df = merged_df.merge(
        dfs[MODEL_KEYS[-1]][['file', 'frame', 'globe_region', 'country']],
        on=['file', 'frame'], how='inner',
    )

    if args.mode == "allVLM":
        vocabularies, labels = build_allVLM_vocabularies(merged_df)
    else:
        vocabularies, labels = build_indVLM_vocabularies(merged_df)

    print("=" * 60)
    for label, vocab in zip(labels, vocabularies):
        print(f"{label}: {sorted(vocab)}")
    print("=" * 60)

    deduped_vocabs, global_duplicates, canonical_map = deduplicate_vocabularies(
        vocabularies,
        threshold=args.threshold,
        model_name=args.embed_model,
    )

    print("\n" + "=" * 60)
    print("Global duplicate groups (canonical → duplicates):")
    if global_duplicates:
        for canonical, dupes in global_duplicates.items():
            print(f"  '{canonical}'  →  {dupes}")
    else:
        print("  (no duplicates found at this threshold)")

    print("\nDeduped vocabularies:")
    for label, deduped in zip(labels, deduped_vocabs):
        print(f"  {label}: {sorted(deduped)}")

    print("\nFull canonical map (term → canonical):")
    for term, canonical in sorted(canonical_map.items()):
        marker = "  ← merged" if term != canonical else ""
        print(f"  '{term}' → '{canonical}'{marker}")
    print("=" * 60)

    results = {
        "deduped_vocabularies": deduped_vocabs,
        "global_duplicates": global_duplicates,
        "canonical_map": canonical_map,
        "labels": labels,
        "mode": args.mode,
    }
    pd.to_pickle(results, args.output)
    print(f"\nSaved to {args.output}")

    print(f"\nWriting pd_annotation_dedup.pkl files (column '{args.dedup_col}'):")
    write_back_deduped_columns(canonical_map, dedup_col=args.dedup_col)
