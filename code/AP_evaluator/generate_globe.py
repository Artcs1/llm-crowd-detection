"""
build_table.py
--------------
Reads AP scores from per-region CSV files (AF.csv, AN.csv, …, O.csv),
infers model family / parameter size / modality / detector-source from each
filename (no hardcoded model list), and writes one LaTeX table PER DETECTOR
TYPE (DetAny3D / UniDepth / WildDet3D) that mirrors the paper style.

CSV format expected:
    name,G1,G2,G3,G4,G5,AP
    results_Cosmos-Reason2-2B_llm_unidepth_3D_p1.txt,...,47.94
    ...

Filename parsing rules (same logic as the reference script):
  - Strip leading "results_" / "results_full_" and trailing extension.
  - Modality  : "vlm" anywhere in name → VLM, else LLM.
  - Param size: first match of  \\d+B(-A\\d+B)?  (e.g. 3B, 30B-A3B).
  - Family    : the token immediately before the param-size token when the
                name is split on [-_], cleaned up so VL / VL- suffixes are
                stripped (Qwen2.5-VL-7B → family Qwen2.5).
  - Detector  : detany / unidepth / wilddet, matched anywhere in the name.
                Rows with different detectors are no longer collapsed into
                the same key — each detector gets its own table.
  - Stage     : SFT / RL training-stage suffix (e.g. "-SFT", "-RL") right
                before the modality/detector tokens. Stripped before family
                matching so "Qwen2.5-7B-Instruct-RL" still resolves to
                family "Qwen2.5", params "7B" — but tagged with stage "RL"
                so it gets its own row instead of colliding with the base
                model's row.

Sorting:
  - Detectors : DetAny3D, UniDepth, WildDet3D (then any unrecognized ones).
  - Families  : alphabetical (override via FAMILY_ORDER if desired).
  - Param sizes: by numeric value of the leading integer (3B < 7B < 30B …).
  - Modality  : VLM before LLM.
  - Stage     : base model first, then +SFT, then +RL.
"""

import os
import re
import csv
import argparse
from collections import defaultdict

# ── 1. Constants ──────────────────────────────────────────────────────────────

REGIONS = ["AF", "AN", "CA", "EU", "GE", "LA", "LE", "ME", "NE", "SA"]#, "O"]

# Optional explicit family ordering (families not listed sort alphabetically
# after the ones listed here).
FAMILY_ORDER = [
    "Cosmos-Reason2",
    "Qwen2.5",
    "Qwen3",
    "Gemini",          # covers Gemini-3-Pro, gemini-2, …
]

# Detector / depth-source labels + display order
DETECTOR_LABELS = {
    "detany": "DetAny3D",
    "unidepth": "UniDepth",
    "wilddet": "WildDet3D",
}
DETECTOR_ORDER = {"detany": 1, "unidepth": 2, "wilddet": 3}

# Training-stage labels + display order (base model first)
STAGE_ORDER = {"": 0, "SFT": 1, "RL": 2}


# ── 2. Filename parser ────────────────────────────────────────────────────────

def extract_model_info(raw_name: str):
    """
    Infer (family, params, modality, detector, stage) from a result filename.

    Examples
    --------
    results_Cosmos-Reason2-2B_llm_unidepth_3D_p1.txt  →  ('Cosmos-Reason2', '2B',  'LLM', 'unidepth', '')
    results_Qwen2.5-VL-72B-Instruct_vlm_image_detany_3D_p1.txt →  ('Qwen2.5', '72B', 'VLM', 'detany', '')
    results_Qwen3-VL-30B-A3B-Instruct_vlm_wilddet_…   →  ('Qwen3',          '30B-A3B', 'VLM', 'wilddet', '')
    results_Qwen2.5-7B-Instruct-RL_llm_detany_3D_p1.txt →  ('Qwen2.5', '7B', 'LLM', 'detany', 'RL')
    results_gemini-2-flash_vlm_…                       →  ('gemini-2',       '',    'VLM', 'unknown', '')
    """
    name = raw_name.strip()
    # Strip leading tag and extension
    name = re.sub(r'^results_(full_)?', '', name)
    name = re.sub(r'\.(txt|csv)$', '', name)

    # ── training stage (SFT / RL) ──────────────────────────────────────────────
    # Strip it out early so it never interferes with family/param parsing, but
    # remember it so the row can be tagged and kept separate from the base model.
    stage = ""
    stage_match = re.search(r'-(SFT|RL)(?=[-_]|$)', name, re.IGNORECASE)
    if stage_match:
        stage = stage_match.group(1).upper()
        name = name[:stage_match.start()] + name[stage_match.end():]

    # ── modality ──────────────────────────────────────────────────────────────
    # Match "vlm" bounded by non-alpha chars on both sides so "vlm_image" is
    # also caught (underscore is not a \w word boundary anchor in \b).
    modality = "VLM" if re.search(r'(?<![a-zA-Z])vlm(?![a-zA-Z])', name, re.IGNORECASE) else "LLM"

    # ── detector / depth source ────────────────────────────────────────────────
    detector_match = re.search(r'(?<![a-zA-Z])(detany|unidepth|wilddet)(?![a-zA-Z])', name, re.IGNORECASE)
    detector = detector_match.group(1).lower() if detector_match else "unknown"

    # ── param size ────────────────────────────────────────────────────────────
    # Match e.g. 30B-A3B first, then plain NB
    param_match = re.search(r'(\d+B-A\d+B|\d+B)', name)
    params = param_match.group(1) if param_match else ""

    # ── family ────────────────────────────────────────────────────────────────
    # Strategy: take the portion of the name before the param token (or before
    # the first underscore if no param), split on [-], drop trailing
    # "VL"/"Instruct"-like tokens, join back with "-".
    if param_match:
        prefix = name[:param_match.start()]          # e.g. "Qwen2.5-VL-"
    else:
        prefix = name.split('_')[0]                  # e.g. "gemini-2-flash"

    prefix = prefix.rstrip('-_')                     # strip trailing separators
    tokens = re.split(r'[-]', prefix)
    # Drop pure noise tokens: "VL", "Instruct", empty strings
    noise = {"VL", "Instruct", ""}
    tokens = [t for t in tokens if t not in noise]
    family = "-".join(tokens) if tokens else prefix

    return family, params, modality, detector, stage


# ── 3. Sorting helpers ────────────────────────────────────────────────────────

def _family_sort_key(family: str):
    """Sort by FAMILY_ORDER position, then alphabetically."""
    for i, prefix in enumerate(FAMILY_ORDER):
        if family.lower().startswith(prefix.lower()):
            return (i, family)
    return (len(FAMILY_ORDER), family)


def _param_sort_key(params: str):
    """Sort by the leading integer (3B < 7B < 30B)."""
    m = re.match(r'(\d+)', params)
    return int(m.group(1)) if m else 0


# ── 4. CSV loading ────────────────────────────────────────────────────────────

def load_ap_scores(csv_dir: str):
    """
    Returns
    -------
    ap_by_detector       : dict  detector → region → { (family, params, modality, stage): float }
    sorted_keys_by_detector : dict  detector → sorted list of (family, params, modality, stage) tuples
    detectors_present    : sorted list of detector keys found, in display order
    """
    ap_by_detector: defaultdict = defaultdict(lambda: {r: {} for r in REGIONS})
    all_keys_by_detector: defaultdict = defaultdict(set)
    detectors_present: set = set()

    for region in REGIONS:
        path = os.path.join(csv_dir, f"results/{region}.csv")
        if not os.path.isfile(path):
            print(f"  [WARNING] {path} not found – skipping region {region}")
            continue

        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        for row in rows:
            raw = row.get("name", "").strip()
            if not raw:
                continue
            try:
                score = float(row["AP"])
            except (KeyError, ValueError):
                continue

            family, params, modality, detector, stage = extract_model_info(raw)
            key = (family, params, modality, stage)
            detectors_present.add(detector)

            if key in ap_by_detector[detector][region]:
                print(f"  [WARNING] Duplicate key {key} for detector '{detector}' in {region}: '{raw}'")
            ap_by_detector[detector][region][key] = score
            all_keys_by_detector[detector].add(key)

    # Sort each detector's keys: family order → param size → modality (VLM before LLM) → stage
    sorted_keys_by_detector = {}
    for detector, all_keys in all_keys_by_detector.items():
        sorted_keys_by_detector[detector] = sorted(
            all_keys,
            key=lambda k: (
                _family_sort_key(k[0]),
                _param_sort_key(k[1]),
                0 if k[2] == "VLM" else 1,
                STAGE_ORDER.get(k[3], 999),
            ),
        )

    ordered_detectors = sorted(detectors_present, key=lambda d: DETECTOR_ORDER.get(d, 999))

    return ap_by_detector, sorted_keys_by_detector, ordered_detectors


# ── 5. LaTeX helpers ──────────────────────────────────────────────────────────

def fmt(val) -> str:
    if val is None:
        return r"\multicolumn{1}{c}{—}"
    return f"{val:.2f}"


def colorize(val, best_val, worst_val) -> str:
    s = fmt(val)
    if val is None:
        return s
    if best_val is not None and abs(val - best_val) < 1e-9:
        return r"\textcolor{red}{" + s + "}"
    if worst_val is not None and abs(val - worst_val) < 1e-9:
        return r"\textcolor{blue}{" + s + "}"
    return s


def family_tex(family: str) -> str:
    """Format family name for \\shortstack: split on '-' → one line each."""
    parts = family.split("-")
    return r"\\".join(parts)


def stage_label(stage: str) -> str:
    """Display label for the stage column: base model gets 'M', SFT/RL get '+SFT' / '+RL'."""
    return f"+{stage}" if stage else "M"


# ── 6. LaTeX generation ───────────────────────────────────────────────────────

def build_latex_for_detector(ap: dict, sorted_keys: list, detector_key: str):
    """Build a single table*'s LaTeX, scoped to one detector's data."""

    if not sorted_keys:
        return ""

    detector_label = DETECTOR_LABELS.get(detector_key, detector_key)

    # ── decide, for each adjacent pair of rows, what decoration (if any)
    # separates them: a \cmidrule when param size changes, an \addlinespace
    # when modality changes (within the same size), or nothing when only the
    # stage (base/+SFT/+RL) changes. This mirrors the actual rendering loop
    # below and lets us compute correct multirow spans (data rows *plus* any
    # decoration line that itself occupies vertical space inside the span).
    decorations = []  # decorations[i] separates sorted_keys[i] and sorted_keys[i+1]
    for i in range(len(sorted_keys) - 1):
        fam, sz, mod, stage = sorted_keys[i]
        next_fam, next_sz, next_mod, next_stage = sorted_keys[i + 1]
        if next_fam != fam:
            decorations.append(None)  # family boundary uses \hline, handled separately
        elif next_sz != sz:
            decorations.append("cmidrule")
        elif next_mod != mod:
            decorations.append("addlinespace")
        else:
            decorations.append(None)  # stage-only change: no decoration line

    def compute_span(group_key_fn):
        """
        span(g) = (# rows in group g) + (# decoration lines strictly between
        two rows that are both in group g). This correctly accounts for
        \\cmidrule/\\addlinespace consuming visual space inside a multirow
        span, while stage-only transitions (no decoration) don't inflate it.
        """
        counts = defaultdict(int)
        for k in sorted_keys:
            counts[group_key_fn(k)] += 1
        for i, dec in enumerate(decorations):
            if dec is None:
                continue
            g1 = group_key_fn(sorted_keys[i])
            g2 = group_key_fn(sorted_keys[i + 1])
            if g1 == g2:
                counts[g1] += 1
        return counts

    family_span = compute_span(lambda k: k[0])
    size_span = compute_span(lambda k: (k[0], k[1]))
    mod_span = compute_span(lambda k: (k[0], k[1], k[2]))

    lines = []
    def w(s=""): lines.append(s)

    w(r"\begin{table*}[h!]")
    w(r"    \centering")
    w(r"    \footnotesize")
    w(
        r"    \caption{Full fine-grained experiments performance comparison on the EgoGroups dataset using "
        r"\textbf{" + detector_label + r"} as the 3D detection/depth source, segmented"
    )
    w(r"by GLOBE regions. Model combination")
    w(r"indicates the backbone family, number of parameters, and input modality. We report")
    w(r"each region's AP score individually, and compute MAP as the average across all regions.")
    w(r"The best and worst values per row are highlighted in red and blue, respectively.}")
    w(r"    \resizebox{0.6\linewidth}{!}{")
    w(r"    \begin{tabular}{lllllccccccccc}")
    w(r"    \toprule")
    w(r"        \multirow{1}{*}{Model} & & & ")
    region_header = " & ".join(r"\textbf{" + r + "}" for r in REGIONS)
    w(r"        & " + region_header + r" \\")
    w(r"    \midrule")

    emitted_families: set = set()
    emitted_sizes: set = set()
    emitted_mods: set = set()
    prev_family = None

    for idx, (fam, sz, mod, stage) in enumerate(sorted_keys):

        # ── family separator ──────────────────────────────────────────────────
        if prev_family is not None and fam != prev_family:
            w(r"        \hline")
        prev_family = fam

        # ── per-row scores ────────────────────────────────────────────────────
        scores = [ap[r].get((fam, sz, mod, stage)) for r in REGIONS]
        valid = [s for s in scores if s is not None]
        best = max(valid) if valid else None
        worst = min(valid) if valid else None
        cells = [colorize(s, best, worst) for s in scores]

        # ── col 1: family multirow ────────────────────────────────────────────
        if fam not in emitted_families:
            span = family_span[fam]
            fam_cell = (
                r"        \multirow{" + str(span) + r"}{*}{\rotatebox{10}{\shortstack{"
                + family_tex(fam) + r"}}}"
            )
            emitted_families.add(fam)
        else:
            fam_cell = "        "

        # ── col 2: param-size multirow ────────────────────────────────────────
        size_key = (fam, sz)
        if size_key not in emitted_sizes:
            span_sz = size_span[size_key]
            sz_label = sz if sz else r"\textemdash"
            size_cell = r" & \multirow{" + str(span_sz) + r"}{*}{" + sz_label + "} "
            emitted_sizes.add(size_key)
        else:
            size_cell = " & "

        # ── col 3: modality multirow ──────────────────────────────────────────
        mod_key = (fam, sz, mod)
        if mod_key not in emitted_mods:
            span_mod = mod_span[mod_key]
            mod_cell = r" & \multirow{" + str(span_mod) + r"}{*}{" + mod + "} "
            emitted_mods.add(mod_key)
        else:
            mod_cell = " & "

        # ── col 4: stage / method ─────────────────────────────────────────────
        stage_cell = r" & " + stage_label(stage) + " "

        # ── assemble full row ─────────────────────────────────────────────────
        w(fam_cell + size_cell + mod_cell + stage_cell + " & " + " & ".join(cells) + r"  \\")

        # ── inter-row decorations ─────────────────────────────────────────────
        if idx + 1 < len(sorted_keys):
            next_fam, next_sz, next_mod, next_stage = sorted_keys[idx + 1]
            if next_fam == fam:                # still same family
                if next_sz != sz:               # new param size → cmidrule
                    w(r"        \cmidrule(lr){3-14}")
                elif next_mod != mod:            # same size, new modality → space
                    w(r"        \addlinespace")
                # same size & modality, different stage (SFT/RL) → no decoration,
                # row simply follows directly beneath the base model's row.

    w(r"    \bottomrule")
    w(r"    \end{tabular}")
    w(r"    }")
    w(r"    \label{table:sup_egogroups_globe_regions_" + detector_key + "}")
    w(r"\end{table*}")

    return "\n".join(lines)


def build_latex(ap_by_detector: dict, sorted_keys_by_detector: dict, ordered_detectors: list, output_path: str):
    tables = []
    for detector_key in ordered_detectors:
        table_latex = build_latex_for_detector(
            ap_by_detector[detector_key],
            sorted_keys_by_detector.get(detector_key, []),
            detector_key,
        )
        if table_latex:
            tables.append(table_latex)
        else:
            print(f"  [WARNING] no rows found for detector '{detector_key}', skipping its table.")

    tex = "\n\n".join(tables)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(tex)
    print(f"\nLaTeX tables written to: {output_path}")
    return tex


# ── 7. Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build LaTeX results tables (one per detector type) from per-region CSV files."
    )
    parser.add_argument(
        "--csv_dir", default=".",
        help="Directory containing results/AF.csv, results/AN.csv, … results/SA.csv  (default: current dir)",
    )
    parser.add_argument(
        "--output", default="table_egogroups.tex",
        help="Output .tex file path  (default: table_egogroups.tex)",
    )
    args = parser.parse_args()

    print(f"Loading AP scores from: {os.path.abspath(args.csv_dir)}")
    ap_by_detector, sorted_keys_by_detector, ordered_detectors = load_ap_scores(args.csv_dir)

    print(f"\nDetectors found: {ordered_detectors}")
    for detector_key in ordered_detectors:
        sorted_keys = sorted_keys_by_detector.get(detector_key, [])
        print(f"\n[{DETECTOR_LABELS.get(detector_key, detector_key)}] "
              f"discovered {len(sorted_keys)} unique (family, params, modality, stage) combinations:")
        for key in sorted_keys:
            regions_found = [r for r in REGIONS if key in ap_by_detector[detector_key][r]]
            print(f"  {key}  →  found in {len(regions_found)}/{len(REGIONS)} regions")

    build_latex(ap_by_detector, sorted_keys_by_detector, ordered_detectors, args.output)


if __name__ == "__main__":
    main()
