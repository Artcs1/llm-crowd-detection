#!/usr/bin/env python3
"""
Generate LaTeX tables from three CSV files (scattered.txt, moderate.txt, crowded.txt).
Each file contains: name, G1, G2, G3, G4, G5, AP
Row type is determined by filename suffix:
  - _p1        -> M (metadata)
  - _baseline1 -> A (all frames)
  - _baseline2 -> S (single frame)

Filenames also encode a detector / depth source (detany / unidepth / wilddet).
One LaTeX table is produced PER detector type, each with Scattered/Moderate/Crowded
columns side by side (same layout as before).
"""

import csv
import re
import sys
from collections import defaultdict, Counter

# ── constants ────────────────────────────────────────────────────────────────

DETECTOR_LABELS = {
    'detany': 'DetAny3D',
    'unidepth': 'UniDepth',
    'wilddet': 'WildDet3D',
}
DETECTOR_ORDER = {'detany': 1, 'unidepth': 2, 'wilddet': 3}

SIZE_ORDER = {
    "Cosmos":  ["2B", "8B"],
    "Qwen2.5": ["3B", "7B", "32B", "72B"],
    "Qwen3":   ["4B", "30B"],
    "Gemini":  ["3-Pro"],
}
FAMILY_ORDER = ["Cosmos", "Qwen2.5", "Qwen3", "Gemini"]

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    """Return dict: name -> {G1..G5, AP}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["name"]] = {k: row[k] for k in ("G1", "G2", "G3", "G4", "G5", "AP")}
    return data


def row_type(name):
    # SFT / RL training-stage variants get their own row-type label so they
    # show up as separate rows (behind the base "+ metadata" / M row) rather
    # than being merged into it.
    stage_match = re.search(r'-(SFT|RL)(?:_|$)', name, re.IGNORECASE)
    if stage_match:
        return f"+{stage_match.group(1).upper()}"

    if "_p1" in name:
        return "M"
    elif "_baseline2" in name:
        return "S"
    elif "_baseline1" in name:
        return "A"
    return None


def _strip_prefix_ci(text, prefix):
    """Case-insensitive prefix strip. Returns remainder, or None if no match."""
    if text[:len(prefix)].lower() == prefix.lower():
        return text[len(prefix):]
    return None


def _strip_suffix_ci(text, suffix):
    if suffix and text[-len(suffix):].lower() == suffix.lower():
        return text[:-len(suffix)]
    return text


def parse_model_info(name):
    """
    Returns (family, size, modality, detector)
    family: Cosmos | Qwen2.5 | Qwen3 | Gemini | <raw model string, for unrecognized names>
    size  : 2B, 8B, 3B, 7B, 32B, 72B, 4B, 30B, 3-Pro, "?" (unrecognized) …
    modality: LLM | VLM
    detector: detany | unidepth | wilddet | unknown

    Matching is case-insensitive. If a filename only *partially* matches a known
    family prefix but has leftover text after stripping the expected size/suffix
    tokens (e.g. "qwen2.5-7b-groups-grpo"), it is treated as NOT matching that
    family — falling back to using the raw model string as its own family. This
    prevents unrelated variants (e.g. two different ablations that happen to
    share a prefix) from being silently merged into the same row and
    overwriting each other.
    """
    stem = name.replace("results_", "").replace(".txt", "")

    # modality
    if "_vlm_" in stem.lower():
        modality = "VLM"
    else:
        modality = "LLM"

    # detector / depth source (also allow end-of-string match)
    detector_match = re.search(r'_(detany|unidepth|wilddet)(?:_|$)', stem, re.IGNORECASE)
    detector = detector_match.group(1).lower() if detector_match else "unknown"

    # model name part
    model_part = re.split(r'_llm_|_vlm_', stem, flags=re.IGNORECASE)[0]

    # Strip SFT / RL training-stage suffix before family/size matching so
    # "Qwen2.5-7B-Instruct-RL" still resolves to family=Qwen2.5, size=7B.
    model_part = re.sub(r'-(SFT|RL)$', '', model_part, flags=re.IGNORECASE)

    def try_family(prefix, family_name, extra_suffixes=()):
        rest = _strip_prefix_ci(model_part, prefix)
        if rest is None:
            return None
        for suf in extra_suffixes:
            rest = _strip_suffix_ci(rest, suf)
        size_match = re.match(r"(\d+B(?:-A\d+B)?)", rest, re.IGNORECASE)
        if not size_match:
            return None
        leftover = rest[size_match.end():]
        if leftover != "":
            # extra text after the size token (e.g. "-groups-grpo") means this
            # isn't a clean match for this family -> don't force-classify it.
            return None
        size = re.sub(r"-A\d+B$", "", size_match.group(1), flags=re.IGNORECASE)
        return family_name, size

    family = None
    size = None

    if model_part.lower() == "gemini":
        family, size = "Gemini", "3-Pro"
    else:
        for prefix, fam, sufs in [
            ("Cosmos-Reason2-", "Cosmos", ()),
            ("Qwen2.5-VL-", "Qwen2.5", ("-Instruct",)),
            ("Qwen2.5-", "Qwen2.5", ("-Instruct",)),
            ("Qwen3-VL-", "Qwen3", ("-Instruct-2507", "-Instruct")),
            ("Qwen3-", "Qwen3", ("-Instruct-2507", "-Instruct")),
        ]:
            result = try_family(prefix, fam, sufs)
            if result is not None:
                family, size = result
                break

    if family is None:
        # Unrecognized / partially-matching name: keep it as its own distinct
        # family (using the raw, case-preserved string) rather than merging
        # it into a known family or a shared 'Unknown' bucket.
        family = model_part
        size = "?"

    return family, size, modality, detector


def fmt(val, bold=False):
    try:
        f = float(val)
        s = f"{f:.2f}"
    except (TypeError, ValueError):
        s = str(val)
    return r"\textbf{" + s + "}" if bold else s


# ── find column-wise maxima for bolding ──────────────────────────────────────

def find_maxima(rows):
    """
    rows: list of dicts with keys sc, mo, cr each containing G1..AP as floats,
          plus 'row_type' (M/A/S). Scoped to a single detector's rows.
    Returns per-split per-col maximum (only among M rows, matching paper).
    """
    cols = ["G1", "G2", "G3", "G4", "G5", "AP"]
    maxima = {split: {c: -1 for c in cols} for split in ("sc", "mo", "cr")}
    for row in rows:
        if row["row_type"] == "M":
            for split in ("sc", "mo", "cr"):
                for c in cols:
                    try:
                        v = float(row[split][c])
                        if v > maxima[split][c]:
                            maxima[split][c] = v
                    except (TypeError, ValueError):
                        pass
    return maxima


# ── build structured data ────────────────────────────────────────────────────

def build_rows_by_detector(sc_data, mo_data, cr_data):
    """
    Collects one entry per (detector, family, size, modality, row_type).
    Returns dict: detector -> list of rows, ordered as in the paper.
    """
    all_names = set(sc_data) | set(mo_data) | set(cr_data)

    # group: detector -> family -> size -> modality -> row_type -> row
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))))
    detectors_present = set()

    for name in all_names:
        rt = row_type(name)
        if rt is None:
            continue
        family, size, modality, detector = parse_model_info(name)
        detectors_present.add(detector)

        sc = sc_data.get(name, {c: "--" for c in ("G1", "G2", "G3", "G4", "G5", "AP")})
        mo = mo_data.get(name, {c: "--" for c in ("G1", "G2", "G3", "G4", "G5", "AP")})
        cr = cr_data.get(name, {c: "--" for c in ("G1", "G2", "G3", "G4", "G5", "AP")})

        rt_group = grouped[detector][family][size][modality]
        if rt in rt_group:
            print(f"Warning: duplicate row for detector='{detector}' family='{family}' "
                  f"size='{size}' modality='{modality}' row_type='{rt}' from name='{name}' "
                  f"overwrote a previous entry.", file=sys.stderr)
        rt_group[rt] = {
            "sc": sc, "mo": mo, "cr": cr, "row_type": rt,
            "family": family, "size": size, "modality": modality, "detector": detector,
        }

    # flatten in canonical order, per detector
    rows_by_detector = {}
    RT_ORDER = ["S", "A", "M", "+SFT", "+RL"]
    for detector, fam_group in grouped.items():
        rows = []
        # known families first, in the paper's canonical order
        for family in FAMILY_ORDER:
            sizes = SIZE_ORDER.get(family, sorted(fam_group.get(family, {}).keys()))
            for size in sizes:
                if size not in fam_group.get(family, {}):
                    continue
                mod_group = fam_group[family][size]
                for modality in ["VLM", "LLM"]:
                    if modality not in mod_group:
                        continue
                    rt_group = mod_group[modality]
                    for rt in RT_ORDER:
                        if rt in rt_group:
                            rows.append(rt_group[rt])
                    # any other row types not in RT_ORDER (unexpected) still appended
                    for rt in sorted(rt_group.keys()):
                        if rt not in RT_ORDER:
                            rows.append(rt_group[rt])

        # any families NOT in FAMILY_ORDER (e.g. unrecognized/ablation names)
        # are still appended, sorted, so they're never silently dropped.
        extra_families = sorted(f for f in fam_group.keys() if f not in FAMILY_ORDER)
        for family in extra_families:
            sizes = sorted(fam_group[family].keys())
            for size in sizes:
                mod_group = fam_group[family][size]
                for modality in ["VLM", "LLM"]:
                    if modality not in mod_group:
                        continue
                    rt_group = mod_group[modality]
                    for rt in RT_ORDER:
                        if rt in rt_group:
                            rows.append(rt_group[rt])
                    for rt in sorted(rt_group.keys()):
                        if rt not in RT_ORDER:
                            rows.append(rt_group[rt])

        rows_by_detector[detector] = rows

    return rows_by_detector, detectors_present


# ── LaTeX generation for a single detector's table ───────────────────────────

def render_table(rows, detector_key):
    if not rows:
        return ""

    maxima = find_maxima(rows)
    cols = ["G1", "G2", "G3", "G4", "G5", "AP"]
    detector_label = DETECTOR_LABELS.get(detector_key, detector_key)

    lines = []
    lines.append(r"\begin{table*}[h!]")
    lines.append(r"    \centering")
    lines.append(r"    \footnotesize")
    lines.append(
        r"    \caption{Fine-grained performance comparison using \textbf{" + detector_label +
        r"} as the 3D detection/depth source, across Scattered, Moderate, and Crowded density "
        r"splits. Model combination indicates the backbone family, number of parameters, and "
        r"modality. MT indicates the processing approach (S: single frame, A: all frames, "
        r"M: + metadata). $G_1$--$G_5$ represent AP scores for group sizes 1--5$^+$, with AP "
        r"showing the average performance across all groups.}"
    )
    lines.append(r"    \resizebox{0.99\linewidth}{!}{")
    lines.append(r"    \begin{tabular}{llllcccccccccccccccccc}")
    lines.append(r"    \toprule")
    lines.append(r"        & \multirow{2}{*}{Model} &  & \multirow{2}{*}{MT}")
    lines.append(r"        & \multicolumn{6}{c}{\textbf{Scattered}}")
    lines.append(r"        & \multicolumn{6}{c}{\textbf{Moderate}} & \multicolumn{6}{c}{\textbf{Crowded}} \\")
    lines.append(r"        \cmidrule(lr){5-10} \cmidrule(lr){11-16} \cmidrule(lr){17-22}")
    lines.append(r"        & & & & G$_{1}$ & G$_{2}$ & G$_{3}$ & G$_{4}$ & G$_{5}$ & AP"
                 r" & G$_{1}$ & G$_{2}$ & G$_{3}$ & G$_{4}$ & G$_{5}$ & AP"
                 r" & G$_{1}$ & G$_{2}$ & G$_{3}$ & G$_{4}$ & G$_{5}$ & AP \\")
    lines.append(r"    \midrule")

    # track multirow spans
    prev_family = None
    prev_size = None
    prev_mod = None

    # pre-compute span counts
    family_counts = Counter()
    size_counts = Counter()
    mod_counts = Counter()
    for r in rows:
        family_counts[r["family"]] += 1
        size_counts[(r["family"], r["size"])] += 1
        mod_counts[(r["family"], r["size"], r["modality"])] += 1

    def cells(split_data, split_key):
        parts = []
        for c in cols:
            v = split_data.get(c, "--")
            try:
                is_bold = abs(float(v) - maxima[split_key][c]) < 0.005
            except (TypeError, ValueError):
                is_bold = False
            parts.append(fmt(v, is_bold))
        return " & ".join(parts)

    for idx, r in enumerate(rows):
        family = r["family"]
        size = r["size"]
        modality = r["modality"]
        rt = r["row_type"]

        # family cell
        if family != prev_family:
            fc = family_counts[family]
            family_display = family.replace('_', r'\_')
            family_cell = r"\multirow{" + str(fc) + r"}{*}{\rotatebox{90}{" + family_display + r"}}"
            if prev_family is not None:
                lines.append(r"        \hline")
            prev_family = family
            prev_size = None
            prev_mod = None
        else:
            family_cell = ""

        # size cell: start a new multirow block if size changed, or we just
        # crossed into a new family (idx == 0 always starts a new block)
        new_size_block = idx == 0 or size != prev_size or family != rows[idx - 1]["family"]
        if new_size_block:
            sc2 = size_counts[(family, size)]
            size_cell = r"\multirow{" + str(sc2) + r"}{*}{" + size + r"}"
            if prev_size is not None and prev_size != size:
                lines.append(r"        \cmidrule(lr){3-22}")
            prev_size = size
            prev_mod = None
        else:
            size_cell = ""

        # modality cell
        if modality != prev_mod:
            mc = mod_counts[(family, size, modality)]
            mod_cell = r"\multirow{" + str(mc) + r"}{*}{" + modality + r"}"
            prev_mod = modality
        else:
            mod_cell = ""

        sc_cells = cells(r["sc"], "sc")
        mo_cells = cells(r["mo"], "mo")
        cr_cells = cells(r["cr"], "cr")

        line = (f"         & {size_cell} & {mod_cell} & {rt}"
                f" & {sc_cells} & {mo_cells} & {cr_cells} \\\\")
        line = f"        {family_cell}" + line[8:]
        lines.append(line)

        # addlinespace after last VLM row before LLM
        if idx + 1 < len(rows):
            next_r = rows[idx + 1]
            if (next_r["family"] == family and next_r["size"] == size
                    and next_r["modality"] == "LLM" and modality == "VLM"):
                lines.append(r"        \addlinespace")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    }")
    lines.append(r"    \label{table:results_" + detector_key + "}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────

def make_tables(sc_path, mo_path, cr_path):
    sc = load_csv(sc_path)
    mo = load_csv(mo_path)
    cr = load_csv(cr_path)

    rows_by_detector, detectors_present = build_rows_by_detector(sc, mo, cr)

    ordered_detectors = sorted(detectors_present, key=lambda d: DETECTOR_ORDER.get(d, 999))

    tables = []
    for detector_key in ordered_detectors:
        table_latex = render_table(rows_by_detector.get(detector_key, []), detector_key)
        if table_latex:
            tables.append(table_latex)
        else:
            print(f"Warning: no rows found for detector '{detector_key}', skipping its table.",
                  file=sys.stderr)

    print(f"Detectors found: {ordered_detectors}", file=sys.stderr)
    return "\n\n".join(tables)


if __name__ == "__main__":
    import os
    if len(sys.argv) == 4:
        scattered_path = sys.argv[1]
        moderate_path = sys.argv[2]
        crowded_path = sys.argv[3]
    else:
        base = os.path.dirname(os.path.abspath(__file__))
        scattered_path = os.path.join(base, "results/scattered.csv")
        moderate_path = os.path.join(base, "results/moderate.csv")
        crowded_path = os.path.join(base, "results/crowded.csv")

    table = make_tables(scattered_path, moderate_path, crowded_path)
    print(table)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "segmented_table.tex")
    with open(out, "w") as f:
        f.write(table)
    print(f"\n[saved to {out}]", file=sys.stderr)
