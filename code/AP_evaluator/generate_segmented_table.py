#!/usr/bin/env python3
"""
Generate a LaTeX table from three CSV files (scattered.txt, moderate.txt, crowded.txt).
Each file contains: name, G1, G2, G3, G4, G5, AP
Row type is determined by filename suffix:
  - _p1       -> M (metadata)
  - _baseline1 -> A (all frames)
  - _baseline2 -> S (single frame)
"""

import csv
import re
import sys
from collections import defaultdict

# ── helpers ──────────────────────────────────────────────────────────────────

def load_csv(path):
    """Return dict: name -> {G1..G5, AP}"""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["name"]] = {k: row[k] for k in ("G1","G2","G3","G4","G5","AP")}
    return data

def row_type(name):
    if "_p1" in name:
        return "M"
    elif "_baseline2" in name:
        return "S"
    elif "_baseline1" in name:
        return "A"
    return None

def parse_model_info(name):
    """
    Returns (family, size, modality)
    family: Cosmos | Qwen2.5 | Qwen3 | Gemini
    size  : 2B, 8B, 3B, 7B, 32B, 72B, 4B, 30B, 3-Pro …
    modality: LLM | VLM
    """
    stem = name.replace("results_","").replace(".txt","")

    # modality
    if "_vlm_" in stem:
        modality = "VLM"
    else:
        modality = "LLM"

    # model name part
    model_part = stem.split("_llm_")[0].split("_vlm_")[0]

    # family + size
    if model_part.startswith("Cosmos-Reason2-"):
        family = "Cosmos"
        size = model_part.replace("Cosmos-Reason2-","")   # 2B / 8B
    elif model_part.startswith("Qwen2.5-VL-"):
        family = "Qwen2.5"
        size_str = model_part.replace("Qwen2.5-VL-","").replace("-Instruct","")
        size = size_str   # 3B-Instruct → 3B after strip
    elif model_part.startswith("Qwen2.5-"):
        family = "Qwen2.5"
        size_str = model_part.replace("Qwen2.5-","").replace("-Instruct","")
        size = size_str
    elif model_part.startswith("Qwen3-VL-"):
        family = "Qwen3"
        size_str = model_part.replace("Qwen3-VL-","").replace("-Instruct","").replace("-Instruct-2507","")
        size = size_str
    elif model_part.startswith("Qwen3-"):
        family = "Qwen3"
        size_str = model_part.replace("Qwen3-","").replace("-Instruct","").replace("-Instruct-2507","")
        size = size_str
    elif model_part == "gemini":
        family = "Gemini"
        size = "3-Pro"
    else:
        family = model_part
        size = "?"

    # normalise sizes like "30B-A3B" → "30B"
    size = re.sub(r"-A\d+B$","", size)

    return family, size, modality


# canonical size ordering per family
SIZE_ORDER = {
    "Cosmos":  ["2B","8B"],
    "Qwen2.5": ["3B","7B","32B","72B"],
    "Qwen3":   ["4B","30B"],
    "Gemini":  ["3-Pro"],
}
FAMILY_ORDER = ["Cosmos","Qwen2.5","Qwen3","Gemini"]


def fmt(val, bold=False):
    try:
        f = float(val)
        s = f"{f:.2f}"
    except:
        s = str(val)
    return r"\textbf{" + s + "}" if bold else s


# ── find column-wise maxima for bolding ──────────────────────────────────────

def find_maxima(all_rows):
    """
    all_rows: list of dicts with keys sc, mo, cr each containing G1..AP as floats,
              plus 'row_type' (M/A/S).
    Returns per-split per-col maximum (only among M rows, matching paper).
    """
    cols = ["G1","G2","G3","G4","G5","AP"]
    maxima = {split: {c: -1 for c in cols} for split in ("sc","mo","cr")}
    for row in all_rows:
        if row["row_type"] == "M":
            for split in ("sc","mo","cr"):
                for c in cols:
                    try:
                        v = float(row[split][c])
                        if v > maxima[split][c]:
                            maxima[split][c] = v
                    except:
                        pass
    return maxima


# ── build structured data ────────────────────────────────────────────────────

def build_rows(sc_data, mo_data, cr_data):
    """
    Collects one entry per (family, size, modality, row_type).
    Returns list ordered as in the paper.
    """
    # index all names from any file
    all_names = set(sc_data) | set(mo_data) | set(cr_data)

    # group: family -> size -> modality -> row_type -> row
    grouped = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for name in all_names:
        rt = row_type(name)
        if rt is None:
            continue
        family, size, modality = parse_model_info(name)
        sc = sc_data.get(name, {c:"--" for c in ("G1","G2","G3","G4","G5","AP")})
        mo = mo_data.get(name, {c:"--" for c in ("G1","G2","G3","G4","G5","AP")})
        cr = cr_data.get(name, {c:"--" for c in ("G1","G2","G3","G4","G5","AP")})
        grouped[family][size][modality][rt] = {"sc":sc,"mo":mo,"cr":cr,"row_type":rt,
                                                "family":family,"size":size,"modality":modality}

    # flatten in canonical order
    rows = []
    for family in FAMILY_ORDER:
        sizes = SIZE_ORDER.get(family, sorted(grouped.get(family,{}).keys()))
        for size in sizes:
            if size not in grouped.get(family,{}):
                continue
            mod_group = grouped[family][size]
            # VLM rows first (S, A, M), then LLM (M only)
            for modality in ["VLM","LLM"]:
                if modality not in mod_group:
                    continue
                rt_group = mod_group[modality]
                for rt in ["S","A","M"]:
                    if rt in rt_group:
                        rows.append(rt_group[rt])
    return rows


# ── LaTeX generation ─────────────────────────────────────────────────────────

def make_table(sc_path, mo_path, cr_path):
    sc = load_csv(sc_path)
    mo = load_csv(mo_path)
    cr = load_csv(cr_path)

    rows = build_rows(sc, mo, cr)
    maxima = find_maxima(rows)
    cols = ["G1","G2","G3","G4","G5","AP"]

    lines = []
    lines.append(r"\begin{table*}[h!]")
    lines.append(r"    \centering")
    lines.append(r"    \footnotesize")
    lines.append(r"    \caption{Fine-grained performance comparison on dataset for whole video and single frame evaluation. "
                 r"Model combination indicates the backbone family, number of parameters, and modality. "
                 r"Method indicates the processing approach. $G_1$--$G_5$ represent AP scores for group sizes 1--5$^+$, "
                 r"with AP showing the average performance across all groups. "
                 r"Our approach uses metadata (+metadata) for zero-shot evaluation.}")
    lines.append(r"    \vspace{0.3cm}")
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
    prev_size   = None
    prev_mod    = None

    # pre-compute span counts
    from collections import Counter
    family_counts = Counter()
    size_counts   = Counter()
    mod_counts    = Counter()
    for r in rows:
        family_counts[r["family"]] += 1
        size_counts[(r["family"],r["size"])] += 1
        mod_counts[(r["family"],r["size"],r["modality"])] += 1

    for r in rows:
        family   = r["family"]
        size     = r["size"]
        modality = r["modality"]
        rt       = r["row_type"]

        # family cell
        if family != prev_family:
            fc = family_counts[family]
            family_cell = r"\multirow{" + str(fc) + r"}{\*}{\rotatebox{90}{" + family + r"}}"
            prev_family = family
            prev_size = None
            prev_mod  = None
            # add midrule between families (not before first)
            if lines[-1] != r"    \midrule":
                lines.append(r"         \midrule")
        else:
            family_cell = ""

        # size cell
        if size != prev_size or family != rows[rows.index(r)-1]["family"] if rows.index(r)>0 else True:
            sc2 = size_counts[(family,size)]
            size_cell = r"\multirow{" + str(sc2) + r"}{*}{" + size + r"}"
            if prev_size is not None and prev_size != size:
                lines.append(r"        \cmidrule(lr){3-22}")
            prev_size = size
            prev_mod  = None
        else:
            size_cell = ""

        # modality cell
        if modality != prev_mod:
            mc = mod_counts[(family,size,modality)]
            mod_cell = r"\multirow{" + str(mc) + r"}{*}{" + modality + r"}"
            prev_mod = modality
        else:
            mod_cell = ""

        # data cells
        def cells(split_data, split_key):
            parts = []
            for c in cols:
                v = split_data.get(c,"--")
                try:
                    is_bold = abs(float(v) - maxima[split_key][c]) < 0.005
                except:
                    is_bold = False
                parts.append(fmt(v, is_bold))
            return " & ".join(parts)

        sc_cells = cells(r["sc"], "sc")
        mo_cells = cells(r["mo"], "mo")
        cr_cells = cells(r["cr"], "cr")

        line = (f"         & {size_cell} & {mod_cell} & {rt}"
                f" & {sc_cells} & {mo_cells} & {cr_cells} \\\\")
        # prepend family cell
        line = f"        {family_cell}" + line[8:]

        lines.append(line)

        # addlinespace after last VLM row before LLM
        idx = rows.index(r)
        if idx + 1 < len(rows):
            next_r = rows[idx+1]
            if (next_r["family"] == family and next_r["size"] == size
                    and next_r["modality"] == "LLM" and modality == "VLM"):
                lines.append(r"        \addlinespace")

    lines.append(r"    \bottomrule")
    lines.append(r"    \end{tabular}")
    lines.append(r"    }")
    lines.append(r"    \label{table:results_merged}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    if len(sys.argv) == 4:
        scattered_path = sys.argv[1]
        moderate_path  = sys.argv[2]
        crowded_path   = sys.argv[3]
    else:
        # default: look next to this script
        base = os.path.dirname(os.path.abspath(__file__))
        scattered_path = os.path.join(base, "scattered.txt")
        moderate_path  = os.path.join(base, "moderate.txt")
        crowded_path   = os.path.join(base, "crowded.txt")

    table = make_table(scattered_path, moderate_path, crowded_path)
    print(table)

    # also write to file
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_table.tex")
    with open(out, "w") as f:
        f.write(table)
    print(f"\n[saved to {out}]", file=sys.stderr)

