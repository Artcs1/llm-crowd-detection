import csv
import sys
import re

# Human-readable labels for each detector / depth source, used in captions & labels
DETECTOR_LABELS = {
    'detany': 'DetAny3D',
    'unidepth': 'UniDepth',
    'wilddet': 'WildDet3D',
}
DETECTOR_ORDER = {'detany': 1, 'unidepth': 2, 'wilddet': 3}


def extract_model_info(name):
    """
    Extract model information from filename.
    Returns: (model_family, params, modality, method, detector)
    """
    # Remove common prefixes and suffix
    name = name.replace('results_full_', '').replace('results_', '')
    name = name.replace('.txt', '').replace('.csv', '')

    # Determine modality
    if 'vlm_image' in name.lower():
        modality = 'VLM'
    else:
        modality = 'LLM'

    # Detect SFT / RL training-stage suffixes (e.g. "Qwen2.5-7B-Instruct-RL")
    # so that these variants show up as their own method row under the same
    # base model family, instead of being tagged as "+ metadata".
    stage_match = re.search(r'-(SFT|RL)(?:_|$)', name, re.IGNORECASE)
    stage = stage_match.group(1).upper() if stage_match else None

    # Determine method
    if stage:
        method = f'+{stage}'
    elif '_p1_bbox' in name:
        method = '+ metadata-bbox'
    elif '_p1' in name:
        method = '+DATA'
    elif 'baseline2' in name:
        method = 'Sequentially'
    elif 'baseline1' in name:
        method = 'Atomic'
    else:
        method = 'Other'

    # Extract model family and parameters (case-insensitive so lowercase
    # filenames like "qwen2.5-7b-groups-grpo" are still captured)
    model_match = re.search(r'(Qwen[\d.]+|Cosmos-Reason\d+|gemini-\d)', name, re.IGNORECASE)
    model_family = model_match.group(1) if model_match else None

    # Extract parameter size (e.g., 3B, 7B, 32B, 72B, 30B-A3B, 4B)
    param_match = re.search(r'(\d+B-A\d+B)', name, re.IGNORECASE)
    if param_match:
        params = param_match.group(1)
    else:
        param_match = re.search(r'(\d+B)', name, re.IGNORECASE)
        params = param_match.group(1) if param_match else None

    # If either family or params couldn't be parsed, fall back to using the
    # full (cleaned) filename as the "family" so unrelated/unparseable
    # filenames never collide with each other or with 'Unknown' entries.
    if model_family is None or params is None:
        model_family = name
        params = 'Unknown'

    # Extract detector / depth-source type (detany / unidepth / wilddet)
    detector_match = re.search(r'_(detany|unidepth|wilddet)(?:_|$)', name, re.IGNORECASE)
    detector = detector_match.group(1).lower() if detector_match else 'unknown'

    return model_family, params, modality, method, detector


def find_max_values(rows, start_col, end_col):
    """Find maximum values for each column in a specific range."""
    num_cols = len(rows[0]) if rows else 0
    max_values = [None] * num_cols

    for col_idx in range(start_col, min(end_col, num_cols)):
        col_values = []
        for row in rows:
            if len(row) > col_idx:
                try:
                    value = float(row[col_idx])
                    col_values.append(value)
                except ValueError:
                    pass
        if col_values:
            max_values[col_idx] = max(col_values)

    return max_values


def format_value(value, max_val):
    """Format a value, bolding if it's the maximum."""
    try:
        val = float(value)
        if max_val is not None and abs(val - max_val) < 1e-9:
            return f"\\textbf{{{value}}}"
        else:
            return value
    except ValueError:
        return value


def create_merged_latex_table(single_frame_data, video_data, detector_key):
    """
    Create a merged LaTeX table with both Single Frame and Video columns,
    scoped to a single detector/depth-source type.
    """

    have_sf = bool(single_frame_data) and len(single_frame_data) > 1
    have_video = bool(video_data) and len(video_data) > 1

    if not have_sf and not have_video:
        return ""

    def organize_data(data):
        organized = {}
        for row in data[1:]:  # Skip header
            if len(row) < 2:
                continue

            name = row[0]
            model_family, params, modality, method, detector = extract_model_info(name)
            if detector != detector_key:
                continue

            key = (model_family, params, modality, method)
            if key in organized:
                print(f"Warning: duplicate key {key} from '{name}' "
                      f"overwrote a previous entry for detector '{detector_key}'.")
            organized[key] = row[1:]  # Store values

        return organized

    single_frame_org = organize_data(single_frame_data) if have_sf else {}
    video_org = organize_data(video_data) if have_video else {}

    all_keys = set(single_frame_org.keys()) | set(video_org.keys())
    if not all_keys:
        return ""

    organized_data = {}
    for key in all_keys:
        model_family, params, modality, method = key

        if model_family not in organized_data:
            organized_data[model_family] = {}
        if params not in organized_data[model_family]:
            organized_data[model_family][params] = {'VLM': [], 'LLM': []}

        organized_data[model_family][params][modality].append({
            'method': method,
            'single_frame_values': single_frame_org.get(key, [''] * 6),
            'video_values': video_org.get(key, [''] * 6)
        })

    param_order = {
        '2B': 1, '3B': 2, '4B': 3, '7B': 4, '8B': 5, '30B-A3B': 6, '32B': 7, '72B': 8, '235B-A22B': 9
    }

    method_order = {
        'Sequentially': 1,
        'Atomic': 2,
        '+DATA': 3,
        '+SFT': 4,
        '+RL': 5,
        '+ metadata-bbox': 6,
        'Other': 7
    }

    all_sf_values = [single_frame_org[k] for k in all_keys if k in single_frame_org]
    all_video_values = [video_org[k] for k in all_keys if k in video_org]

    max_sf_values = find_max_values(all_sf_values, 0, 6) if all_sf_values else [None] * 6
    max_video_values = find_max_values(all_video_values, 0, 6) if all_video_values else [None] * 6

    detector_label = DETECTOR_LABELS.get(detector_key, detector_key)

    latex_output = []

    latex_output.append("\\begin{table*}[h!]")
    latex_output.append("    \\centering")
    latex_output.append("    \\footnotesize")
    latex_output.append(
        f"    \\caption{{Fine-grained performance comparison using \\textbf{{{detector_label}}} as the "
        "3D detection/depth source, for whole video and single frame evaluation. Model combination indicates "
        "the backbone family, number of parameters, and modality. Method indicates the processing approach. "
        "$G_1$--$G_5$ represent AP scores for group sizes 1--5$^+$, with AP showing the average performance "
        "across all groups. Our approach uses metadata (+metadata) for zero-shot evaluation.}"
    )
    latex_output.append("    \\resizebox{0.9\\linewidth}{!}{")
    latex_output.append("    \\begin{tabular}{llllcccccccccccc}")
    latex_output.append("    \\toprule")

    latex_output.append("        & \\multirow{2}{*}{Model} &  & \\multirow{2}{*}{Method}")
    latex_output.append("        & \\multicolumn{6}{c}{\\textbf{Single Frame}}")
    latex_output.append("        & \\multicolumn{6}{c}{\\textbf{Video}} \\\\")
    latex_output.append("        \\cmidrule(lr){5-10} \\cmidrule(lr){11-16}")
    latex_output.append("        & & & & G$_{1}$ & G$_{2}$ & G$_{3}$ & G$_{4}$ & G$_{5}$ & AP & G$_{1}$ & G$_{2}$ & G$_{3}$ & G$_{4}$ & G$_{5}$ & AP \\\\")
    latex_output.append("    \\midrule")

    sorted_param_keys = sorted(organized_data.keys())
    for model_family in sorted_param_keys:
        model_params = organized_data[model_family]

        total_rows = 0
        for params in model_params:
            for modality in ['VLM', 'LLM']:
                total_rows += len(model_params[params][modality])

        first_row = True
        sorted_params_list = sorted(model_params.keys(), key=lambda x: param_order.get(x, 999))

        for params in sorted_params_list:
            param_data = model_params[params]
            param_rows = len(param_data['VLM']) + len(param_data['LLM'])
            first_param_row = True

            for modality in ['VLM', 'LLM']:
                if not param_data[modality]:
                    continue

                sorted_entries = sorted(param_data[modality],
                                         key=lambda x: method_order.get(x['method'], 999))
                modality_rows = len(sorted_entries)
                first_modality_row = True

                for entry in sorted_entries:
                    row_parts = []

                    if first_row:
                        model_display = model_family.replace('_', '\\_')
                        row_parts.append(f"        \\multirow{{{total_rows}}}{{*}}{{\\rotatebox{{10}}{{{model_display}}}}}")
                        first_row = False
                    else:
                        row_parts.append("        ")

                    if first_param_row:
                        row_parts.append(f"& \\multirow{{{param_rows}}}{{*}}{{{params}}}")
                        first_param_row = False
                    else:
                        row_parts.append("& ")

                    if first_modality_row:
                        row_parts.append(f"& \\multirow{{{modality_rows}}}{{*}}{{{modality}}}")
                        first_modality_row = False
                    else:
                        row_parts.append("& ")

                    row_parts.append(f"& {entry['method']}")

                    for i, val in enumerate(entry['single_frame_values']):
                        formatted_val = format_value(val, max_sf_values[i]) if val else '-'
                        row_parts.append(f"& {formatted_val}")

                    for i, val in enumerate(entry['video_values']):
                        formatted_val = format_value(val, max_video_values[i]) if val else '-'
                        row_parts.append(f"& {formatted_val}")

                    latex_output.append(" ".join(row_parts) + " \\\\")

                if modality == 'VLM' and param_data['LLM']:
                    latex_output.append("        \\addlinespace")

            if params != sorted_params_list[-1]:
                latex_output.append("        \\cmidrule(lr){3-16}")

        if model_family != sorted_param_keys[-1]:
            latex_output.append("        \\hline")

    latex_output.append("    \\bottomrule")
    latex_output.append("    \\end{tabular}")
    latex_output.append("    }")
    latex_output.append(f"    \\label{{table:results_{detector_key}}}")
    latex_output.append("\\end{table*}")

    return "\n".join(latex_output)


def csv_to_latex(input_file, output_file='table.tex'):
    try:
        with open(input_file, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    if not data:
        print("Error: CSV file is empty.")
        return

    header = data[0]
    video_data = [header]
    single_frame_data = [header]

    for row in data[1:]:
        if len(row) > 0:
            name = row[0]
            if name.startswith('results_full_'):
                video_data.append(row)
            elif name.startswith('results_'):
                single_frame_data.append(row)

    detectors_present = set()
    for row in data[1:]:
        if len(row) > 0:
            _, _, _, _, detector = extract_model_info(row[0])
            detectors_present.add(detector)

    ordered_detectors = sorted(
        detectors_present,
        key=lambda d: DETECTOR_ORDER.get(d, 999)
    )

    tables = []
    for detector_key in ordered_detectors:
        table_latex = create_merged_latex_table(single_frame_data, video_data, detector_key)
        if table_latex:
            tables.append(table_latex)
        else:
            print(f"Warning: no rows found for detector '{detector_key}', skipping its table.")

    full_latex = "\n\n".join(tables)

    try:
        with open(output_file, 'w') as f:
            f.write(full_latex)
        print(f"LaTeX tables successfully saved to '{output_file}'")
        print(f"Detectors found: {ordered_detectors}")
    except Exception as e:
        print(f"Error writing output file: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else 'table.tex'
    else:
        input_file = input("Enter the CSV/TXT file path: ").strip()
        output_file = input("Enter output file name (default: table.tex): ").strip()
        if not output_file:
            output_file = 'table.tex'

    csv_to_latex(input_file, output_file)
