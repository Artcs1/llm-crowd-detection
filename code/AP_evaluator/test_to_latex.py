import csv
import sys
import re

def extract_model_info(name):
    """
    Extract model information from filename.
    Returns: (model_family, params, modality, method)
    """
    # Remove common prefixes and suffix
    name = name.replace('results_full_', '').replace('results_', '')
    name = name.replace('.txt', '').replace('.csv', '')
    
    # Determine modality
    if 'vlm_image' in name.lower():
        modality = 'VLM'
    else:
        modality = 'LLM'
    
    # Determine method
    if '_p1' in name:
        method = '+ metadata'
    elif 'baseline2' in name:
        method = 'Sequentially'
    elif 'baseline1' in name:
        method = 'Atomic'
    else:
        method = 'Other'
    
    # Extract model family and parameters
    # Look for patterns like Qwen2.5, Qwen3
    #model_match = re.search(r'(Qwen[\d.]+)', name)
    model_match = re.search(r'(Qwen[\d.]+|Cosmos-Reason\d+|gemini-\d)', name)
    model_family = model_match.group(1) if model_match else 'Unknown'
    #print(model_family)
    
    # Extract parameter size (e.g., 3B, 7B, 32B, 72B, 30B-A3B, 4B)
    param_match = re.search(r'(\d+B)', name)
    if param_match:
        params = param_match.group(1)
    else:
        # Handle special cases like 30B-A3B
        param_match = re.search(r'(\d+B-A\d+B)', name)
        params = param_match.group(1) if param_match else 'Unknown'
    
    return model_family, params, modality, method


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


def create_merged_latex_table(single_frame_data, video_data):
    """
    Create a merged LaTeX table with both Single Frame and Video columns.
    
    Parameters:
    - single_frame_data: data from results_ files
    - video_data: data from results_full_ files
    """
    
    if not single_frame_data or len(single_frame_data) <= 1:
        return ""
    if not video_data or len(video_data) <= 1:
        return ""
    
    # Organize both datasets
    def organize_data(data):
        organized = {}
        for row in data[1:]:  # Skip header
            if len(row) < 2:
                continue
            
            name = row[0]
            model_family, params, modality, method = extract_model_info(name)
            
            key = (model_family, params, modality, method)
            organized[key] = row[1:]  # Store values
        
        return organized
    
    single_frame_org = organize_data(single_frame_data)
    video_org = organize_data(video_data)
    
    # Get all unique keys (model configurations)
    all_keys = set(single_frame_org.keys()) | set(video_org.keys())
    
    # Organize by hierarchy for display
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
    
    # Sort parameter sizes
    param_order = {
            '2B':1, '3B': 2, '4B': 3, '7B': 4, '8B': 5, '30B-A3B': 6, '32B': 7, '72B': 8, '235B-A22B':9
    }
    
    # Method order: Sequentially, Atomic, + metadata
    method_order = {
        'Sequentially': 1,
        'Atomic': 2,
        '+ metadata': 3,
        'Other': 4
    }
    
    # Calculate max values for Single Frame (columns 0-5) and Video (columns 6-11)
    all_sf_values = []
    all_video_values = []
    
    for key in all_keys:
        if key in single_frame_org:
            all_sf_values.append(single_frame_org[key])
        if key in video_org:
            all_video_values.append(video_org[key])
    
    max_sf_values = find_max_values(all_sf_values, 0, 6)
    max_video_values = find_max_values(all_video_values, 0, 6)
    
    # Generate LaTeX table
    latex_output = []
    
    latex_output.append("\\begin{table*}[h!]")
    latex_output.append("    \\centering")
    latex_output.append("    \\footnotesize")
    latex_output.append("    \\caption{Fine-grained performance comparison on dataset for whole video and single frame evaluation. Model combination indicates the backbone family, number of parameters, and modality. Method indicates the processing approach. $G_1$--$G_5$ represent AP scores for group sizes 1--5$^+$, with AP showing the average performance across all groups. Our approach uses metadata (+metadata) for zero-shot evaluation.}")
    latex_output.append("    \\vspace{-0.3cm}")
    latex_output.append("    \\resizebox{0.9\\linewidth}{!}{")
    latex_output.append("    \\begin{tabular}{llllcccccccccccc}")
    latex_output.append("    \\toprule")
    
    # Column headers
    latex_output.append("        & \\multirow{2}{*}{Model} &  & \\multirow{2}{*}{Method}")
    latex_output.append("        & \\multicolumn{6}{c}{\\textbf{Single Frame}}")
    latex_output.append("        & \\multicolumn{6}{c}{\\textbf{Video}} \\\\")
    latex_output.append("        \\cmidrule(lr){5-10} \\cmidrule(lr){11-16}")
    latex_output.append("        & & & & G$_{1}$ & G$_{2}$ & G$_{3}$ & G$_{4}$ & G$_{5}$ & AP & G$_{1}$ & G$_{2}$ & G$_{3}$ & G$_{4}$ & G$_{5}$ & AP \\\\")
    latex_output.append("    \\midrule")
    
    # Data rows
    for model_family in sorted(organized_data.keys()):
        model_params = organized_data[model_family]
        
        # Count total rows for this model family
        total_rows = 0
        for params in model_params:
            for modality in ['VLM', 'LLM']:
                total_rows += len(model_params[params][modality])
        
        first_row = True
        
        for params in sorted(model_params.keys(), key=lambda x: param_order.get(x, 999)):
            param_data = model_params[params]
            
            # Count rows for this parameter size
            param_rows = len(param_data['VLM']) + len(param_data['LLM'])
            
            first_param_row = True
            
            for modality in ['VLM', 'LLM']:
                if not param_data[modality]:
                    continue
                
                # Sort methods: Sequentially, Atomic, + metadata
                sorted_entries = sorted(param_data[modality], 
                                       key=lambda x: method_order.get(x['method'], 999))
                
                modality_rows = len(sorted_entries)
                first_modality_row = True
                
                for entry in sorted_entries:
                    row_parts = []
                    
                    # Model family (rotated, spans all rows)
                    if first_row:
                        model_display = model_family.replace('.', '.').replace('_', '\\_')
                        row_parts.append(f"        \\multirow{{{total_rows}}}{{*}}{{\\rotatebox{{90}}{{{model_display}}}}}")
                        first_row = False
                    else:
                        row_parts.append("        ")
                    
                    # Parameter size (spans VLM + LLM rows)
                    if first_param_row:
                        row_parts.append(f"& \\multirow{{{param_rows}}}{{*}}{{{params}}}")
                        first_param_row = False
                    else:
                        row_parts.append("& ")
                    
                    # Modality (spans methods for this modality)
                    if first_modality_row:
                        row_parts.append(f"& \\multirow{{{modality_rows}}}{{*}}{{{modality}}}")
                        first_modality_row = False
                    else:
                        row_parts.append("& ")
                    
                    # Method
                    row_parts.append(f"& {entry['method']}")
                    
                    # Single Frame values with bold for max
                    for i, val in enumerate(entry['single_frame_values']):
                        formatted_val = format_value(val, max_sf_values[i]) if val else '-'
                        row_parts.append(f"& {formatted_val}")
                    
                    # Video values with bold for max
                    for i, val in enumerate(entry['video_values']):
                        formatted_val = format_value(val, max_video_values[i]) if val else '-'
                        row_parts.append(f"& {formatted_val}")
                    
                    latex_output.append(" ".join(row_parts) + " \\\\")
                
                # Add space after VLM before LLM
                if modality == 'VLM' and param_data['LLM']:
                    latex_output.append("        \\addlinespace")
            
            # Add rule between different parameter sizes
            if params != sorted(model_params.keys(), key=lambda x: param_order.get(x, 999))[-1]:
                latex_output.append("        \\cmidrule(lr){3-16}")
    
    latex_output.append("    \\bottomrule")
    latex_output.append("    \\end{tabular}")
    latex_output.append("    }")
    latex_output.append("    \\label{table:results_merged}")
    latex_output.append("\\end{table*}")
    
    return "\n".join(latex_output)


def csv_to_latex(input_file, output_file='table.tex'):
    """
    Convert a CSV file to a merged LaTeX table.
    
    Parameters:
    - input_file: path to the CSV file
    - output_file: path to save the LaTeX output
    """
    
    # Read the CSV file
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
    
    # Separate data: results_full_ -> Video, results_ -> Single Frame
    header = data[0]
    video_data = [header]  # results_full_
    single_frame_data = [header]  # results_
    
    for row in data[1:]:
        if len(row) > 0:
            name = row[0]
            if name.startswith('results_full_'):
                video_data.append(row)
            elif name.startswith('results_'):
                single_frame_data.append(row)
    
    # Generate merged LaTeX table
    latex_code = create_merged_latex_table(single_frame_data, video_data)
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            f.write(latex_code)
        #print(f"LaTeX table successfully saved to '{output_file}'")
        #print(f"Single Frame rows: {len(single_frame_data)-1}")
        #print(f"Video rows: {len(video_data)-1}")
        #print("\nPreview:")
        #print(latex_code)
    except Exception as e:
        print(f"Error writing output file: {e}")


# Main execution
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
