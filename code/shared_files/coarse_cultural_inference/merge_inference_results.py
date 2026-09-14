import os
import json
import argparse


# accept a path to the results folder using argparse
parser = argparse.ArgumentParser(description='Merge inference results from multiple files into a single JSON file.')
parser.add_argument('base_dir', type=str, help='Path to the folder containing inference results files.')
parser.add_argument('model', type=str, help='Name of the model.')
parser.add_argument('num_parts', type=int, help='Number of parts to merge.')

args = parser.parse_args()

# load first file to determine total number of annotations
with open(os.path.join(args.base_dir, 'all_annotations.json'), 'r') as f:
    res_json = json.load(f)

total_annotations = len(res_json['annotations'])
annotations_per_part = total_annotations // args.num_parts


files = [f'annotations_{i}_of_{args.num_parts}.json' for i in range(1, args.num_parts + 1)]
for i in range(1, args.num_parts + 1):
    file_path = os.path.join(args.base_dir, 'results_cultural', args.model, f'annotations_{i}_of_{args.num_parts}.json')
    with open(file_path, 'r') as f:
        part_json = json.load(f)

    start_idx = (i - 1) * annotations_per_part
    end_idx = start_idx + annotations_per_part if i < args.num_parts else total_annotations
    res_json['annotations'][start_idx:end_idx] = part_json['annotations'][start_idx:end_idx]

# save the merged results to a new JSON file
output_path = os.path.join(args.base_dir, 'results_cultural', args.model, 'annotations.json')
with open(output_path, 'w') as f:
    json.dump(res_json, f, indent=4)