llm-crowd-detection pipeline: inputs/outputs

Flow: inference_coarse.py (individual/split LLM inference) → merge_inference_results.py (combine splits) → preprocess_cultural.py (build analysis-ready DataFrame).

1. inference_coarse.py
Runs an LLM per group to predict activity, clothing, handholding, and hugging attributes.

Run: python inference_coarse.py <dir> <model> [--api_base] [--api_key] [--temperature] [--max_tokens] [--frame_path] [--num_parts N] [--part_id K]
Input: <dir>/all_annotations.json (group bounding boxes per video frame) + video frames referenced inside it.
Output: <dir>/results_cultural/<model_name>/annotations.json — or annotations_<part_id>_of_<num_parts>.json when sharded. Saved incrementally every 25 annotations.


2. merge_inference_results.py
Stitches the per-shard results back into one combined annotation set.

Run: python merge_inference_results.py <base_dir> <model> <num_parts>
Input: <base_dir>/all_annotations.json (for total count/template) plus the sharded part files <base_dir>/results_cultural/<model>/annotations_{1..num_parts}_of_{num_parts}.json.
Output: <base_dir>/results_cultural/<model>/annotations.json (merged, single file).


3. preprocess_cultural.py
Aggregates per-frame group-level predictions into a flat per-frame row (activity/clothing lists, handholding/hugging counts + binaries, location/region mapping via country_to_region_globe).

Run: python preprocess_cultural.py <data_dir> <model_name>
Input: <data_dir>/results_cultural/<model_name>/annotations.json (merged LLM outputs) plus per-video metadata from <data_dir>/jsons_step1/<videoName>.json (dataset, density, city, country, source).
Output: <data_dir>/results_cultural/<model_name>/pd_annotations.pkl — a pandas DataFrame pickle ready for analysis/plotting.
