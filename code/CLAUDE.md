# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Pipeline for using LLMs/VLMs (via [DSPy](https://dspy.ai/)) to detect social groups of people in crowd scenes
(from bounding-box/3D-position detections or raw images), then evaluate the predicted groupings and
downstream group-activity recognition against ground truth (JRDB-Act protocol). Also contains DSPy-free
SFT/GRPO fine-tuning of Qwen2.5(-VL)-7B for the same task.

Layout: all code lives in `code/` (run scripts with `code/` as the working directory); outputs go to the
sibling `../results/` (`predictions/`, `groups/`, `detection_files/`, `grouping_files/`). Datasets live
outside the repo and are referenced by absolute paths (`/lp-dev/jmurrugarral/{gold_SEKAI_900_3,EgoGroups_train,EgoGroups_test,EVP,...}`,
`/home/jmurrugarral/EgoGroups_Synth_train`) or relative ones (`../../<dataset>`). Fine-tuned checkpoints go to
`sft_output/`.

## Environment / running things

- Conda env `py10-dspy` (Python 3.10); `pip install -r ../requirements.txt` plus `conda install -c conda-forge ffmpeg`.
  Training deps (`torch`, `transformers`, `peft`, `trl`, `accelerate`, `datasets`, `scikit-learn`) are already in that
  env; check before installing anything.
- No test suite, linter, or CI. Verify changes by running the relevant script on a single frame/scenario.
- Model access is normally a local vLLM OpenAI-compatible server (`vllm_commands.txt` has example invocations). The
  served model name must equal the `model` argument passed to the scripts:
  ```bash
  vllm serve Qwen/Qwen2.5-VL-72B-Instruct --api-key testkey --tensor-parallel-size 4
  ```
- Single frame / folder of scenario JSONs:
  ```bash
  python3 fetch_groups.py <scenario.json> <single|full> <llm|vlm_text|vlm_image> <model> <frame_id> \
      --prompt_method p1 --depth_method detany_3D --api_base http://localhost:8000/v1 --api_key testkey [--save_image]
  python3 batch_fetch_groups.py <folder_of_jsons>/ single vlm_image <model> <frame_id> --prompt_method p1
  ```
  The frame image is expected at `{frame_path}/{scenario_name}/{frame_id:05d}.jpeg`.
- AP eval: `cd AP_evaluator && python JRDB_eval.py -g gt.txt -d det.txt -t task_2 --labelmap label_map/task_2.pbtxt`
- F1 eval: `cd F1_evaluator && python ours_jrdb_evaluation.py --det pred.pkl --gt gt.pkl --o out`
- Browse results: `python app.py -d <predictions_dir>` (Flask, port 8888, `templates/index.html`).
- Training: see `README_training.md` (authoritative for flags, checkpoint naming, concurrency). Key points:
  run SFT before RL; give concurrent runs distinct `MASTER_PORT`s (every script inits `torch.distributed` on
  29500); `--gpu` sets `CUDA_VISIBLE_DEVICES` before torch import.
- Git on this shared mount may fail with "dubious ownership"; use `git -c safe.directory=$PWD ...` rather than
  changing global config without asking.

## Pipeline stages

1. **Grouping inference** — per-frame person detections (bbox + optional 3D position) → LLM/VLM partitions
   people into groups.
   - `fetch_groups.py` / `batch_fetch_groups.py` (one file / one folder), `fetch_groups_multi.py` (several
     `--frame_ids`, one merged JSON).
   - Drivers: `scripts/batch_fetch_groups_multi.py`, `scripts/batch_fetch_groups_methods_single.bash`,
     `scripts/run_groups*.bash`, `run_batch_groups.bash <model> <port> <api_key> <prompt_method>`.
   - `gemini_dspy.py` / `batch_gemini_dspy.py` — Gemini via a custom `GeminiVertexLM(dspy.LM)` (Vertex AI
     `google-genai`) instead of the OpenAI-compatible `dspy.LM`.
   - `naive_cluster.py` — non-LLM DBSCAN baseline (2D distance by default) writing the same prediction shape as
     `save_frame`, so it flows through the same converters/evaluators.
2. **Downstream group-level tasks** (activity, clothing, hand-holding, hugging): `inference_tasks_multi.py`,
   `scripts/batch_inference_tasks_multi.py`; `inference_coarse.py` / `gemini_inference_coarse.py` shard work via
   `--num_parts`/`--part_id` (`run_parallel.bash`), checkpointing every 25 items. `shared_files/` holds the
   cultural-inference and activity-vocabulary standardization code.
3. **Convert to evaluator formats**
   - `compute_detections.py` → JRDB-style `det_*.txt` for the AP evaluator. `--output_mode coarse` = one merged
     bbox per group; `finegrained` = one row per person tagged with group id (ungrouped people → singletons).
   - `compute_groupings.py` → pickled `{scenario_idx: {frame: [[person_ids...], ...]}}` for the F1 evaluator.
   - `generate_ids_for_vlm_only.py` — for `baseline1`/`baseline2` (which predict raw group boxes, not person-id
     groups): Hungarian-match predicted boxes to GT boxes (IoU ≥ 0.5) to recover a person-id grouping pkl.
   - `scripts/generate_det_files.bash`, `scripts/generate_group_files.bash` sweep all
     dataset/model/prompt/mode/depth combinations through the two converters.
4. **Evaluation**
   - `AP_evaluator/` — AP eval forked from the AVA action-detection script. `AP_evaluator/README.md` documents the
     9-column det/gt `.txt` format and tasks 1–5; read it before touching `compute_detections.py`, `JRDB_eval.py`,
     or `GT_conversion_script.py` (which converts raw JRDB annotations to GT and also produced `F1_evaluator/out/gt.pkl`).
     `evaluate()` parses each GT/det file once and filters scenes in memory (`select_scenes`) — don't reintroduce
     per-scene re-reads. `-o/--output`: `all`/`scattered`/`moderate`/`crowded` (density = `scene_index % 3`), a
     GLOBE region code (`AF`,`AN`,`CA`,`EU`,`GE`,`LA`,`LE`,`ME`,`NE`,`SA`,`O`), or `<REGION>_<density>`.
     `evaluation.py` batch-runs this into `results/<output>.csv`; for a bare region code it averages the three
     density sub-evaluations with equal weight (NaN → 0) rather than pooling. `generate_*.py` render LaTeX tables.
   - `F1_evaluator/` — graph-based clustering/F1 eval (`graph.py`, `utils_eval.py`); `batch_jrdb_evaluation.py`
     sweeps a dataset; `text_to_latex.py` renders tables.
5. **Fine-tuning** (no DSPy; `trl.SFTTrainer` then `trl.GRPOTrainer`) — details in `README_training.md`.
   - `sft_training_{llm,vlm}_nodspy_merged.py`, `rl_training_{llm,vlm}_nodspy_merged.py`. LLM scripts take
     `--mode single|full` (`full` adds a per-person `movement_direction`); VLM scripts take
     `--variant p1_visual|idsonly`. GRPO reward = sum of `pairwise_f1_reward`, `affinity_bce_reward`, `ari_reward`,
     `partition_validity_reward` (unparseable output → fixed `PARSE_FAILURE_REWARD`). RL starts from the merged SFT
     checkpoint plus a fresh LoRA config (GRPOTrainer rejects a `PeftModel` + new `peft_config`).
   - `max_completion_length` must exceed the largest output (frames can have 70+ people); too small a cap
     silently truncates every completion to invalid JSON and the reward collapses to a constant with no error.
   - Data loaders (`--dataset`), all returning the same example schema: `sft_data_utils.py` (JRDB; matches
     `gt.pkl` frames to `JRDB_fixed_gold` shards by track-id-set matching), `egogroups_data_utils.py`
     (`gold_SEKAI_900_3`; `scenario_idx = clip_number - 1`; GT in `F1_evaluator/out/gt_gold_sekai_{2,22,42}.pkl`),
     `egogroups_train_data_utils.py` (`EgoGroups_train`; `scenario_idx + 901` = base clip, 5 ten-frame sub-clips,
     target = frame 10, reads `jsons_step4`), `egogroups_synth_train_data_utils.py` (`EgoGroups_Synth_train`,
     `gt_3D` only). `gt_sekai_*.pkl` (no `_gold_`) belong to a different dataset (`SEKAI_540_3`) — don't mix them up.
     If a loader suddenly returns only a few examples, check the GT pkls haven't regressed to tiny placeholder
     stubs and that detection track_ids still align with GT (see `fix_egogroups_track_ids.py`) before debugging
     the matching logic.
   - `mingle_sft_training.py` + `mingle_train_data_utils.py` — separate MINGLE-style pairwise ("are these two
     grouped?" → `Yes`/`No`) VLM SFT on EgoGroups_train with RGB+depth crops; multi-GPU via `accelerate launch`.
   - `debug_grpo_inference.py` / `debug_vlm_generate.py` — load a merged checkpoint with plain `transformers` and
     print raw generations, to tell bad completions apart from parsing bugs.
   - Test a checkpoint: `vllm serve sft_output/<merged_dir> --served-model-name X --api-key testkey --port 8000`,
     then `batch_fetch_groups.py ... local/X ...` (the model arg is only a results-path label but must contain `/`).
6. **Notebooks** — `inference.ipynb` (in-process vLLM via a custom `VLLMOfflineLM(dspy.LM)` that calls
   `vllm.LLM.chat()` directly, output to `demo_predictions/`), `inference_sft.ipynb` / `inference_sft_cot.ipynb`
   (drive a nodspy checkpoint through DSPy), `analysis/cultural_coarse.ipynb`.

One-off data-prep scripts (already run; not part of the normal pipeline): `fix_egogroups_track_ids.py`,
`combine_egogroups_videos_frames.py`, `restructure_gold_sekai_depth.py`.

## Core shared modules

- `utils.py` — imported wholesale (`from utils import *`) by nearly everything: arg parsers (`parse_args`,
  `parse_args_allframes`, `parse_args_inference_tasks`), DSPy module selection (`get_dspy_cot` / `get_full_dspy_cot`
  keyed by `mode` × `prompt_method`), input building (`get_frame_bboxes`, `get_allframes_bboxes`,
  `get_movement_direction`), inference (`inference*`, `full_inference*`), saving (`save_frame`, `save_full_frame`,
  `save_json`), `draw_bboxes_with_ids`, `compute_group_bbox`.
- `prompts.py` / `ablation_prompts.py` — all DSPy Signatures/Modules, both imported into `utils.py`.
  `IdentifyGroups*` = text-only LLM; `vlm_IdentifyGroups*Image` / `*Text` = VLM. Suffixes: `_Direction`,
  `_Transitive`, `_DirectionTransitive`, `2D` (pixel coords), `_withbboxes`, `_idsonly`, `_AllFrames` (full setting).
  `RecognizeGroup{Activity,Clothing,Handholding,Hugging}` are the downstream-task signatures.
- `utils2.py` and `prompt2.py` are stale copies that nothing imports — edit `utils.py` / `prompts.py`.

## Key conventions

- **`mode`**: `llm` (text-only), `vlm_text` (VLM, text input), `vlm_image` (VLM + frame image).
- **`setting`**: `single` = target frame's detections only. `full` = target frame's detections enriched with a
  per-person `movement_direction` (8-way label relative to the data's x/y axes, or `stationary`, from earliest
  pre-target position), plus for `vlm_image` a `video` of frames 1..target (every 3rd frame when target is 22 or 42).
  For `p1`/`p1_bbox` the video frames are raw; for `p1_visual`/`p1_visual_only` every frame is annotated with the
  *target* frame's boxes (an approximation).
  **Known bug:** `vlm_IdentifyGroups_AllFramesText` still declares nested `all_frames: list[list[dict]]` and a
  required `target_frame`, but `full_inference` calls it with a flat `all_frames=` and no `target_frame`, so
  `full` + `vlm_text` is broken until the signature is updated.
- **`prompt_method`**: `p1` (3D), `p2` (+direction), `p3` (+transitive), `p4` (both), `p5`/`2D` (pixel centroids),
  `p1_bbox` (bare image + separate `boundingboxes` field), `p1_visual` (boxes/ids drawn on image + xyz),
  `p1_visual_only` (drawn image + bare `person_ids`, no coordinates), `p1_depthanything` (reuses the `p1`
  signatures), `baseline1`/`baseline2` (predict raw group boxes; `vlm_image` only). `p2`–`p4` are single-setting
  only; `p1_visual*` are `vlm_image` only.
- **`depth_method`**: `naive_3D_{60,110,160}FOV`, `unidepth_3D`, `detany_3D`, `wilddet_3D` (also `gt_3D` for
  synthetic data). It is forced to `'3D'` when `data['dataset'] == 'JRDB_gold'`.
- **Results path layout** is hard-coded as path joins in each stage, roughly
  `results/{predictions|groups|detection_files|grouping_files}/<dataset>/results[_full]/<model>/<vlm_mode>/<depth_method>/<prompt_method>/<frame_id>/<scenario>/`.
  `fetch_groups*`, `compute_detections.py`, and `compute_groupings.py` each build slightly different variants that
  must stay consistent for the stages to chain. Model names are split on `/` (`model.split("/")[1]`), so any
  `model` argument must contain a `/`.
- Bounding boxes from `Qwen3`/`Cosmos` models are on a 0–1000 scale and get rescaled by image `(W, H)` (see the
  `'Qwen3' in model or 'Cosmos' in model` branches in `utils.py` and `compute_detections.py`).
- A checkpoint trained outside DSPy must be run through a custom `dspy.Adapter` that reproduces the training prompt
  and parse format exactly (see `inference_sft.ipynb`). Stock `ChatAdapter`/`JSONAdapter` won't match.
- `lm.history[-1]` is a plain dict — use `json.dumps(..., default=str)`; `lm.inspect_history()` only prints.
- `transformers` 4.57.2 (in `py10-dspy`) has a bug when loading tokenizers from checkpoints whose `config.json`
  records `"transformers_version" <= 4.57.2`: `AttributeError: 'dict' object has no attribute 'model_type'`,
  including inside vLLM/GRPOTrainer. Fix it by upgrading `transformers` or bumping that field in the checkpoint's
  `config.json`. Don't work around it with `use_fast=False`, because that breaks `assistant_only_loss` masking in
  the SFT scripts.
