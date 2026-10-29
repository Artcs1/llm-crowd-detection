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
     `--variant p1_visual|idsonly` and `--mode single|full`. VLM `full` is the single annotated keyframe plus
     a per-person `movement_direction` in the text, not a video. p1_visual: from the 3D track. idsonly:
     `person_id` + `movement_direction` only, from the 2D box-centre motion (`utils.get_movement_direction_2d`,
     example field `movement_2d`), system prompt `SYSTEM_PROMPT_FULL_IDSONLY`, checkpoints `...-full-idsonly-<dataset>`.
     It is tested with `p1_visual_only` full (see Key conventions). GRPO reward = sum of `pairwise_f1_reward`, `affinity_bce_reward`, `ari_reward`,
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
     `egogroups_city_data_utils.py` (`--dataset egogroups-city`; see "EgoGroups splits" below) runs the
     `egogroups_train_data_utils.py` builders over both EgoGroups folders and keeps the city train/val clips;
     its `benchmark=` argument (`--dataset egogroups-benchmark3` -> `'benchmark3'`, via `CITY_LIKE` in the SFT/RL
     scripts) keeps another division's clips instead (`'id_val'` is returned as `city_split == 'val'`).
   - `mingle_sft_training.py` + `mingle_train_data_utils.py` — separate MINGLE-style pairwise ("are these two
     grouped?" → `Yes`/`No`) VLM SFT on EgoGroups_train with RGB+depth crops; multi-GPU via `accelerate launch`.
     `--dataset egogroups-city` uses `mingle_city_data_utils.py`; `--dataset egogroups-benchmark3` uses the same
     module after `use_benchmark('benchmark3')` (TRAINVAL = train + id_val, VAL_SCENARIOS = id_val).
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
  `get_movement_direction`, `get_movement_direction_2d`), inference (`inference*`, `full_inference*`), saving (`save_frame`, `save_full_frame`,
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
  For `p1`/`p1_bbox` the video frames are raw; for `p1_visual`/`p1_visual_only_video` every frame is annotated with
  the *target* frame's boxes (an approximation). Exception: `p1_visual_only` full sends **no video** (below).
  **Known bug:** `vlm_IdentifyGroups_AllFramesText` still declares nested `all_frames: list[list[dict]]` and a
  required `target_frame`, but `full_inference` calls it with a flat `all_frames=` and no `target_frame`, so
  `full` + `vlm_text` is broken until the signature is updated.
- **`prompt_method`**: `p1` (3D), `p2` (+direction), `p3` (+transitive), `p4` (both), `p5`/`2D` (pixel centroids),
  `p1_bbox` (bare image + separate `boundingboxes` field), `p1_visual` (boxes/ids drawn on image + xyz),
  `p1_visual_only` (drawn image, no coordinates), `p1_depthanything` (reuses the `p1`
  signatures), `baseline1`/`baseline2` (predict raw group boxes; `vlm_image` only). `p2`–`p4` are single-setting
  only; `p1_visual*` are `vlm_image` only.
- **`p1_visual_only` (training `--variant idsonly`), the same as training in both settings:**
  - single: annotated keyframe + `person_ids` (`vlm_IdentifyGroupsImage_idsonly`; SFT/RL `--mode single`).
  - full: annotated keyframe + `all_frames` = `[{person_id, movement_direction}]`, with no video and no
    coordinates. `vlm_IdentifyGroups_MotionImage_idsonly`; SFT/RL `--mode full`.
    - Movement (`get_movement_direction_2d`): box centre from the person's earliest pre-target frame to the
      target, `stationary` below 0.2 × target box height (≈ the 3D 0.3 m), same 8-way labels and axes as
      `get_movement_direction`.
  - `p1_visual_only_video` (full, zero-shot only; the former `p1_visual_only` full): annotated keyframe +
    annotated video + bare `person_ids`, no movement (`vlm_IdentifyGroups_AllFramesImage_idsonly`).
    - The existing zero-shot `results_full/.../p1_visual_only` predictions (EgoGroups_test, base models,
      before Oct 29 2026) were made this way and keep that folder name.
    - `--skip_existing` reuses them, so a new full `p1_visual_only` run of those models needs a new folder.
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

## EgoGroups splits

Clips are numbered 1-2700: `EgoGroups_test` holds 1-900, `EgoGroups_train` holds 901-2700 (5 ten-frame sub-clips
each, `clip_XXXX_YYYYY`). Per-clip city / country / `globe_region` / `density` are in `annotations/jsons_step1/clip_XXXX.json`.
Five train/val/test definitions exist; the env var `EGO_VAL_SPLIT` picks one in the baselines' loaders (`geo180`,
`density180`, `city`, `benchmark3`; `benchmark2` has lists, GT and folders only):

- `geo180` (default): train = EgoGroups_train minus the 180 val clips of `egogroups_val_geo180.txt` (6 per region x
  density), test = EgoGroups_test (1-900).
- `density180` (older runs): val = the last 60 clips of each density (1141-1200, 1741-1800, 2641-2700).
- `city` ("egogroups-city"): user-provided division of ALL 2700 clips, `egogroups_city_{train,val,test}.txt`
  (1620 / 180 / 900; `egogroups_city.py` loads them as `CITY`). No city and no source video is shared between
  splits; density is 22/33/44% in each split. Every split mixes clips of both original folders, so test != EgoGroups_test.
  - Folders `<root>/EgoGroupsCity_Train` (train + val, 1800 clips) and `<root>/EgoGroupsCity_Test` (900 clips), same
    layout as the originals (`jsons_step1..6`, `videos_frames`), built by `scripts/make_egogroupscity_folders.py`
    as relative links (how to rebuild them from EgoGroups_train / EgoGroups_test elsewhere: `README_egogroups_city.md`). Depth: `<root>/EVP/depth/gold_egogroupscity_{train,test}`, links into
    `gold_egogroups_{test,train}`. Clip names keep their ORIGINAL numbers, so each folder's clip ids are
    non-contiguous (EgoGroupsCity_Test: 312 clips from 1-900 + 588 from 901-2700; EgoGroupsCity_Train: 588 + 1212).
    Clips 901-2700 have no `jsons_step6`: their `jsons_step6` entries link to `jsons_step4` (same boxes, track ids,
    `detany_3D`); `jsons_step5` exists only for part of those clips.
  - GT with one numbering, `seq = clip_number - 1` for all clips (`python egogroups_city.py gt`):
    `AP_evaluator/out/gt_group_10fps_city_{train,val,test}.txt`, `F1_evaluator/out/gt_group_10fps_city_{train,val,test}.pkl`.
    15 clips of 1-900 have nobody at any keyframe, so they have no txt rows (as in `gt_group_10fps_test.txt`).
    Never mix seq numberings: `gt_group_10fps_test.*` uses `clip - 1`, `gt_group_10fps_train.*` uses `clip - 901`
    (and some of its box values are floats such as `709.0`).
  - Testing ANY checkpoint on it: `EGO_DATA=EgoGroupsCity_Test bash scripts/run_groups.bash ...`, then
    `compute_detections.py` / `compute_groupings.py --dataset EgoGroupsCity_Test` (outputs under
    `results/{detection,grouping}_files/EgoGroupsCity_Test_10`). `JRDB_eval.py` reads density / region from
    `annotations/jsons_step1` when the detection path contains `EgoGroupsCity`, so `--output <density|REGION|REGION_density>`
    works. The baselines' testers take `--split city_test` (RTGD, Social-CAD, S3R2, P-HAR), `runs/EgoGroupsCity_Test`
    (GroupTrans), `test_egogroupscity.sh` (DVT3), `run_mingle.py --dataset egogroupscity_test` (MINGLE).
  - Training: `--dataset egogroups-city` (SFT / RL VLM scripts, MINGLE; checkpoints `-egogroups-city`), `EGO_VAL_SPLIT=city`
    (baselines). Full commands: `<root>/TRAINING_TESTING_SPECS.txt` section 9.
- `benchmark2` (in-distribution / out-of-distribution): user-provided lists `laaf4_train_clip_ids.txt` (1359),
  `laaf_id_val_clip_ids.txt` (194), `laaf_id_test_clip_ids.txt` (388), `laaf_ood_test_clip_ids.txt` (759); disjoint,
  together all 2700 clips, no city or source video shared. OOD test = all clips of African, Eastern Europe and Latin
  America (no other split has them). `egogroups_city.BENCHMARKS['benchmark2']`; GT `gt_group_10fps_benchmark2_{train,val,id_test,ood_test}.{txt,pkl}`
  (`python egogroups_city.py gt --benchmark benchmark2`, seq = clip - 1); folders `EgoGroupsBenchmark2_{Train,IDTest,OODTest}`
  (+ `_jrdbstyle` copies of the two tests, + `EVP/depth/gold_egogroupsbenchmark2_*`) from
  `scripts/make_egogroupscity_folders.py --benchmark benchmark2`. `compute_detections.py` / `compute_groupings.py`
  take those folder names as `--dataset`. Not wired into the training loaders yet.
- `benchmark3` (OOD_city / OOD_regions): `benchmark3_{train,id_val,id_test,ood_test}_clip_ids.txt` (1533 / 219 / 438 / 510),
  no city or source video shared. id_test = **OOD_city** (unseen cities, seen regions), ood_test = **OOD_regions** (the only
  African + Latin America clips). GT `gt_group_10fps_benchmark3_{train,id_val,id_test,ood_test}.{txt,pkl}`; folders
  `EgoGroupsBenchmark3_Train` (train + id_val), `EgoGroupsOODCity_Test`, `EgoGroupsOODRegions_Test` (+ `_jrdbstyle` copies,
  + depth) via `--benchmark benchmark3`; tester `--dataset` options as for benchmark2; zero-shot runner `GT=benchmark3_id_test`.
  Reproduction steps + training/testing of every method: `README_benchmark3.md`, specs section 10
  (`EGO_VAL_SPLIT=benchmark3`, `--dataset egogroups-benchmark3`, test splits `ood_city` / `ood_regions`).
  - Training = the 1533 train clips; validation (checkpoint / threshold selection only, never trained on) = the 219
    id_val clips; the 948 test clips are never loaded. Methods reading `EgoGroupsBenchmark3_Train` (DVT3, P-HAR,
    GroupTrans) hold id_val out by clip id (DVT3: `DVT_EGO_VAL_CLIPS=219`). MINGLE builds pairs from train + id_val,
    drops every id_val pair from training and validates on `--val_pairs` balanced pairs drawn round-robin from id_val.
    7 train clips and 1 id_val clip have nobody in them (no examples).
- `k05` / `k20` / `k60` / `k86` / `k86v2` (small-data benchmark3, city diversity): `K{05,20,60,86,86v2}_clip_ids.txt`, 300 of
  benchmark3's 1533 train clips each, from 9 / 23 / 65 / 90 / 91 cities (7 / 16 / 36 / 46 / 45 countries; k86v2 = a
  second 90-city draw, 144 clips shared with k86, 1457 SFT train examples, 6 empty clips); lists overlap
  (e.g. K20&K60 115 clips). `BENCHMARKS[k]` (`egogroups_city.KSUBSETS`) = benchmark3 with only these train clips, same
  id_val / id_test / ood_test (GT `gt_group_10fps_<k>_*` byte-identical to benchmark3's except train). Our SFT / RL /
  MINGLE: `--dataset egogroups-k05` etc. (checkpoints `-egogroups-k05`; SFT examples train 1478 / 1496 / 1451 / 1464,
  val 1080; clips with nobody: 1 / 0 / 7 / 4); test on `EgoGroupsOODCity_Test` / `EgoGroupsOODRegions_Test` as
  benchmark3. Not wired into the baselines (no `EGO_VAL_SPLIT=k05`).
- `k05_400` / `k20_400` / `k60_400` / `k80_400` (Oct 29 2026): the same idea with 400 train clips each,
  `K{05,20,60,80}-400_clip_ids.txt`, 7 / 22 / 65 / 86 cities (6 / 16 / 37 / 43 countries), all in benchmark3 train; same
  id_val / id_test / ood_test (GT `gt_group_10fps_<k>_*`).
  - **Validation = benchmark3's id_val (219 clips) in every method**, never part of the train subset. Checkpoint / epoch
    / threshold selection uses it exactly as for benchmark3. Tests: OOD_city = id_test (438), OOD_regions = ood_test (510).
    - Ours / MINGLE: `BENCHMARKS[k]` replaces only `train`; the id_val GT is byte-identical to benchmark3's.
    - DVT3: `scene_split.py` holds out `BENCHMARKS[k]["id_val"]`.
    - P-HAR: `VAL_CLIPS`.
    - GroupTrans: `group_test.txt` of the train folder.
    - `prepare_egogroups_city.py` and `prepare_city.py` assert the id_val equals benchmark3's.
  - Ours / MINGLE: `--dataset egogroups-k05_400` etc.
  - Wired into DVT3, P-HAR and GroupTrans via `EGO_VAL_SPLIT=k05_400` (any `egogroups_city.KSUBSETS` key works):
    - DVT3: `BENCH=k05_400 GPU=G bash run_egogroups_city.sh`. It trains on `VLM-GroupDetection/EgoGroupsK05-400_Train`
      (`prepare_egogroups_city.py k05_400`: 400 + 219 id_val clips), works in `EgoGroupsK05-400_Train_run`, and tests
      both OOD sets with tag `EgoGroupsK05-400_ep<N>`.
    - P-HAR: `EGO_VAL_SPLIT=k05_400 PAR_WD=1e-3 python train_egogroups.py G` (exp `EgoGroups_k05_400_wd1e-3`).
      Test with the same env and `--split ood_city` / `ood_regions`.
    - GroupTrans: `python egotools/prepare_city.py k05_400` -> `runs/EgoGroupsK05-400_Train`. Train from that folder
      (`--taskname ego_k05_400_traj_with_app`, as for benchmark3) and test from `runs/EgoGroupsOODCity_Test` /
      `_OODRegions_Test`.
  - Code: `egogroups_city.BENCHMARKS['benchmark3']` is the single source of the lists. Baselines: `RTGD/rtgd_data.py`
    (`EGO_VAL_SPLIT == "benchmark3"`, `OOD_SPLITS`; Social-CAD, S3R2 reuse it), `PAR/.../egogroups_par.py`
    (`OOD_SPLITS`), `VLM-GroupDetection/scene_split.py` + `prepare_egogroups_city.py benchmark3` +
    `run_egogroups_city.sh` (`BENCH=benchmark3`) / `test_egogroupscity.sh` (`TESTSET=`), `GroupTrans/egotools/prepare_city.py
    benchmark3`, `MINGLE/run_mingle.py --dataset oodcity_test|oodregions_test`. Run dirs get `_benchmark3`.
  - id_val's regions are uneven vs train (Anglo 20% of train but 8 val clips; Latin Europe / Nordic Europe 27% / 21%
    of val), so checkpoint selection leans on those regions. Eastern Europe is in-distribution here (OOD in benchmark2).
- Known issue (not fixed): `JRDB_eval.py`'s `"EgoGroups_test" in detections` tests the lines of the open detection file,
  not its path, so it is never true and per-region / per-density AP for EgoGroups_test reads clip metadata from
  `/lp-dev/jmurrugarral/gold_SEKAI_900_3/`.
