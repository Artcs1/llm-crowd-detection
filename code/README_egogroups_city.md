# egogroups-city split: building the folders from EgoGroups_train / EgoGroups_test

`egogroups-city` re-divides all 2,700 EgoGroups clips into train / val / test so that no city and no source video is
shared between splits:

| split | clips | list |
|---|---|---|
| train | 1,620 | `egogroups_city_train.txt` |
| val | 180 | `egogroups_city_val.txt` |
| test | 900 | `egogroups_city_test.txt` |

All three lists are also in `egogroups_city_split.json`, one clip number per line in each `.txt`. Densities are
22/33/44% (scattered / moderate / crowded) in every split. Every split mixes clips of both original folders:
`EgoGroups_test` holds clips 1-900 and `EgoGroups_train` holds clips 901-2700.

Nothing is copied: the new folders are relative symlinks into the original ones, so they take no extra space and
work under any mount point.

## What you need

```
<root>/
  EgoGroups_train/   jsons_step1..6/, videos_frames/   (clips 0901-2700, 5 sub-clips each: clip_XXXX_YYYYY)
  EgoGroups_test/    jsons_step1..6/, videos_frames/   (clips 0001-0900)
  EVP/depth/gold_egogroups_{train,test}/                (optional: EVP depth maps, only MINGLE uses them)
  llm-crowd-detection/                                  (this repo)
```

`<root>` defaults to the parent of `llm-crowd-detection`. If your datasets live elsewhere, pass `--root`.

## 1. Dataset folders

```bash
cd llm-crowd-detection/code
python scripts/make_egogroupscity_folders.py            # or: --root /path/to/datasets   [--no_depth]
```

This creates:

```
<root>/EgoGroupsCity_Train/   the 1,620 train + 180 val clips (1,800 x 5 sub-clips); val = egogroups_city_val.txt
<root>/EgoGroupsCity_Test/    the 900 test clips (4,500 sub-clips)
<root>/EVP/depth/gold_egogroupscity_{train,test}/     (if the EVP depth maps exist)
```

Each folder has the same sub-folders as the originals (`jsons_step1..6`, `videos_frames`). Two details:

- **Clip names keep their ORIGINAL numbers**, so each folder's clip ids are non-contiguous:
  - `EgoGroupsCity_Test` = 312 clips from 1-900 + 588 from 901-2700.
  - `EgoGroupsCity_Train` = 588 + 1,212.
- **`EgoGroups_train` has no `jsons_step6`.** For its clips the `jsons_step6` entry links to their `jsons_step4`
  file. Step6 is step4 plus the depthanything / wilddet fields, with the same boxes, track ids and `detany_3D`
  values. `jsons_step5` exists only for part of those clips, and missing files stay missing.

Re-running the script is safe: it keeps existing links and adds missing ones. Check that nothing is broken:

```bash
find <root>/EgoGroupsCity_Train <root>/EgoGroupsCity_Test -maxdepth 2 -xtype l | wc -l    # expect 0
```

## 2. Ground truth

The city GT files are in the repo. To rebuild them from `gt_group_10fps_{test,train}.*`, run:

```bash
python egogroups_city.py gt
```

This writes `AP_evaluator/out/gt_group_10fps_city_{train,val,test}.txt` and
`F1_evaluator/out/gt_group_10fps_city_{train,val,test}.pkl`.

The city GT files use **one numbering for every clip: `seq = clip_number - 1`**. The original files differ:
`gt_group_10fps_test.*` also uses `clip - 1`, but `gt_group_10fps_train.*` uses `clip - 901`. 15 clips of 1-900 have
nobody at any keyframe, so they have no txt rows, the same as in `gt_group_10fps_test.txt`.

Per-region / per-density AP (`JRDB_eval.py --output NE`, `crowded`, `AN_moderate`, ...) on `EgoGroupsCity` detection
files reads each clip's metadata from `code/annotations/jsons_step1/clip_XXXX.json`. That folder is not tracked in
git, so copy it over if you need those breakdowns.

## 3. Per-method copies (only for the baselines you run)

- **DVT3 / P-HAR** (they read DVT's converted `images/` + `labels/` format):
  ```bash
  python VLM-GroupDetection/prepare_egogroups_city.py
  ```
  This needs `VLM-GroupDetection/EgoGroups_{train,test}` from `convert_egogroups.py`, and creates
  `VLM-GroupDetection/EgoGroupsCity_{Train,Test}`.
- **GroupTrans:**
  ```bash
  cd GroupTrans && python egotools/prepare_city.py
  ```
  This needs `runs/ego_test` and `runs/ego_train_geo180` with their appearance features (`prepare_setup.py` +
  `extract_features.py`), and creates `runs/EgoGroupsCity_{Train,Test}`.
- **RTGD / Social-CAD / JRDB-Act / S3R2:** nothing to build. They read `~/tracking_annotations.json` and the frames
  under `<root>/EgoGroupsCity_*`.

## 4. Using it

Testing any checkpoint on the city test set:

```bash
EGO_DATA=EgoGroupsCity_Test bash scripts/run_groups.bash <model> <port> <api_key> p1_visual single
python compute_detections.py --dataset EgoGroupsCity_Test --mode single --depth_method detany_3D \
    --prompt_method p1_visual --model <model name> --vlm_mode vlm_image --frame_id 10
```

Zero-shot runs (vLLM server + parallel clients + AP / F1):

```bash
bash scripts/run_zeroshot_egogroupscity.sh <gpu> <HF model> <prompt> <port> <n clients>
```

Training:

- SFT / RL: `--dataset egogroups-city`, checkpoints `-egogroups-city`.
- MINGLE: `mingle_sft_training.py --dataset egogroups-city`.
- Baselines: `EGO_VAL_SPLIT=city`.
- Baseline testers on any model: `--split city_test`, `test_egogroupscity.sh` (DVT3), or
  `run_mingle.py --dataset egogroupscity_test`.

All commands are in `TRAINING_TESTING_SPECS.txt` section 9 (next to the repos) and in `CLAUDE.md`
("EgoGroups splits").
