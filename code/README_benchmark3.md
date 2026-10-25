# benchmark3: OOD_city and OOD_regions test sets from EgoGroups_train / EgoGroups_test

`benchmark3` re-divides all 2,700 EgoGroups clips (clip numbers 1-2700) into four disjoint splits. No city and no
source video is shared between any two splits. It gives two out-of-distribution tests:

- **OOD_city** (`id_test`): cities never seen in training, from regions that are seen in training.
- **OOD_regions** (`ood_test`): regions never seen in training. These are the only African and Latin America clips.

| split | clips | sub-clips | list | folder |
|---|---|---|---|---|
| train | 1,533 | 7,665 | `benchmark3_train_clip_ids.txt` | `EgoGroupsBenchmark3_Train` (train + id_val) |
| id_val | 219 | 1,095 | `benchmark3_id_val_clip_ids.txt` | `EgoGroupsBenchmark3_Train` (train + id_val) |
| **id_test = OOD_city** | 438 | 2,190 | `benchmark3_id_test_clip_ids.txt` | `EgoGroupsOODCity_Test` (+ `_jrdbstyle`) |
| **ood_test = OOD_regions** | 510 | 2,550 | `benchmark3_ood_test_clip_ids.txt` | `EgoGroupsOODRegions_Test` (+ `_jrdbstyle`) |

The four lists have one clip number per line and are in `code/`. Together they hold every clip 1-2700 exactly once.

Nothing is copied: the new folders are relative symlinks into the original ones, so they take no extra space and
work under any mount point.

## Split statistics (for reference / to check your build)

Leaks between splits, counted over all four splits:

| | total | in more than one split |
|---|---|---|
| cities | 210 | 0 |
| source videos | 494 | 0 |
| video segments | 2,090 | 0 |
| countries | 92 | 22, only among train / id_val / id_test (never ood_test) |

The video id is the first 11 characters of `original_path`'s file name. For `chunk_XXX.mp4` files it is the `[id]` in
the parent folder's name instead.

Density (the `density` field of each clip; people per keyframe = annotated people averaged over the 5 keyframes):

| split | scattered | moderate | crowded | people per keyframe, mean / median |
|---|---|---|---|---|
| train | 18.6% | 33.2% | 48.2% | 20.6 / 18.8 |
| id_val | 27.4% | 27.9% | 44.7% | 18.7 / 16.2 |
| id_test (OOD_city) | 32.6% | 33.3% | 34.0% | 16.5 / 13.8 |
| ood_test (OOD_regions) | 22.0% | 36.1% | 42.0% | 16.7 / 13.3 |

Regions:

| region | train | id_val | id_test | ood_test |
|---|---|---|---|---|
| African | 0 | 0 | 0 | 255 |
| Latin America | 0 | 0 | 0 | 255 |
| Anglo | 308 | 8 | 25 | 0 |
| Confucian Asia | 225 | 13 | 56 | 0 |
| Eastern Europe | 174 | 25 | 50 | 0 |
| Germanic Europe | 169 | 24 | 56 | 0 |
| Latin Europe | 131 | 59 | 80 | 0 |
| Middle East | 186 | 22 | 57 | 0 |
| Nordic Europe | 146 | 47 | 59 | 0 |
| South-East Asia | 184 | 21 | 55 | 0 |
| Other | 10 | 0 | 0 | 0 |

In every split, about 2/3 of the clips come from Sekai and 1/3 from our own videos.

## What you need

```
<root>/
  EgoGroups_train/   jsons_step1..6/, videos_frames/   (clips 0901-2700, 5 sub-clips each: clip_XXXX_YYYYY)
  EgoGroups_test/    jsons_step1..6/, videos_frames/   (clips 0001-0900)
  EVP/depth/gold_egogroups_{train,test}/                (optional: EVP depth maps, only MINGLE uses them)
  llm-crowd-detection/                                  (this repo)
```

`<root>` defaults to the parent of `llm-crowd-detection`. If your datasets live elsewhere, pass `--root` (folders) or
set `EGO_ROOT` (JRDB-style copies).

Files from this repo that the steps use:

- the four `benchmark3_*_clip_ids.txt` lists;
- `egogroups_city.py` (`BENCHMARKS['benchmark3']`) and `scripts/make_egogroupscity_folders.py` (`--benchmark benchmark3`);
- `scripts/make_egogroups_jrdbstyle.py`;
- `AP_evaluator/out/gt_group_10fps_{test,train}.txt` and `F1_evaluator/out/gt_group_10fps_{test,train}.pkl`
  (the original GT, tracked in git);
- for the MINGLE metric only: `~/tracking_annotations.json` (person boxes of all 2,700 clips);
- for per-region / per-density AP only: `code/annotations/jsons_step1/clip_XXXX.json`. This folder is not tracked in
  git, so copy it from the original server.

## 1. Dataset folders

```bash
cd llm-crowd-detection/code
python scripts/make_egogroupscity_folders.py --benchmark benchmark3     # [--root /path/to/datasets] [--no_depth]
```

This creates:

```
<root>/EgoGroupsBenchmark3_Train/    train + id_val (1,752 clips x 5 = 8,760 sub-clips)
<root>/EgoGroupsOODCity_Test/        id_test  (438 clips, 2,190 sub-clips)
<root>/EgoGroupsOODRegions_Test/     ood_test (510 clips, 2,550 sub-clips)
<root>/EVP/depth/gold_egogroupsbenchmark3_train/, gold_egogroupsoodcity_test/, gold_egogroupsoodregions_test/
                                     (only if the EVP depth maps exist)
```

Each folder has the same sub-folders as the originals (`jsons_step1..6`, `videos_frames`). Two details:

- **Clip names keep their ORIGINAL numbers**, so each folder's clip ids are non-contiguous and mix both original
  folders (clips 1-900 come from `EgoGroups_test`, 901-2700 from `EgoGroups_train`).
- **`EgoGroups_train` has no `jsons_step6`.** For its clips the `jsons_step6` entry links to their `jsons_step4`
  file. Step6 is step4 plus the depthanything / wilddet fields, with the same boxes, track ids and `detany_3D`
  values. `jsons_step5` exists only for part of the clips, and missing files stay missing.

Re-running the script is safe: it keeps existing links and adds missing ones. Expected output:

```
EgoGroupsBenchmark3_Train 1752 clips {'jsons_step1': 8760, ..., 'jsons_step5': 6620, 'jsons_step6': 8760, 'videos_frames': 8760, 'depth': 8760}
EgoGroupsOODCity_Test 438 clips {'jsons_step1': 2190, ..., 'jsons_step5': 1677, 'jsons_step6': 2190, 'videos_frames': 2190, 'depth': 2190}
EgoGroupsOODRegions_Test 510 clips {'jsons_step1': 2550, ..., 'jsons_step5': 1907, 'jsons_step6': 2550, 'videos_frames': 2550, 'depth': 2550}
```

Check that nothing is broken:

```bash
find <root>/EgoGroupsBenchmark3_Train <root>/EgoGroupsOODCity_Test <root>/EgoGroupsOODRegions_Test -xtype l | wc -l   # expect 0
```

## 2. JRDB-style copies (for models trained on JRDB)

```bash
python scripts/make_egogroups_jrdbstyle.py EgoGroupsOODCity_Test       # -> <root>/EgoGroupsOODCity_Test_jrdbstyle
python scripts/make_egogroups_jrdbstyle.py EgoGroupsOODRegions_Test    # -> <root>/EgoGroupsOODRegions_Test_jrdbstyle
```

These rewrite every `detany_3D` position from the camera frame (x right, y down, z forward) into the JRDB robot frame
(x forward, y left, z up): `X = z, Y = -x, Z = -y + 0.89`. This is a rotation plus a shift, so every distance between
people is unchanged. Only `jsons_step6` is written (2,190 and 2,550 files). `videos_frames` is a link to the normal
folder's frames.

## 3. Ground truth

```bash
python egogroups_city.py gt --benchmark benchmark3
```

This writes `AP_evaluator/out/gt_group_10fps_benchmark3_{train,id_val,id_test,ood_test}.txt` (AP) and
`F1_evaluator/out/gt_group_10fps_benchmark3_{train,id_val,id_test,ood_test}.pkl` (F1). Expected output:

| file | txt clips | pkl clips |
|---|---|---|
| train | 1,526 of 1,533 | 1,533 |
| id_val | 218 of 219 | 219 |
| id_test | 432 of 438 | 438 |
| ood_test | 509 of 510 | 510 |

These files use **one numbering for every clip: `seq = clip_number - 1`**. The original files differ:
`gt_group_10fps_test.*` also uses `clip - 1`, but `gt_group_10fps_train.*` uses `clip - 901`. Clips with nobody at a
keyframe have no txt rows, the same as in the original files.

Sanity check: scoring a GT file against itself must give AP ~1.0 (G1..G5 ~1.0):

```bash
cd AP_evaluator
python JRDB_eval.py -g out/gt_group_10fps_benchmark3_id_test.txt -d out/gt_group_10fps_benchmark3_id_test.txt \
    -t task_3 --labelmap ./label_map/task_3.pbtxt --output all       # 0.9998, 0.9997, 0.9998, 0.9994, 1.0, 1.0
```

## 4. Testing a model

`compute_detections.py` / `compute_groupings.py` accept `EgoGroupsOODCity_Test`, `EgoGroupsOODRegions_Test` and their
`_jrdbstyle` copies as `--dataset` (image size 1080x1920, `seq = clip - 1`, like `EgoGroupsCity_Test`).
Use the `_jrdbstyle` folders for JRDB-trained models and the normal ones for everything else.

Zero-shot or any checkpoint, one vLLM server plus parallel clients, then AP and F1. `DATA` picks the test folder and
`GT` the ground truth:

```bash
DATA=EgoGroupsOODCity_Test    GT=benchmark3_id_test  bash scripts/run_zeroshot_egogroupscity.sh <gpu> <HF model or ckpt> p1_visual <port> 48 [single|full]
DATA=EgoGroupsOODRegions_Test GT=benchmark3_ood_test bash scripts/run_zeroshot_egogroupscity.sh <gpu> <HF model or ckpt> p1_visual <port> 48 [single|full]
# trained checkpoint: also NAME=Qwen/<served name>  (must contain one '/')
# JRDB-trained:       DATA=EgoGroupsOODCity_Test_jrdbstyle (same GT)
```

Or by hand, for a model whose predictions are already in `../results/predictions/<DATA>/`:

```bash
D=EgoGroupsOODCity_Test; GT=benchmark3_id_test      # or EgoGroupsOODRegions_Test / benchmark3_ood_test
python compute_detections.py --dataset $D --mode single --depth_method detany_3D --prompt_method p1_visual \
    --model <model name> --vlm_mode vlm_image --frame_id 10
DET=../results/detection_files/${D}_10/results_<model name>_vlm_image_detany_3D_p1_visual.txt
PKL=../results/grouping_files/${D}_10/results_<model name>_vlm_image_detany_3D_p1_visual.pkl
python F1_evaluator/det_txt_to_pkl.py $DET $PKL --split test          # --split test = seq is clip - 1

# AP (+ G1..G5); --output crowded / moderate / scattered / AN / CA ... for the per-density / per-region breakdowns
(cd AP_evaluator && python JRDB_eval.py -g out/gt_group_10fps_$GT.txt -d ../$DET -t task_3 --labelmap ./label_map/task_3.pbtxt --output all)
# F1 (our evaluator)
(cd F1_evaluator && python ours_jrdb_evaluation.py --det ../$PKL --gt out/gt_group_10fps_$GT.pkl --o all)
# MINGLE group-box F1
(cd F1_evaluator && python mingle_metric.py --dets ../../results/grouping_files/${D}_10 --gt out/gt_group_10fps_$GT.pkl \
    --csv mingle_$D.csv --clips <root>/$D/jsons_step1)
```

The `_jrdbstyle` detection / grouping files land in `<DATA>_jrdbstyle_10/` and are scored against the same GT.

## 5. Training on benchmark3 (train = 1,533 clips, validation = the 219 id_val clips)

Every method has a benchmark3 option that mirrors its egogroups-city one: `EGO_VAL_SPLIT=benchmark3` (baselines) or
`--dataset egogroups-benchmark3` (ours, MINGLE). Run outputs get `*_benchmark3` / `-egogroups-benchmark3` names, so
nothing older is overwritten. Per-method data folders (only for the methods you run):

```bash
python ../../VLM-GroupDetection/prepare_egogroups_city.py benchmark3   # DVT3 / P-HAR / GroupTrans labels (needs VLM-GroupDetection/EgoGroups_{train,test})
cd ../../GroupTrans && python egotools/prepare_city.py benchmark3       # GroupTrans (needs runs/ego_test, runs/ego_train_geo180)
```

| method | train | test on OOD_city / OOD_regions |
|---|---|---|
| Ours SFT | `python sft_training_vlm_nodspy_merged.py --dataset egogroups-benchmark3 --variant p1_visual --mode single --gpu G` | `DATA=EgoGroupsOODCity_Test GT=benchmark3_id_test NAME=Qwen/<name> bash scripts/run_zeroshot_egogroupscity.sh <gpu> <ckpt> p1_visual <port> 48` |
| Ours RL | `python rl_training_vlm_nodspy_merged.py --dataset egogroups-benchmark3 ...` (after the SFT run) | same |
| MINGLE | `python mingle_sft_training.py --dataset egogroups-benchmark3 --neg_per_pos 1.6 --val_pairs 1500 ...` | `MINGLE/run_mingle.py --dataset oodcity_test` / `oodregions_test` |
| RTGD | `EGO_VAL_SPLIT=benchmark3 CUDA_VISIBLE_DEVICES="" python rtgd_run.py train --data ego --out runs/ego --threads 8` | `rtgd_run.py test --data ego --split ood_city` / `ood_regions` |
| Social-CAD / JRDB-Act | `EGO_VAL_SPLIT=benchmark3 python -m sg.train train --data ego --out runs/ego --gpu G` (`--head jrdbact`) | `--split ood_city` / `ood_regions` |
| S3R2 | `EGO_VAL_SPLIT=benchmark3 run_shgd.py --data ego` | `--split ood_city` / `ood_regions` |
| GroupTrans | `cd runs/EgoGroupsBenchmark3_Train && python ../../jrdb_train_ours.py --gpu G --taskname ego_benchmark3_traj_with_app --appearance --refresh_ana 1 --scene_visits_per_epoch 1 --num_epoch 200` | from `runs/EgoGroupsOODCity_Test` / `runs/EgoGroupsOODRegions_Test` |
| P-HAR | `EGO_VAL_SPLIT=benchmark3 PAR_WD=1e-3 python train_egogroups.py G` | `test_egogroups.py --split ood_city` / `ood_regions` |
| DVT3 | `BENCH=benchmark3 GPU=G bash run_egogroups_city.sh` (trains, then tests both) | `TESTSET=EgoGroupsOODCity_Test CK=<ckpt> TAG=<name> bash test_egogroupscity.sh` |

All commands are also in `TRAINING_TESTING_SPECS.txt` section 10 (next to the repos).
