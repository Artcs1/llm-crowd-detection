# Testing an EgoGroups-trained model on JRDB (EgoGroups-style 3D)

Models trained on EgoGroups (benchmark3, K05 / K20 / K60 / K86, city, ...) expect EgoGroups-style 3D positions (camera
frame of detany_3D: x right, y down, z depth). For the JRDB test they are run on `JRDB_fixed_gold_egostyle`: the 1,829
JRDB test keyframes with each person's lidar position rewritten in that frame (a rotation plus a shift, so distances
between people are unchanged; `scripts/make_jrdb_egostyle.py`). Scores are against the normal JRDB test ground truth.

## 1. Files

Layout (folder names matter: the scripts read `../../<folder>` from `llm-crowd-detection/code`):

```
/data/add_disk0/jmurrugarral/
├── JRDB_fixed_gold_egostyle/
│   ├── jsons_gold/            1,829 JSON files (95 MB)
│   └── videos_frames -> ../JRDB_fixed_gold/videos_frames   (relative link)
├── JRDB_fixed_gold/
│   └── videos_frames/         one folder per keyframe; single mode reads only 00015.jpeg
└── llm-crowd-detection/       this repo (git clone https://github.com/Artcs1/llm-crowd-detection.git)
    └── code/sft_output/<merged checkpoint>/
```

- **Data:** copied with `rsync` from the original server (`JRDB_fixed_gold_egostyle` with `-a`, so the link is kept;
  `JRDB_fixed_gold/videos_frames` with `--include='*/' --include='00015.jpeg' --exclude='*'` for single-frame testing).
- **Model:** the *merged* checkpoint folder, e.g. `sft_output/qwen3-vl-4b-groups-merged-nodspy-egogroups-benchmark3`
  (all of it: weight shards, `config.json`, tokenizer / processor files). Not the `-lora-` folder.
- **Code and ground truth:** in git (`AP_evaluator/out/gt_group.txt`, `F1_evaluator/out/gt.pkl`,
  `AP_evaluator/label_map/task_3.pbtxt`, evaluators, `batch_fetch_groups.py`, `scripts/test_jrdb_egostyle.sh`,
  `scripts/jrdb_groupings.py`).
- **Environment:** conda env `py10-dspy` (Python 3.10, vLLM 0.19, dspy, litellm). Another env name: `ENV=<name>`.

Check:

```bash
cd /data/add_disk0/jmurrugarral
ls JRDB_fixed_gold_egostyle/jsons_gold | wc -l                                    # 1829
ls JRDB_fixed_gold_egostyle/videos_frames/cubberly-auditorium-2019-04-22_1_00000/   # 00015.jpeg
```

## 2. Run

From `llm-crowd-detection/code`, one GPU with ~60 GB free (inside `tmux`):

```bash
cd /data/add_disk0/jmurrugarral/llm-crowd-detection/code
GPU=0 PORT=8401 \
  MODEL=sft_output/qwen3-vl-4b-groups-merged-nodspy-egogroups-benchmark3 \
  NAME=Qwen/Qwen3-VL-4B-Instruct-SFT-Benchmark3Train \
  bash scripts/test_jrdb_egostyle.sh
```

- `MODEL`: the merged checkpoint folder. `NAME`: any served name with exactly one `/`; its part after `/` names the
  output files, so use a different `NAME` per model (e.g. `Qwen/Qwen3-VL-4B-Instruct-SFT-K05`). The benchmark3 SFT model was
  already run under `Qwen/Qwen3-VL-4B-Instruct-SFT-Benchmark3Train` (1,812 / 1,829 keyframes in the shared repo); reusing that
  name in the shared repo only fills in the missing 17.
- Options: `NPARTS=32` parallel clients, `GPU_UTIL=0.85` vLLM memory share. Lower both if the machine is short of RAM
  (on the original server the out-of-memory killer stopped runs) or the GPU is shared.
- It starts a vLLM server, runs p1_visual single-frame inference on all 1,829 keyframes, stops the server and scores.
  ~30-40 min for a 4B model on one GPU.
- **Resuming:** rerunning the same command only redoes keyframes with no prediction yet. If the "inference done" line
  says fewer than 1829, rerun it before trusting the scores.

## 3. Output

`sft_output/logs/jrdb_egostyle_test/<name after />_metrics.txt`:

- `AP, G1..G5`: overall AP, then AP for groups of 1, 2, 3, 4, 5+ (JRDB_eval.py task_3).
- `MINGLE`: mIoU, P, R, F1 on group boxes, number of predicted / GT groups (mingle_det.py).
- `F1 half` / `F1 exact`: P / R / F1 for G1..G5+; `Group` = overall F1 (singletons excluded, the number we report),
  `All` = with singletons. Half: a predicted group matches if it shares at least half the members; exact: same members.

Intermediate files: predictions `../results/predictions/JRDB_fixed_gold_egostyle/results/<name>/...`, AP-format
detections `../results/detection_files/JRDB_fixed_gold_egostyle_15/`, F1 groupings
`../results/grouping_files/JRDB_fixed_gold_egostyle_15/`, logs (server, clients) next to the metrics file.

## Notes

- Weights written on the shared NFS are mode 600; if the model folder is unreadable: `chmod -R a+rX <folder>`.
- JRDB-trained models are tested on `JRDB_fixed_gold` (native lidar 3D) instead; this script is only for EgoGroups-trained ones.
