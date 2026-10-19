# SFT / RL training scripts

DSPy-free ("nodspy") fine-tuning scripts for the social-group-detection LLM/VLM: `trl.SFTTrainer`
for supervised fine-tuning, `trl.GRPOTrainer` (GRPO/RLVR — reward computed entirely from code, no
learned reward model) for a further RL stage on top. All scripts pin their own `CUDA_VISIBLE_DEVICES`
via `--gpu` (or an already-exported env var), so several can run concurrently on different GPUs.

## Files

| File | Model | Stage |
|---|---|---|
| `sft_training_llm_nodspy_merged.py` | Qwen2.5-7B-Instruct (text-only) | SFT |
| `sft_training_vlm_nodspy_merged.py` | Qwen2.5-VL-7B-Instruct (image-grounded) | SFT |
| `rl_training_llm_nodspy_merged.py` | Qwen2.5-7B-Instruct | RL (GRPO) |
| `rl_training_vlm_nodspy_merged.py` | Qwen2.5-VL-7B-Instruct | RL (GRPO) |

Each RL script requires the matching SFT checkpoint (same `--dataset`/`--mode`/`--variant`) to
already exist on disk — **run SFT before RL**.

Data-loading dependencies (no CLI of their own, imported by the scripts above):
- `sft_data_utils.py` — JRDB (`JRDB_train_fixed_gold`, `F1_evaluator/out/gt.pkl`)
- `egogroups_data_utils.py` — `gold_SEKAI_900_3` (3 fixed target frames per clip: 2/22/42,
  `F1_evaluator/out/gt_gold_sekai_{2,22,42}.pkl`)
- `egogroups_train_data_utils.py` — `EgoGroups_train` (1800 base clips × 5 ten-frame sub-clips,
  each an independent scenario targeting its own last frame,
  `F1_evaluator/out/gt_group_10fps_train.pkl`)

## `--dataset` (both LLM scripts and both VLM scripts)

- `jrdb` (default)
- `egogroups` — `gold_SEKAI_900_3`
- `egogroups-subset` — same, minus all-singleton examples (no real/non-singleton groups)
- `egogroups-train` — `EgoGroups_train`
- `egogroups-train-subset` — same, minus all-singleton examples

## `--mode` (LLM scripts only) / `--variant` (VLM scripts only)

LLM: `single` (bare single-frame input) or `full` (target frame enriched with a per-person
`movement_direction` label computed from every earlier frame in the scenario).

VLM: `p1_visual` (bbox/id-annotated image + numeric x,y,z) or `idsonly` (annotated image + bare
`person_ids` list, no coordinates). No VLM "full" setting exists.

## RL-only flags

- `--epochs N` — `num_train_epochs` (LLM default 6, VLM default 3). Any non-default value adds a
  `-<N>ep` suffix to every saved checkpoint dir, so it never collides with a different epoch count.
- `--save-every-epoch` — also save a full merged checkpoint after every epoch, to
  `<merged_dir>-epoch<N>`, not just at the end. Off by default (each epoch's checkpoint is a full
  model copy — multiplies disk usage and adds real time per epoch). Useful for testing intermediate
  progress on a long run instead of waiting for it to fully finish.
- `--wandb` / `--wandb-project` (LLM script only) — log to Weights & Biases instead of nowhere.
- `--prompts-per-step N` (LLM script only) — prompts sampled together per optimizer step. Leave at
  the default (1) unless you know what you're changing; the batch-size math was tuned to avoid OOM.

## Checkpoint naming

`sft_output/qwen2.5-{7b,vl-7b}-groups-{lora,merged}-nodspy[-full][-idsonly][-<dataset>]` for SFT;
`sft_output/qwen2.5-{7b,vl-7b}-groups-grpo-{lora,merged}-fourlosses[-full][-idsonly][-<dataset>][-<N>ep][-epoch<N>]`
for RL. `jrdb` contributes no dataset suffix; `-full`/`-idsonly` always precede `-<dataset>` when
both apply. **Check which flags actually produced a given checkpoint before reusing it** — the
suffixes aren't auto-discoverable from the checkpoint alone.

## Running concurrently

Every training script initializes a `torch.distributed` process group even for a single GPU, using
the default rendezvous port 29500. **Running more than one of these at the same time on the same
machine will collide** with `DistNetworkError: ... EADDRINUSE` on the second and later launches.
Give each concurrent run its own port:

```bash
MASTER_PORT=29501 python rl_training_llm_nodspy_merged.py --dataset jrdb --mode single --gpu 2
MASTER_PORT=29502 python rl_training_llm_nodspy_merged.py --dataset jrdb --mode full --gpu 3
MASTER_PORT=29503 python rl_training_vlm_nodspy_merged.py --dataset jrdb --variant p1_visual --gpu 6
```

## Example commands

```bash
# ---- SFT ----
python sft_training_llm_nodspy_merged.py --dataset egogroups-train --mode single --gpu 0
python sft_training_vlm_nodspy_merged.py --dataset egogroups-train --variant p1_visual --gpu 1

# ---- RL (after the matching SFT checkpoint above exists) ----
python rl_training_llm_nodspy_merged.py --dataset egogroups-train --mode single --epochs 5 --save-every-epoch --gpu 0
python rl_training_vlm_nodspy_merged.py --dataset egogroups-train --variant p1_visual --epochs 3 --save-every-epoch --gpu 1
```

## Testing a checkpoint

Host it with `vllm serve` (any free GPU; cap `--gpu-memory-utilization` if sharing a GPU with
another server):

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve sft_output/qwen2.5-7b-groups-grpo-merged-fourlosses-egogroups-train-5ep \
  --served-model-name my-checkpoint \
  --api-key testkey \
  --port 8000
```

Then drive it through `batch_fetch_groups.py`/`fetch_groups.py` pointed at the dataset the
checkpoint was actually trained on (`--prompt_method p1`, matching `--mode`/setting). The 5th
positional argument to `batch_fetch_groups.py` (`--model`, effectively) is just a label used to
build the results directory path — it **must contain a `/`** (`save_frame()` does
`model.split("/")[1]`), e.g. `local/my-checkpoint`, and doesn't need to match `--served-model-name`;
only `--api_base`'s port controls which model actually answers.
