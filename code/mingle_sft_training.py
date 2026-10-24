# SFT fine-tune of Qwen2.5-VL-7B-Instruct into a MINGLE-style pairwise-judgment model ("are these
# two people grouped?"), trained on EgoGroups_train's own ground truth via mingle_train_data_utils.py
# (MINGLE itself ships no training code or data -- see the plan this script implements). Modeled on
# sft_training_vlm_nodspy_merged.py's structure (arg parsing, --gpu-before-torch-import trick, LoRA
# config, SFTConfig, save+merge+reload sanity check), but:
#   - reuses MINGLE's own crop/prompt utilities (scripts/utils/{image_processing,prompts,tools}.py)
#     instead of this repo's own bbox-drawing/prompt code
#   - two images per example (RGB crop + depth crop), via trl's plural 'images' dataset column
#   - completion is the literal string "Yes"/"No" (matching MINGLE's own output format exactly), so
#     the resulting checkpoint needs no new inference code -- run_mingle.py --model <merged_dir>
#     evaluates it directly, same as the MINGLE-weights and base-model comparisons already run.
#
# Multi-GPU: run via `accelerate launch --multi_gpu --num_processes=N mingle_sft_training.py [args]`
# (no --gpu flag -- every GPU must stay visible so accelerate can assign one per process). This
# data-parallelizes across GPUs (each process trains on its own shard of the same batch, gradients
# all-reduced every step) -- it does not reduce total compute per example, so expect roughly an Nx
# wall-clock speedup, not more. Single-GPU (`python mingle_sft_training.py --gpu K`) still works
# unchanged.
#
# Checkpoints: a full merged checkpoint is saved after EVERY epoch
# (sft_output/<prefix>-mingle-pairwise-merged-egogroups-train-epoch<N>, via
# SaveMergedEachEpochCallback below), plus one more, unsuffixed, at the very end
# (sft_output/<prefix>-mingle-pairwise-merged-egogroups-train) holding whichever epoch had the
# best validation loss (load_best_model_at_end=True) -- matching the paper's own model-selection
# method. --wandb / --wandb-project log training/eval loss to Weights & Biases (off by default).
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description='SFT fine-tune Qwen2.5-VL-7B-Instruct into a MINGLE-style pairwise grouping '
                    'judge, trained on EgoGroups_train ground truth.'
    )
    parser.add_argument('--base_model', type=str, choices=['qwen2-vl', 'qwen2.5-vl'], default='qwen2-vl',
                        help="'qwen2-vl' (Qwen2-VL-7B-Instruct) is what the released MINGLE-1.0 weights were "
                             "fine-tuned from (HF model card + config.json: Qwen2VLForConditionalGeneration); "
                             "'qwen2.5-vl' is the other base the paper compares. Also sets the checkpoint "
                             "name prefix (qwen2-vl-7b-... / qwen2.5-vl-7b-...).")
    parser.add_argument('--dataset', type=str, choices=['egogroups', 'jrdb', 'jrdbego', 'egogroups-city'], default='egogroups',
                        help="egogroups (default): EgoGroups_train pairs (mingle_train_data_utils.py), validation from the "
                             "180 held-out EgoGroups clips. jrdb: JRDB train pairs (mingle_jrdb_train_data_utils.py), "
                             "training on the 20 JRDB train scenes, validation from the 7 JRDB val scenes; checkpoints "
                             "are named ...-jrdb-train... instead of ...-egogroups-train... jrdbego: both training sets "
                             "(plain union of the two pair sets, each built and cached as for its own dataset); --val_pairs "
                             "is split in half between the JRDB val scenes and the EgoGroups val clips, so the validation "
                             "loss weighs the two equally; checkpoints ...-jrdbego-train...-geo180... egogroups-city: the "
                             "egogroups-city split (egogroups_city.py; clips of both EgoGroups folders, no city shared "
                             "between splits), train pairs of its 1620 train clips, validation from its 180 val clips; "
                             "checkpoints ...-egogroups-city-train...")
    parser.add_argument('--neg_per_pos', type=float, default=1,
                        help='sampled negative pairs per positive pair per frame (see '
                             'mingle_train_data_utils.build_pairwise_examples); fractions allowed, e.g. 1.5 = '
                             "the paper's Table 1 training split (28,341 Yes / 43,654 No). Default 1 = balanced "
                             'Yes/No, matching the released MINGLE-1.0 ("61k annotated image pairs '
                             '(Balanced, Y/N only)").')
    parser.add_argument('--neg_per_pos_jrdb', type=float, default=None,
                        help="--dataset jrdbego only: negatives per positive for the JRDB pairs (default: --neg_per_pos); "
                             "e.g. 1.92 = the ratio of the JRDB-only runs (scripts/run_mingle_jrdb.sh). Adds -jrdbneg<value> "
                             "to the checkpoint names.")
    parser.add_argument('--val_pairs', type=int, default=0,
                        help="0 (default): no validation; the final checkpoint is the last epoch, and every "
                             "epoch's merged checkpoint is saved. N > 0: validate on N pairs, half Yes and half No (balanced whatever --neg_per_pos is), taken round-robin over the "
                             "180 held-out EgoGroups validation clips (1000 = ~6 per clip) and keep the epoch with the "
                             "lowest validation loss. The validation clips are never trained on either way.")
    parser.add_argument('--build_only', action='store_true',
                        help="build the records cache (crops + prompts) and exit, without loading the model "
                             "or touching a GPU. Run this once first so a multi-GPU launch doesn't hold all "
                             "its GPUs idle while rank 0 builds the dataset.")
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help="peak learning rate (the paper doesn't give one). 1e-4 trained Qwen2-VL stably but made "
                             "Qwen2.5-VL diverge late in epoch 1; use a lower value (e.g. 5e-5) for Qwen2.5-VL. A "
                             "non-default value adds -lr<value> to every checkpoint name.")
    parser.add_argument('--warmup_ratio', type=float, default=0.0,
                        help='fraction of training steps for linear learning-rate warmup (default 0 = none)')
    parser.add_argument('--seed', type=int, default=42,
                        help="training seed (LoRA init, data order). The paper repeated every experiment "
                             "twice; run once per seed. A non-default seed adds -seed<N> to every checkpoint "
                             "name so repeats don't overwrite each other. The pair sampling itself stays "
                             "fixed, so both repeats train on the same data.")
    parser.add_argument('--limit', type=int, default=None,
                        help='cap total examples built (smoke testing)')
    parser.add_argument('--epochs', type=int, default=5,
                        help="num_train_epochs. Default 5, matching MINGLE's own paper (arXiv:2509.13484): "
                             "'Each model was trained for 5 epochs and the best-performing one is selected "
                             "based on the validation set performance.' eval_strategy='epoch' below saves a "
                             "checkpoint every epoch, so the best one can still be picked afterward the same way.")
    parser.add_argument('--gpu', type=str, default=None,
                        help="CUDA_VISIBLE_DEVICES value for a single-GPU run, parsed before torch "
                             "is imported so it actually takes effect (same trick as "
                             "sft_training_vlm_nodspy_merged.py). Do not pass this when launching "
                             "with `accelerate launch --multi_gpu` -- every GPU must stay visible "
                             "for that to shard the data across GPUs; see the module docstring.")
    parser.add_argument('--wandb', action='store_true',
                        help="Log to Weights & Biases instead of nowhere (default report_to='none'). "
                             "Requires wandb to already be authenticated in this environment "
                             "(`wandb login` or a WANDB_API_KEY env var) -- not checked here. Same "
                             "convention as rl_training_llm_nodspy_merged.py's --wandb.")
    parser.add_argument('--wandb-project', type=str, default='mingle-pairwise-sft',
                        help="wandb project name to log to when --wandb is set. Ignored otherwise.")
    return parser.parse_args()


args = parse_args()
print(f'neg_per_pos: {args.neg_per_pos:g}  limit: {args.limit}  epochs: {args.epochs}  gpu: {args.gpu or "(default)"}')

# LOCAL_RANK is set by both `accelerate launch` and `torchrun` in multi-process mode -- when present,
# every GPU must stay visible (each process picks its own below), so the single-GPU default/override
# below is skipped entirely.
_distributed = 'LOCAL_RANK' in os.environ
if args.gpu is not None:
    if _distributed:
        raise ValueError("--gpu is for single-GPU runs only; launch multi-GPU via `accelerate launch "
                         "--multi_gpu --num_processes=N mingle_sft_training.py` instead (no --gpu).")
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
elif not _distributed:
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')

# Standard way the HF Trainer's wandb integration picks up a project name; only set when --wandb is
# actually requested so a plain run never even imports/touches wandb (same as the RL scripts).
if args.wandb:
    os.environ.setdefault('WANDB_PROJECT', args.wandb_project)

import datetime
import importlib.util
import json
import torch
import torch.distributed as dist
from datasets import Dataset, Sequence, Image as HFImage
from peft import LoraConfig
from PIL import Image as PILImage
from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration,
                          TrainerCallback)

# (HF model id, model class, checkpoint-name prefix) per --base_model.
BASE_MODELS = {
    'qwen2-vl': ('Qwen/Qwen2-VL-7B-Instruct', Qwen2VLForConditionalGeneration, 'qwen2-vl-7b'),
    'qwen2.5-vl': ('Qwen/Qwen2.5-VL-7B-Instruct', Qwen2_5_VLForConditionalGeneration, 'qwen2.5-vl-7b'),
}
MODEL_NAME, MODEL_CLASS, CKPT_PREFIX = BASE_MODELS[args.base_model]
# Every non-default data/optimization choice is spelled out in the checkpoint names, so runs never overwrite each
# other (e.g. a --neg_per_pos 2 run can't clobber the 1:1 checkpoints).
RUN_SUFFIX = ((f'-neg{args.neg_per_pos:g}' if args.neg_per_pos != 1 else '')
              + (f'-lr{args.learning_rate:g}' if args.learning_rate != 1e-4 else '')
              + (f'-seed{args.seed}' if args.seed != 42 else '')
              + (f'-jrdbneg{args.neg_per_pos_jrdb:g}' if args.neg_per_pos_jrdb is not None else '')
              # EgoGroups val split of mingle_train_data_utils.py: default geo180 runs get their own checkpoint names
              + ('-geo180' if args.dataset in ('egogroups', 'jrdbego') and os.environ.get('EGO_VAL_SPLIT', 'geo180') == 'geo180'
                 else ''))

# Under `accelerate launch`/`torchrun`, every process sees all GPUs and must place its model (and,
# below, initialize its NCCL process-group membership) on its OWN GPU via LOCAL_RANK -- done this
# early (before any dataset work) so both the barrier-based cache prepopulation below and the later
# model load target the right device. SFTTrainer's own internal Accelerator would otherwise set this,
# but not until it's constructed, which is too late for both uses here.
if _distributed:
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
from trl import SFTConfig, SFTTrainer

import mingle_city_data_utils
import mingle_jrdb_train_data_utils
import mingle_train_data_utils

# Pair sources: one per dataset, each with its own builder, crop cache and records cache (jrdbego uses both, unchanged)
# (egogroups-city: city train + val clips of both EgoGroups folders, mingle_city_data_utils.py; crops are shared with
# 'egogroups', whose cache is keyed by the unique clip names)
SOURCE_UTILS = {'jrdb': mingle_jrdb_train_data_utils, 'egogroups': mingle_train_data_utils,
                'egogroups-city': mingle_city_data_utils}
SOURCES = ['jrdb', 'egogroups'] if args.dataset == 'jrdbego' else [args.dataset]
data_utils = SOURCE_UTILS[SOURCES[0]]  # DATA_ROOT is the same for both
if args.neg_per_pos_jrdb is not None and args.dataset != 'jrdbego':
    raise ValueError('--neg_per_pos_jrdb is for --dataset jrdbego (use --neg_per_pos for a single dataset)')


def neg_per_pos(source):
    return args.neg_per_pos_jrdb if source == 'jrdb' and args.neg_per_pos_jrdb is not None else args.neg_per_pos
TRAIN_TAG = {'jrdb': 'jrdb-train', 'egogroups': 'egogroups-train', 'jrdbego': 'jrdbego-train',
             'egogroups-city': 'egogroups-city-train'}[args.dataset]  # checkpoint-name part

# MINGLE's scripts/utils/ is a namespace package (no __init__.py) named "utils" -- same name as this
# repo's own utils.py (a real module). A plain sys.path trick can't make the namespace package win:
# PEP 420 gives a regular module found anywhere on sys.path priority over a namespace-package
# candidate found earlier, so `from utils.image_processing import ...` would silently resolve to
# this repo's utils.py instead (and crash, since that file has no `image_processing` attribute).
# Loading each file directly by path sidesteps the name collision entirely.
MINGLE_SCRIPTS = os.path.join(data_utils.DATA_ROOT, 'MINGLE', 'scripts')


def _load_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(MINGLE_SCRIPTS, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_image_processing = _load_module('mingle_image_processing', 'utils/image_processing.py')
_prompts = _load_module('mingle_prompts', 'utils/prompts.py')
_tools = _load_module('mingle_tools', 'utils/tools.py')

SidewalkBalletImage = _image_processing.SidewalkBalletImage
prompt7 = _prompts.prompt7
compute_median_depth = _tools.compute_median_depth

def crop_cache_dir(source):
    return 'sft_output/mingle_pair_crops_cache' + ('_jrdb' if source == 'jrdb' else '')


def get_pair_crops_and_text(example, source):
    """Returns (rgb_crop_path, depth_crop_path, prompt_text). The two crop images are drawn+saved
    once and cached on disk (keyed by clip_name/tid1_tid2), same rationale as
    sft_training_vlm_nodspy_merged.py's ANNOTATED_CACHE_DIR. depth values / relative boxes are cheap
    (small-image PIL ops, no model inference) so they're recomputed every call, cache hit or not."""
    box1, box2 = example['box1'], example['box2']
    image = PILImage.open(example['image_path'])
    image_depth = PILImage.open(example['depth_path'])

    img = SidewalkBalletImage(image, [box1, box2])
    img_depth = SidewalkBalletImage(image_depth, [box1, box2])
    img_focused = img.focus(box_ids=(0, 1))
    img_depth_focused = img_depth.focus(box_ids=(0, 1))
    depth1, depth2 = compute_median_depth(image_depth, [box1, box2])
    text = prompt7(img_focused.boxes, depth1, depth2)

    cache_dir = os.path.join(crop_cache_dir(source), example['clip_name'])
    rgb_path = os.path.join(cache_dir, f"{example['tid1']}_{example['tid2']}_rgb.jpg")
    depth_path = os.path.join(cache_dir, f"{example['tid1']}_{example['tid2']}_depth.jpg")
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        os.makedirs(cache_dir, exist_ok=True)
        img_focused.show(all_boxes=True, width=1, labels=False).convert('RGB').save(rgb_path, quality=95)
        img_depth_focused.show(all_boxes=True, width=1, labels=False).convert('RGB').save(depth_path, quality=95)

    return rgb_path, depth_path, text


def build_record(example, source):
    rgb_path, depth_path, text = get_pair_crops_and_text(example, source)
    return {
        'images': [rgb_path, depth_path],
        'prompt': [{'role': 'user', 'content': text}],
        'completion': [{'role': 'assistant', 'content': example['label']}],
    }


def build_records(source):
    """Builds the SFT record for every pair, before any train/val split. Slow (reads all 9000 jsons_step4
    files and opens every pair's full RGB + depth frame to crop it / compute median depth), so the result
    is cached to RECORDS_CACHE below and only rebuilt when that file is missing."""
    print('building pairwise examples...')
    examples = SOURCE_UTILS[source].build_pairwise_examples(neg_per_pos=neg_per_pos(source), limit=args.limit)
    n_yes = sum(e['label'] == 'Yes' for e in examples)
    print(f'{len(examples)} pairwise examples ({n_yes} Yes / {len(examples) - n_yes} No) across '
          f'{len({e["scenario_idx"] for e in examples})} scenarios')
    return {'examples': examples, 'records': [build_record(e, source) for e in examples]}


# Records depend on neither --base_model nor --val_pairs (the split happens after loading), so one cache
# serves every model and every split.
def records_cache(source):
    return (f'sft_output/mingle_records_cache/{ {"jrdb": "jrdb_", "egogroups-city": "city_"}.get(source, "")}'
            f'neg{neg_per_pos(source):g}_limit{args.limit}.json')


def write_records_cache(source):
    data = build_records(source)
    path = records_cache(source)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, path)  # atomic: an interrupted build never leaves a half-written cache
    print(f'records cached to {path}')


# Under multi-GPU only rank 0 builds (every rank building would redo the same slow work N times); the
# others wait on a barrier and then just load the cached file. The long process-group timeout matters:
# a first-time build takes far longer than NCCL's default 10-minute barrier timeout.
if _distributed:
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=6))
    if int(os.environ['RANK']) == 0:
        for source in SOURCES:
            if not os.path.exists(records_cache(source)):
                write_records_cache(source)
    dist.barrier()
else:
    for source in SOURCES:
        if not os.path.exists(records_cache(source)):
            write_records_cache(source)

if args.build_only:
    print(f'--build_only: records cache ready at {", ".join(records_cache(s) for s in SOURCES)}; exiting before model load.')
    raise SystemExit(0)


def pick_val(examples, val_scenarios, n_pairs):
    """Indices of n_pairs validation pairs (half Yes, half No) taken round-robin over the held-out scenarios."""
    held_out = [i for i, e in enumerate(examples) if e['scenario_idx'] in val_scenarios]
    val_idx = set()
    if n_pairs > 0:
        import random
        rng = random.Random(42)
        for label in ('Yes', 'No'):
            per_clip = {}
            for i in held_out:
                if examples[i]['label'] == label:
                    per_clip.setdefault(examples[i]['scenario_idx'], []).append(i)
            for rows in per_clip.values():
                rng.shuffle(rows)
            picked = 0
            while picked < n_pairs // 2 and any(per_clip.values()):
                for clip in sorted(per_clip):
                    if per_clip[clip] and picked < n_pairs // 2:
                        val_idx.add(per_clip[clip].pop())
                        picked += 1
    return held_out, val_idx


# The validation scenes / clips of each source (data_utils.VAL_SCENARIOS: the 180 EgoGroups val clips, the 7 JRDB val
# scenes) are never trained on. --val_pairs > 0: the validation set is --val_pairs pairs from them only (jrdbego: half
# from each dataset), always balanced Yes/No (half each, so it stays 1:1 whatever --neg_per_pos the training pairs
# use). Each label is taken round-robin over the clips (one random pair of that label per clip per round, seed 42) so
# every clip with pairs is covered about equally (175 of the 180 EgoGroups clips have pairs; 5 scattered ones have
# nobody in a group). --val_pairs 0: no validation (the held-out clips stay held out).
examples, train_records, val_records, val_examples = [], [], [], []
for source in SOURCES:
    print(f'loading cached records from {records_cache(source)}')
    with open(records_cache(source)) as f:
        _cached = json.load(f)
    src_examples, src_records = _cached['examples'], _cached['records']
    val_scenarios = SOURCE_UTILS[source].VAL_SCENARIOS
    held_out, val_idx = pick_val(src_examples, val_scenarios, args.val_pairs // len(SOURCES))
    train_records += [r for e, r in zip(src_examples, src_records) if e['scenario_idx'] not in val_scenarios]
    val_records += [src_records[i] for i in sorted(val_idx)]
    val_examples += [src_examples[i] for i in sorted(val_idx)]
    examples += src_examples
    print(f'{source}: train {sum(e["scenario_idx"] not in val_scenarios for e in src_examples)}  val {len(val_idx)} '
          f'(of {len(held_out)} pairs in the {len(val_scenarios)} held-out scenes/clips)')
print(f'train: {len(train_records)}  val: {len(val_records)}')

print(train_records[0]['images'])
print(train_records[0]['prompt'][0]['content'])
print(train_records[0]['completion'][0]['content'])

train_dataset = Dataset.from_list(train_records).cast_column('images', Sequence(HFImage()))
val_dataset = (Dataset.from_list(val_records).cast_column('images', Sequence(HFImage()))
               if val_records else None)
print(train_dataset, val_dataset)

# (torch.cuda.set_device for the LOCAL_RANK GPU already happened earlier, right after `import torch`
# -- needed before the cache-population barrier above too, not just here.)

# Default min/max_pixels (no extra cap), matching the released MINGLE-1.0 preprocessor_config.json
# (max_pixels=12845056). A 1280*28*28 cap would downscale ~30% of the crops, many of which are whole
# 1920x1080 frames when the two people are far apart.
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = MODEL_CLASS.from_pretrained(MODEL_NAME, dtype=torch.bfloat16).to('cuda')

# Keep the effective batch at 16 regardless of GPU count (per-device 1 x accumulation x world size).
WORLD_SIZE = int(os.environ.get('WORLD_SIZE', '1'))
GRAD_ACCUM = max(1, 16 // WORLD_SIZE)

lora_config = LoraConfig(
    r=64,  # matches MINGLE's own paper's selected config ("LoRA (rank 64)" in their Table 2 model
           # selection, out of rank 64/128/full-fine-tune options they tried)
    lora_alpha=128,  # keeps this repo's existing alpha=2*r convention (was r=32/alpha=64)
    lora_dropout=0.05,
    bias='none',
    target_modules='all-linear',
    task_type='CAUSAL_LM',
)

sft_config = SFTConfig(
    output_dir=f'sft_output/{CKPT_PREFIX}-mingle-pairwise-lora-{TRAIN_TAG}{RUN_SUFFIX}',
    num_train_epochs=args.epochs,
    per_device_train_batch_size=1,  # two images/example vs. one -- start conservative, tune on OOM
    gradient_accumulation_steps=GRAD_ACCUM,  # effective batch 16 total, matching the whole-frame script's 2x8
    gradient_checkpointing=True,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_ratio,  # transformers 5: a float in [0, 1) is a ratio of total steps
    bf16=True,
    completion_only_loss=True,
    packing=False,
    padding_free=False,
    max_length=8192,  # uncapped crops reach 1920x1080 (~2.6K vision tokens each, two images per
                       # example) plus ~400 text tokens; 8192 leaves headroom so nothing is truncated.
    logging_steps=50,
    eval_strategy='epoch' if val_dataset is not None else 'no',
    save_strategy='epoch',
    # With validation, matches the paper: "the best-performing one is selected based on the validation
    # set performance" -- selects by lowest eval_loss (the default metric_for_best_model) across the
    # epoch checkpoints, so trainer.model below is the best epoch. Without (--val_pairs 0) it's simply
    # the last epoch; every epoch's merged checkpoint is still saved for picking one afterwards.
    load_best_model_at_end=val_dataset is not None,
    seed=args.seed,
    report_to='wandb' if args.wandb else 'none',
    run_name=f'{CKPT_PREFIX}-{TRAIN_TAG}{RUN_SUFFIX}' if args.wandb else None,
)


class SaveMergedEachEpochCallback(TrainerCallback):
    """Saves a full merged checkpoint after every epoch (to sft_output/...-epoch<N>), not just the
    single best-epoch merge that happens after training ends. Reuses the exact pattern
    rl_training_llm_nodspy_merged.py's own SaveMergedEachEpochCallback already worked out (and
    debugged live -- see that class's docstring for the full reasoning):
      - LoraModel.merge_adapter()/unmerge_adapter() (accessible as <PeftModel>.base_model, not on
        PeftModel itself) does a REVERSIBLE in-place merge -- unlike merge_and_unload(), which
        permanently discards the LoRA modules and would corrupt the model for the remaining epochs
        still to train.
      - merge_adapter() only updates base_layer.weight's merged *value* in place, it doesn't drop the
        LoRA submodules -- so state_dict() still has "...base_layer.weight" plus leftover
        "...lora_A/lora_B..." keys instead of the plain keys a fresh model expects. Fix: build a
        cleaned state dict (strip ".base_layer.", drop lora_A/lora_B keys) and load it into a
        freshly-constructed skeleton model, then save that skeleton normally.
    Guarded to run on the main process only -- under multi-GPU (accelerate launch), every process
    would otherwise redundantly merge+reload+save (and race on the same output directory)."""

    def on_epoch_end(self, args_, state, control, **kwargs):
        if int(os.environ.get('RANK', '0')) != 0:
            return
        peft_model = kwargs['model']
        epoch_num = int(round(state.epoch))
        epoch_dir = f'sft_output/{CKPT_PREFIX}-mingle-pairwise-merged-{TRAIN_TAG}{RUN_SUFFIX}-epoch{epoch_num}'

        peft_model.base_model.merge_adapter()
        raw_sd = peft_model.get_base_model().state_dict()
        clean_sd = {
            k.replace('.base_layer.', '.'): v.detach().cpu()
            for k, v in raw_sd.items()
            if '.lora_A.' not in k and '.lora_B.' not in k
        }
        peft_model.base_model.unmerge_adapter()  # restore for the next epoch's training ASAP,
                                                  # before the slow disk-bound skeleton reload below

        skeleton = MODEL_CLASS.from_pretrained(MODEL_NAME, dtype=torch.bfloat16)
        skeleton.load_state_dict(clean_sd, strict=True)
        skeleton.save_pretrained(epoch_dir, safe_serialization=True, max_shard_size='5GB')
        processor.save_pretrained(epoch_dir)
        del skeleton, raw_sd, clean_sd
        print(f'Epoch {epoch_num} merged checkpoint saved to {epoch_dir}')


trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_config,
    processing_class=processor,
    callbacks=[SaveMergedEachEpochCallback()],
)

trainer.train()

# Under multi-GPU (accelerate launch), every process reaches this point -- save/merge/reload/generate
# only once, on the main process, or N processes would race on the same output files and redundantly
# hold a second full model copy in GPU memory for the sanity-check generate() call below.
if not trainer.is_world_process_zero():
    import sys
    sys.exit(0)

adapter_dir = f'sft_output/{CKPT_PREFIX}-mingle-pairwise-lora-{TRAIN_TAG}{RUN_SUFFIX}'
merged_dir = f'sft_output/{CKPT_PREFIX}-mingle-pairwise-merged-{TRAIN_TAG}{RUN_SUFFIX}'

trainer.save_model(adapter_dir)
processor.save_pretrained(adapter_dir)

merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained(merged_dir, safe_serialization=True, max_shard_size='5GB')
processor.save_pretrained(merged_dir)

print(f'LoRA adapter saved to {adapter_dir}')
print(f'Merged checkpoint saved to {merged_dir}')

# Reload fresh from disk before the sanity check -- see sft_training_vlm_nodspy_merged.py's comment
# on the same pattern (avoids a position_ids generate() bug seen right after a full training run).
del model, merged_model, trainer
torch.cuda.empty_cache()

eval_processor = AutoProcessor.from_pretrained(merged_dir)
eval_model = MODEL_CLASS.from_pretrained(merged_dir, dtype=torch.bfloat16).to('cuda')
eval_model.eval()

holdout = val_examples[0] if val_examples else examples[-1]  # (a training pair when --val_pairs 0)
rgb_path, depth_path, text = get_pair_crops_and_text(holdout, SOURCES[0] if val_examples else SOURCES[-1])
holdout_rgb = PILImage.open(rgb_path).convert('RGB')
holdout_depth = PILImage.open(depth_path).convert('RGB')

messages = [{
    'role': 'user',
    'content': [
        {'type': 'image', 'image': holdout_rgb},
        {'type': 'image', 'image': holdout_depth},
        {'type': 'text', 'text': text},
    ],
}]
prompt_text = eval_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = eval_processor(
    text=[prompt_text], images=[holdout_rgb, holdout_depth], return_tensors='pt',
).to(eval_model.device)

with torch.no_grad():
    generated = eval_model.generate(**inputs, max_new_tokens=16, do_sample=False)

completion = eval_processor.batch_decode(
    generated[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True,
)[0].strip().strip('.')
print('raw completion:', completion)
print('true label:', holdout['label'])

# Using the result: run_mingle.py --dataset egogroups_test --model <merged_dir> --gpu 0 --num_parts N
# --part_id K evaluates this checkpoint directly with no new inference code -- the completion format
# above (bare "Yes"/"No") matches exactly what run_mingle.py's pair loop parses. Merge shards with
# merge_mingle_shards.py, then AP_evaluator/JRDB_eval.py -t task_3 for the mAP comparable to the
# existing MINGLE / Qwen2.5-VL-base / Qwen2-VL-base table.
