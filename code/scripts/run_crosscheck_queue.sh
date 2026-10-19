#!/bin/bash
# Group-detection test queue, resumable (--skip_existing), one model at a time spread over 4 GPUs with
# scripts/run_groups_fastgpus_v2.sh. Cross-dataset checks use the test set converted to the model's training 3D
# convention: EgoGroups-trained -> JRDB_fixed_gold_egostyle, JRDB-trained -> EgoGroups_test_jrdbstyle.
# Usage: bash scripts/run_crosscheck_queue.sh <A|B>   A on GPUs 0-3 (with the MINGLE training), B on GPUs 4-7 (with
# the MINGLE JRDB test); vLLM capped at 45%.   Logs: sft_output/logs/crosscheck_queue/<dataset>_<name>/
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/crosscheck_queue
P=$S/qwen2.5-7b-groups-grpo-merged-fourlosses
V=$S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses
# dataset | checkpoint | served name | prompt | type | log name
if [ "$1" = A ]; then
  export GPUS="0 1 2 3" UTILS="0.45 0.45 0.45 0.45" PORT0=8091
  JOBS=(
    # EgoGroups-train-valsplit: EgoGroups test (native) and JRDB test (EgoGroups-style 3D)
    "ego|$S/qwen2.5-7b-groups-merged-nodspy-full-egogroups-train-valsplit|Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit|p1|full|valsplit_llm_full_sft"
    "jrdb_egostyle|$S/qwen2.5-7b-groups-merged-nodspy-egogroups-train-valsplit|Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit|p1|single|valsplit_llm_single_sft"
    "jrdb_egostyle|$S/qwen2.5-7b-groups-grpo-merged-fourlosses-egogroups-train-valsplit-1ep|Qwen/Qwen2.5-7B-Instruct-RL-EgoGroupTrain-ValSplit|p1|single|valsplit_llm_single_rl"
    "jrdb_egostyle|$S/qwen2.5-7b-groups-merged-nodspy-full-egogroups-train-valsplit|Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit|p1|full|valsplit_llm_full_sft"
    # JRDB train-only (-JRDBTrain-v2): EgoGroups test (JRDB-style 3D)
    "ego_jrdbstyle|$S/qwen2.5-7b-groups-merged-nodspy-jrdbtrain|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain-v2|p1|single|v2_llm_single_sft"
    "ego_jrdbstyle|$P-jrdbtrain-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-v2|p1|single|v2_llm_single_rl"
    "ego_jrdbstyle|$S/qwen2.5-7b-groups-merged-nodspy-full-jrdbtrain|Qwen/Qwen2.5-7B-Instruct-SFT-JRDBTrain-v2|p1|full|v2_llm_full_sft"
    "ego_jrdbstyle|$P-full-jrdbtrain-1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-v2|p1|full|v2_llm_full_rl"
    "ego_jrdbstyle|$S/qwen2.5-vl-7b-groups-merged-nodspy-jrdbtrain|Qwen/Qwen2.5-VL-7B-Instruct-SFT-JRDBTrain-v2|p1_visual|single|v2_vlm_single_sft"
  )
else
  export GPUS="4 5 6 7" UTILS="0.45 0.45 0.45 0.45" PORT0=8095
  JOBS=(
    "ego_jrdbstyle|$V-jrdbtrain-1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-v2|p1_visual|single|v2_vlm_single_rl"
    # GRPO-on-JRDB-val (-RLVal, -RLValPlus1ep): EgoGroups test (JRDB-style 3D)
    "ego_jrdbstyle|$P-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLVal|p1|single|rlval_llm_single"
    "ego_jrdbstyle|$P-full-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLVal|p1|full|rlval_llm_full"
    "ego_jrdbstyle|$V-jrdbtrain-1ep-rlval|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLVal|p1_visual|single|rlval_vlm_single"
    "ego_jrdbstyle|$P-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1|single|rlvalplus1ep_llm_single"
    "ego_jrdbstyle|$P-full-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1|full|rlvalplus1ep_llm_full"
    "ego_jrdbstyle|$V-jrdbtrain-1ep-rlval-plus1ep|Qwen/Qwen2.5-VL-7B-Instruct-RL-JRDBTrain-RLValPlus1ep|p1_visual|single|rlvalplus1ep_vlm_single"
  )
fi
for j in "${JOBS[@]}"; do
  IFS='|' read -r ds ckpt name prompt type log <<< "$j"
  DATASET=$ds bash scripts/run_groups_fastgpus_v2.sh "$ckpt" "$name" "$prompt" "$type" "$L/${ds}_$log"
done
echo "[$(date)] crosscheck queue $1 done"
