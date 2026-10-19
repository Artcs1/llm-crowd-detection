#!/bin/bash
# Once GPU 0's run (8061) finishes: stop the VLM SFT full client (8067, GPU 6) and the two VLM RL full split clients
# (8068/8069, GPUs 7/3), then resume each over 4 GPUs with scripts/run_groups_jrdb_egostyle_split.sh
# (--skip_existing, so saved keyframes are kept): VLM SFT full on GPUs 6,0,1,2 and VLM RL full on GPUs 7,3,4,5.
umask 000
cd "$(dirname "$0")/.."
S=sft_output
L=sft_output/logs/jrdb_egostyle_8models
CHAIN=sft_output/logs/jrdb_egostyle_8models_chain.log

echo "[$(date)] waiting for client 8061 (GPU 0) to finish"
until grep -q "client 8061 finished" $CHAIN; do sleep 30; done

echo "[$(date)] stopping clients 8067, 8068, 8069 (their servers stop with them)"
pkill -f "batch_fetch_groups.py.*localhost:806[789]/v1"
for g in 0 3 6 7; do
  until [ "$(nvidia-smi -i $g --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 5; done
done
echo "[$(date)] GPUs 0, 3, 6, 7 free"
ts=$(date +%H%M)
for p in 8067 8068 8069; do
  for f in client server; do [ -f $L/${p}_$f.log ] && mv $L/${p}_$f.log $L/${p}_${f}_before_resplit_$ts.log; done
done

bash scripts/run_groups_jrdb_egostyle_split.sh $S/qwen2.5-vl-7b-groups-merged-nodspy-full-egogroups-train \
  Qwen/Qwen2.5-VL-7B-Instruct-SFT-EgoGroupTrain p1_visual full 6:8067 0:8070 1:8071 2:8072 \
  > sft_output/logs/jrdb_egostyle_split_8067_chain.log 2>&1 &
bash scripts/run_groups_jrdb_egostyle_split.sh $S/qwen2.5-vl-7b-groups-grpo-merged-fourlosses-full-egogroups-train-1ep \
  Qwen/Qwen2.5-VL-7B-Instruct-RL-EgoGroupTrain p1_visual full 7:8068 3:8069 4:8073 5:8074 \
  > sft_output/logs/jrdb_egostyle_split_8068_4x_chain.log 2>&1 &
echo "[$(date)] launched both 4-GPU splits"
wait
echo "[$(date)] both VLM full splits done"
