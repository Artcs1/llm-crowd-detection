#!/bin/bash
# Waits until results/predictions/EgoGroups_test/{results,results_full} are writable from this machine, then runs the
# EgoGroups test of the valsplit SFT checkpoint on its own vLLM server (port 8068, capped at 45% -- the GPUs are
# shared with the MINGLE eval) on the first GPU that frees up: GPU 5 once its JRDB-train eval job is done
# (jrdb_jrdbtrain_eval_chain.log "test 8066 finished"), or GPU 6 once the valsplit JRDB test there is done
# (egogroups_valsplit_eval_chain.log "all done").
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
export VLLM_USE_FLASHINFER_SAMPLER=0
LOGS=sft_output/logs/egogroups_valsplit_eval
CKPT=sft_output/qwen2.5-7b-groups-merged-nodspy-egogroups-train-valsplit
NAME=Qwen/Qwen2.5-7B-Instruct-SFT-EgoGroupTrain-ValSplit
R=../results/predictions/EgoGroups_test
echo "[$(date)] waiting for $R/results to be writable"
until [ -w $R/results ] && [ -w $R/results_full ]; do sleep 60; done
echo "[$(date)] writable; waiting for GPU 5 or GPU 6 to free up"
GPU=
until [ -n "$GPU" ]; do
  grep -q "test 8066 finished" sft_output/logs/jrdb_jrdbtrain_eval_chain.log 2>/dev/null && GPU=5
  [ -z "$GPU" ] && grep -q "all done" sft_output/logs/egogroups_valsplit_eval_chain.log 2>/dev/null && GPU=6
  [ -z "$GPU" ] && sleep 60
done
sleep 30  # let the previous server release its memory
echo "[$(date)] starting server 8068 on GPU $GPU"
CUDA_VISIBLE_DEVICES=$GPU vllm serve "$CKPT" --served-model-name "$NAME" --api-key jrdb --port 8068 \
  --gpu-memory-utilization 0.45 > "$LOGS/8068_server.log" 2>&1 &
pid=$!
until curl -sf -H "Authorization: Bearer jrdb" "http://localhost:8068/v1/models" > /dev/null; do
  if ! kill -0 $pid 2>/dev/null; then echo "[$(date)] server died, see $LOGS/8068_server.log"; exit 1; fi
  sleep 15
done
mv $LOGS/8067_ego_test.log $LOGS/8067_ego_test_permfail.log 2>/dev/null
echo "[$(date)] running EgoGroups test on GPU $GPU"
bash scripts/run_groups.bash "$NAME" 8068 jrdb p1 single > "$LOGS/8068_ego_test.log" 2>&1
echo "[$(date)] EgoGroups test finished with exit code $?"
kill $pid; wait $pid 2>/dev/null
