#!/bin/bash
# GPU 0 requeue: when the EgoGroups LLM full RL (pid $RL_PID, child of the EgoGroups chain $CHAIN_PID) exits, stop that
# chain before its VLM single SFT gets going, then run on GPU 0: the JRDB VLM RL-on-val g4p2 no-bias-KL run (moved here
# from the GPU 1 queue), then the g8p4 run (also moved from GPU 1), then the remaining EgoGroups VLM pipelines (scripts/run_ego_val_split_sft_rl_vlm.sh).
# Usage (inside tmux): RL_PID=1365215 CHAIN_PID=3960421 bash scripts/run_gpu0_requeue_g4p2.sh
cd "$(dirname "$0")/.."
L=sft_output/logs/ego_val_split
echo "[$(date)] waiting for the EgoGroups LLM full RL (pid $RL_PID) to exit"
while kill -0 "$RL_PID" 2>/dev/null; do sleep 1; done
kids=$(ps -o pid= --ppid "$CHAIN_PID" 2>/dev/null)
kill -9 "$CHAIN_PID" 2>/dev/null  # SIGKILL: bash would defer SIGTERM until its foreground child (the new SFT) exits
for k in $kids; do kill "$k" 2>/dev/null; done; sleep 20; for k in $kids; do kill -9 "$k" 2>/dev/null; done
echo "[$(date)] RL llm_full exited; stopped the EgoGroups chain $CHAIN_PID (children: ${kids:-none})"
[ -f "$L/sft_vlm_single.log" ] && mv "$L/sft_vlm_single.log" "$L/sft_vlm_single_aborted_requeue.log"
grep -E "Saving|merged|saved" "$L/rl_llm_full.log" | tail -2
GPU=0 bash scripts/run_jrdb_rl_on_val_vlm_nobiaskl_g4p2_retry.sh
GPU=0 bash scripts/run_jrdb_rl_on_val_vlm_nobiaskl_g8p4_gpu0.sh
GPU=0 bash scripts/run_ego_val_split_sft_rl_vlm.sh
echo "[$(date)] GPU 0 requeue done"
