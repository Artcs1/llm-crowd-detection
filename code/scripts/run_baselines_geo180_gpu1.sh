#!/usr/bin/env bash
# Retrain the EgoGroups_train baselines with the geo180 validation split (6 clips per region and density,
# llm-crowd-detection/code/egogroups_val_geo180.txt) on GPU $GPU, all in parallel on that GPU: each job starts once the
# GPU has >= $NEED_MIB free and the previous job had $RAMP s to reach its memory. Same settings as the paper runs; new
# output folders end in _geo180 (the density180 runs keep theirs). RTGD runs on CPU separately (RTGD/runs/train_ego_geo180.log).
# Log: llm-crowd-detection/code/sft_output/logs/baselines_geo180_gpu1.log
#   DVT3       VLM-GroupDetection/run_egogroups_train.sh  -> EgoGroups_train_run_geo180, ckpt/egogroups_train_geo180
#   Social-CAD SocialGroups_ECCV20 sg.train --head eccv20  -> runs/ego_geo180
#   JRDB-Act   SocialGroups_ECCV20 sg.train --head jrdbact -> runs/jrdbact_ego_geo180
#   S3R2       SHGD/logs/train_ego_geo180.sh               -> SHGD/runs/ego_geo180
#   P-HAR      PAR train_egogroups.py, weight decay 1e-3   -> result/[EgoGroups_geo180_wd1e-3]<date>
#   GroupTrans prepare_setup + features + jrdb_train_ours  -> GroupTrans/runs/ego_train_geo180
# Usage: GPU=1 setsid nohup bash llm-crowd-detection/code/scripts/run_baselines_geo180_gpu1.sh > /dev/null 2>&1 &
export EGO_VAL_SPLIT=geo180
GPU=${GPU:-1} NEED_MIB=${NEED_MIB:-30000} RAMP=${RAMP:-900}
R=/data/paola-shared/jmurrugarral
LOG=$R/llm-crowd-detection/code/sft_output/logs/baselines_geo180_gpu1.log
E=$HOME/anaconda3/envs
log() { echo "[$(date)] $*" >> "$LOG"; }
free_mib() { nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '; }
start() {  # name dir command...
    local name=$1 dir=$2; shift 2
    until [ "$(free_mib)" -ge "$NEED_MIB" ]; do sleep 60; done
    (cd "$dir" && setsid nohup bash -c "$*; echo \"[\$(date)] $name exit \$?\" >> $LOG" > /dev/null 2>&1 < /dev/null &)
    log "$name started on GPU $GPU ($(free_mib) MiB free): $dir: $*"
    sleep "$RAMP"
}
log "queue start"
start DVT3 $R/VLM-GroupDetection "GPU=$GPU bash run_egogroups_train.sh > EgoGroups_train_run_geo180_driver.log 2>&1"
start Social-CAD $R/SocialGroups_ECCV20 "$E/DVT3/bin/python -m sg.train train --data ego --out runs/ego --gpu $GPU --workers 8 > runs/logs/train_ego_geo180.log 2>&1"
start JRDB-Act $R/SocialGroups_ECCV20 "$E/DVT3/bin/python -m sg.train train --head jrdbact --data ego --out runs/jrdbact_ego --gpu $GPU --workers 8 > runs/logs/train_jrdbact_ego_geo180.log 2>&1"
start S3R2 $R/SHGD "GPU=$GPU bash logs/train_ego_geo180.sh > logs/train_ego_geo180.out 2>&1"
start P-HAR $R/PAR/PAR-main/PanoAct_source-code-12_23 "PAR_WD=1e-3 $E/PAR/bin/python train_egogroups.py $GPU > logs/train_egogroups_geo180_wd1e-3.log 2>&1"
start GroupTrans $R/GroupTrans "$E/DVT3/bin/python egotools/prepare_setup.py ego_train > runs/prepare_ego_train_geo180.log 2>&1 && cd runs/ego_train_geo180 && $E/DVT3/bin/python ../../egotools/extract_features.py $GPU > extract.log 2>&1 && $E/DVT3/bin/python ../../jrdb_train_ours.py --gpu $GPU --taskname ego_traj_with_app --appearance --refresh_ana 1 > train.log 2>&1"
log "all jobs started"
