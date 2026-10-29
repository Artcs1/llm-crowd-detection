#!/bin/bash
# DBSCAN (naive_cluster.py, min_samples 1) on JRDB test (JRDB_fixed_gold, 1,829 keyframes) and the 7 JRDB val scenes
# (JRDB_val_fixed_gold, 372 keyframes, to pick eps), on the shards' lidar 3D positions, 2D (x, y = ground plane) or 3D.
# Names DBSCAN{2D,3D}_t<eps>, like scripts/run_dbscan_egogroups.sh. CPU only. Scoring: scripts/score_dbscan_jrdb.py.
#   P=8 bash scripts/run_dbscan_jrdb.sh
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
P=${P:-8}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
LOGS=sft_output/logs/dbscan_jrdb
mkdir -p "$LOGS"
one() {  # <set> <dims> <eps>
  local D=$1 name=DBSCAN${2}D_t$3
  python3 naive_cluster.py "../../$D/jsons_gold" --frame_id 15 --threshold "$3" --distance_dims "$2" \
      --frame_path "../../$D/videos_frames" --model "baseline/$name" > "$LOGS/${D}_$name.log" 2>&1
  python3 compute_detections.py --dataset "$D" --mode single --depth_method detany_3D --prompt_method naive_cluster \
      --model "$name" --vlm_mode llm --frame_id 15 >> "$LOGS/${D}_$name.log" 2>&1
  echo "[$(date)] $D $name done"
}
export -f one; export LOGS
for D in JRDB_fixed_gold JRDB_val_fixed_gold; do for dims in 2 3; do for t in 0.2 0.4 0.6 0.9 1.2 1.5 1.8 2.1 2.4; do
  echo "$D $dims $t"; done; done; done | xargs -P "$P" -L 1 bash -c 'one $0 $1 $2'
echo "[$(date)] all JRDB DBSCAN runs done"
