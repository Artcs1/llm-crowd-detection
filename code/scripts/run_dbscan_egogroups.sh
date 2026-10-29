#!/bin/bash
# DBSCAN (naive_cluster.py: min_samples 1 = single-linkage on detany_3D positions, 2D x/y or 3D x/y/z) on every
# EgoGroups test set, plus the val clips of each split to pick the threshold (eps) without looking at the test sets.
#   tests: EgoGroups_test, EgoGroupsCity_Test, EgoGroupsOODCity_Test, EgoGroupsOODRegions_Test,
#          EgoGroupsBenchmark2_IDTest, EgoGroupsBenchmark2_OODTest
#   val:   EgoGroups_train (geo180 val), EgoGroupsCity_Train (city val), EgoGroupsBenchmark2_Train (val),
#          EgoGroupsBenchmark3_Train (id_val)
# Names DBSCAN{2D,3D}_t<eps> (results/{predictions,detection_files,grouping_files}/<set>...); the Oct-7 NaiveCluster*
# EgoGroups_test files are left alone. CPU only. Scoring: scripts/score_dbscan_egogroups.py.
#   P=8 bash scripts/run_dbscan_egogroups.sh
# Ground plane of detany_3D (x, z; names DBSCANxz_t<eps>): DIMS=xz EPS='0.2 0.4 0.6 0.9 1.2 1.5 1.8 2.1 2.4 3.0' bash ...
umask 000
source ~/anaconda3/etc/profile.d/conda.sh
conda activate py10-dspy
cd "$(dirname "$0")/.."
P=${P:-8}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4  # P runs share the CPUs with other users
SETS="EgoGroups_test EgoGroupsCity_Test EgoGroupsOODCity_Test EgoGroupsOODRegions_Test EgoGroupsBenchmark2_IDTest
      EgoGroupsBenchmark2_OODTest EgoGroups_train EgoGroupsCity_Train EgoGroupsBenchmark2_Train EgoGroupsBenchmark3_Train"
LOGS=sft_output/logs/dbscan_egogroups
mkdir -p "$LOGS"
one() {  # <set> <dims> <eps>
  local D=$1 dims=$2 t=$3 J=jsons_step6 name=DBSCAN${2}D_t$3
  [ "$dims" = xz ] && name=DBSCANxz_t$3
  [ "$D" = EgoGroups_train ] && J=jsons_step4  # EgoGroups_train has no jsons_step6 (same boxes / detany_3D in step4)
  python3 naive_cluster.py "../../$D/$J" --frame_id 10 --threshold "$t" --distance_dims "$dims" \
      --frame_path "../../$D/videos_frames" --model "baseline/$name" > "$LOGS/${D}_$name.log" 2>&1
  python3 compute_detections.py --dataset "$D" --mode single --depth_method detany_3D --prompt_method naive_cluster \
      --model "$name" --vlm_mode llm --frame_id 10 >> "$LOGS/${D}_$name.log" 2>&1
  echo "[$(date)] $D $name done"
}
export -f one; export LOGS
DIMS=${DIMS:-2 3} EPS=${EPS:-0.2 0.4 0.6 0.9 1.2 1.5 1.8}
for D in $SETS; do for dims in $DIMS; do for t in $EPS; do echo "$D $dims $t"; done; done; done |
  xargs -P "$P" -L 1 bash -c 'one $0 $1 $2'
echo "[$(date)] all DBSCAN runs done"
