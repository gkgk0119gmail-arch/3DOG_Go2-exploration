#!/bin/bash
# Coverage-time trade-off of geometric pruning: prune threshold sweep per region type (kinematic engine).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
for scene in B_construction A_warehouse; do
  for thr in 5 10 20 40; do
    $PY scripts/pose_inspection/run_kinematic.py --exp exp/pose_geoprune.yaml --scene $scene \
      --set planner.semantic.prune_min_gain=$thr --name sweep_box_$thr 2>&1 | grep -E "^\[|^done|Traceback"
  done
  for m in template cad; do
    for thr in 100 200 400 800; do
      $PY scripts/pose_inspection/run_kinematic.py --exp exp/prior_${m}_geoprune.yaml --scene $scene \
        --set planner.semantic.prune_min_gain_fine=$thr --name sweep_${m}_$thr 2>&1 | grep -E "^\[|^done|Traceback"
    done
  done
done
echo sweep finished
