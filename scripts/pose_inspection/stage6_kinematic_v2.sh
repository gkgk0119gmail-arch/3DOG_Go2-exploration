#!/bin/bash
# Re-run every template / CAD kinematic run with the final gain settings (waits for process $1).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
while kill -0 "$1" 2>/dev/null; do sleep 30; done
rm -rf results/pose_inspection/kinematic/*/prior_template results/pose_inspection/kinematic/*/prior_cad \
       results/pose_inspection/kinematic/*/sweep_template_* results/pose_inspection/kinematic/*/sweep_cad_*
scripts/pose_inspection/run_kinematic_batch.sh "prior_template prior_cad" "B_construction A_warehouse"
for scene in B_construction A_warehouse; do
  for m in template cad; do
    for thr in 100 200 400 800; do
      $PY scripts/pose_inspection/run_kinematic.py --exp exp/prior_${m}_geoprune.yaml --scene $scene \
        --set planner.semantic.prune_min_gain_fine=$thr --name sweep_${m}_$thr 2>&1 | grep -E "^\[|^done|Traceback"
    done
  done
done
echo "stage6 v2 finished $(date +%H:%M:%S)"
