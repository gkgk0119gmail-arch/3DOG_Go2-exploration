#!/bin/bash
# Re-run the stage-6 kinematic sweeps whose runs idled below cs_threshold until the time limit (before
# termination.stop_when_idle existed).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
for r in "template 400" "template 800" "cad 800"; do
  set -- $r
  rm -rf results/pose_inspection/kinematic/A_warehouse/sweep_$1_$2
  $PY scripts/pose_inspection/run_kinematic.py --exp exp/prior_$1_geoprune.yaml --scene A_warehouse \
    --set planner.semantic.prune_min_gain_fine=$2 --name sweep_$1_$2 2>&1 | grep -E "^\[|^done|Traceback|seed [0-9]:"
done
echo "rerun finished $(date +%H:%M:%S)"
