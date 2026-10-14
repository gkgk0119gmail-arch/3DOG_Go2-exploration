#!/bin/bash
# Run experiment configs x scenes in Isaac Lab, one process per (scene, config), 6 seeds each by default.
#   scripts/pose_inspection/run_isaac_batch.sh "pose planar" "A_warehouse B_construction" [extra args...]
# Skips a (scene, config) whose results already exist (delete the folder to rerun).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
exps=$1; scenes=$2; shift 2
mkdir -p logs/pose_inspection
for scene in $scenes; do
  for exp in $exps; do
    out=results/pose_inspection/isaac/$scene/$exp
    if ls $out/seed*.json >/dev/null 2>&1; then echo "skip $scene/$exp (exists)"; continue; fi
    echo "== $scene / $exp  $(date +%H:%M:%S)"
    $PY -u scripts/pose_inspection/run_isaac.py --exp exp/$exp.yaml --scene $scene "$@" \
      > logs/pose_inspection/isaac_${scene}_${exp}.log 2>&1
    grep -E "^done:|seed [0-9]+:|Traceback|Error" logs/pose_inspection/isaac_${scene}_${exp}.log | tail -9
  done
done
echo "batch finished $(date +%H:%M:%S)"
