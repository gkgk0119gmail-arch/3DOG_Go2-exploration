#!/bin/bash
# Run experiment configs x scenes with the kinematic engine (6 seeds each by default).
#   scripts/pose_inspection/run_kinematic_batch.sh "pose planar pose_geoprune pose_tilt0 planar_tilt0" "A_warehouse B_construction"
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
for scene in $2; do
  for exp in $1; do
    $PY scripts/pose_inspection/run_kinematic.py --exp exp/$exp.yaml --scene $scene 2>&1 | grep -v "^Warning" | grep -vE "^\s+t=" 
  done
done
