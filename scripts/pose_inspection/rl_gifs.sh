#!/bin/bash
# Isaac POSE seed 1 with trajectories (stage-3 runs predate trajectory logging), then POSE-vs-RL GIFs per scene.
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
for s in B_construction A_warehouse; do
  [ -f results/pose_inspection/isaac/$s/pose_traj/seed1.json ] || \
    $PY -u scripts/pose_inspection/run_isaac.py --exp exp/pose.yaml --scene $s --seeds 1 --name pose_traj \
      > logs/pose_inspection/isaac_${s}_pose_traj.log 2>&1
done
$PY scripts/pose_inspection/make_run_gif.py results/pose_inspection/isaac/B_construction/pose_traj/seed1.json \
  results/pose_inspection/isaac/B_construction/rl_run1_it120/seed1.json --labels "POSE (rule-based)" "RL head" \
  --out media/pose_inspection/construction_pose_vs_rl.gif --step 3
$PY scripts/pose_inspection/make_run_gif.py results/pose_inspection/isaac/A_warehouse/pose_traj/seed1.json \
  results/pose_inspection/isaac/A_warehouse/rl_run1_it120/seed1.json --labels "POSE (rule-based)" "RL head" \
  --out media/pose_inspection/warehouse_pose_vs_rl.gif --step 5 --width 4.5
echo "gifs finished"
