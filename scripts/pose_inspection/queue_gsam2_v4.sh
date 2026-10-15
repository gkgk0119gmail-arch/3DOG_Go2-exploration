#!/bin/bash
# After process $1: perception v4 (mask depth gate) on both scenes. A construction run already made with v4 (its
# seed files carry depth_gate in their config) is kept; an older one is renamed to pose_gsam2_v3.
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
R=results/pose_inspection/isaac
if [ -f $R/B_construction/pose_gsam2/seed0.json ] && ! grep -q depth_gate $R/B_construction/pose_gsam2/seed0.json; then
  mv $R/B_construction/pose_gsam2 $R/B_construction/pose_gsam2_v3
  cp logs/pose_inspection/isaac_B_construction_pose_gsam2.log logs/pose_inspection/isaac_B_construction_pose_gsam2_v3.log
fi
scripts/pose_inspection/run_isaac_batch.sh pose_gsam2 "A_warehouse B_construction"
echo "master queue finished $(date +%H:%M:%S)"
