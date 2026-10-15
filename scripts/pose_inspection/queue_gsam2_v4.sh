#!/bin/bash
# After process $1: keep the perception v3 construction results under a version name, then perception v4 (mask depth
# gate) on both scenes.
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
R=results/pose_inspection/isaac
[ -d $R/B_construction/pose_gsam2 ] && mv $R/B_construction/pose_gsam2 $R/B_construction/pose_gsam2_v3
[ -f logs/pose_inspection/isaac_B_construction_pose_gsam2.log ] && \
  cp logs/pose_inspection/isaac_B_construction_pose_gsam2.log logs/pose_inspection/isaac_B_construction_pose_gsam2_v3.log
scripts/pose_inspection/run_isaac_batch.sh pose_gsam2 "A_warehouse B_construction"
echo "master queue finished $(date +%H:%M:%S)"
