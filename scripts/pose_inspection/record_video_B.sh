#!/bin/bash
# Re-record the construction-site chase video (waits for process $1).
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
rm -rf results/pose_inspection/isaac/B_construction/video_pose
$HOME/miniconda3/envs/isaac/bin/python -u scripts/pose_inspection/run_isaac.py --exp exp/pose.yaml --scene B_construction \
  --seeds 0 --name video_pose --video_out media/pose_inspection/isaac_B_construction_pose.mp4 --video_max_s 420 \
  > logs/pose_inspection/video_B_construction.log 2>&1
grep -E "wrote|^done|Traceback" logs/pose_inspection/video_B_construction.log | tail -3
