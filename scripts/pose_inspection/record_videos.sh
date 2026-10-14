#!/bin/bash
# Chase-camera recordings of one POSE run per scene (waits for process $1 first).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
pid=$1
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
for scene in A_warehouse B_construction; do
  echo "== video $scene $(date +%H:%M:%S)"
  $PY -u scripts/pose_inspection/run_isaac.py --exp exp/pose.yaml --scene $scene --seeds 0 --name video_pose \
    --video_out media/pose_inspection/isaac_${scene}_pose.mp4 --video_max_s 420 > logs/pose_inspection/video_${scene}.log 2>&1
  grep -E "wrote|^done|Traceback" logs/pose_inspection/video_${scene}.log | tail -3
done
echo "videos finished $(date +%H:%M:%S)"
