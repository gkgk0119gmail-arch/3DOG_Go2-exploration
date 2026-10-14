#!/bin/bash
# After the running stage-4/5 batch: redo scene B (gsam2 OOM fix, VLM verdict rules), then record videos.
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
while pgrep -f run_isaac_batch.sh >/dev/null; do sleep 30; done
scripts/pose_inspection/run_isaac_batch.sh "pose_gsam2 pose_vlm" "B_construction"
for scene in A_warehouse B_construction; do
  echo "== video $scene $(date +%H:%M:%S)"
  $PY -u scripts/pose_inspection/run_isaac.py --exp exp/pose.yaml --scene $scene --seeds 0 --name video_pose \
    --video_out media/pose_inspection/isaac_${scene}_pose.mp4 --video_max_s 420 \
    > logs/pose_inspection/video_${scene}.log 2>&1
  grep -E "wrote|^done|Traceback" logs/pose_inspection/video_${scene}.log | tail -3
done
echo "queue finished $(date +%H:%M:%S)"
