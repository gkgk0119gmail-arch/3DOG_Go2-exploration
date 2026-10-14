#!/bin/bash
# After the VLM v4 warehouse batch (process $1): keep the v4 results under a version name, then VLM v5 (measured
# green fraction), the Isaac stage-6 priors and perception v3.
cd "$(dirname "$0")/../.."
while kill -0 "$1" 2>/dev/null; do sleep 30; done
R=results/pose_inspection/isaac
for s in A_warehouse B_construction; do
  [ -d $R/$s/pose_vlm ] && mv $R/$s/pose_vlm $R/$s/pose_vlm_v4_model_green
  [ -f logs/pose_inspection/isaac_${s}_pose_vlm.log ] && mv logs/pose_inspection/isaac_${s}_pose_vlm.log \
    logs/pose_inspection/isaac_${s}_pose_vlm_v4.log
done
for job in "pose_vlm|B_construction A_warehouse" \
           "prior_template prior_cad prior_template_vlm|B_construction A_warehouse" \
           "pose_gsam2|A_warehouse B_construction"; do
  exps=${job%%|*}; scenes=${job##*|}
  echo "== job [$exps] on [$scenes] $(date +%H:%M:%S)"
  scripts/pose_inspection/run_isaac_batch.sh "$exps" "$scenes"
done
echo "master queue finished $(date +%H:%M:%S)"
