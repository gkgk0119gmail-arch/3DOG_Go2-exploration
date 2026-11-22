#!/bin/bash
# KRoC 2027: fair zero-shot baseline (Isaac) in parallel with the controlled false-detection ablation (kinematic),
# then fine-tuning with false detections and its evaluation. Each finished stage is committed and pushed.
cd "$(dirname "$0")/../.."
export PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
push() {
  ( flock 9
    for p in results/pose_inspection/isaac/*/pose_kf_gsam2 results/pose_inspection/isaac/*/rl_run4_fd_gsam2 \
             results/pose_inspection/kinematic/*/gt_* results/pose_inspection/kinematic/*/fd*_* \
             results/pose_inspection/rl/run4_fd/log.csv results/pose_inspection/rl/policies/run4_fd.pt; do
      [ -e "$p" ] && git add -A "$p"
    done
    git commit -qm "$1" && git push -q -u origin kroc-results
  ) 9>/tmp/kroc_git.lock
  echo "pushed: $1  $(date +%H:%M:%S)"
}
# A: fair zero-shot baseline (Isaac, about an hour per scene)
( scripts/pose_inspection/run_isaac_batch.sh pose_kf_gsam2 "A_warehouse B_construction"
  push "kroc: fair zero-shot baseline pose_kf_gsam2 (Isaac)" ) &
# B: controlled false detections + selector ablation (kinematic), then fine-tuning with false detections
scripts/pose_inspection/abl_kinematic.sh
push "kroc: controlled false-detection ablation (kinematic)"
wait
$PY scripts/pose_inspection/rl_train.py --config rl_fd.yaml --name run4_fd --init results/pose_inspection/rl/policies/run3_it250.pt
cp results/pose_inspection/rl/run4_fd/ckpt_0200.pt results/pose_inspection/rl/policies/run4_fd.pt
scripts/pose_inspection/abl_kinematic.sh rl_fd
push "kroc: run4_fd (fine-tuned with false detections), kinematic evaluation"
scripts/pose_inspection/run_isaac_batch.sh rl_run4_fd_gsam2 "A_warehouse B_construction"
push "kroc: run4_fd with zero-shot perception (Isaac)"
echo "kroc queue finished $(date +%H:%M:%S)"
