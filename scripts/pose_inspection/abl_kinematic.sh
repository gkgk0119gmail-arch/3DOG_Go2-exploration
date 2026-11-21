#!/bin/bash
# Controlled false-detection experiment + selector ablation in the kinematic engine (6 seeds per run).
#   scripts/pose_inspection/abl_kinematic.sh [extra method ...]      e.g. abl_kinematic.sh rl_fd
# Every method runs with the same known-free viewpoint restriction and sees exactly the same false hypotheses:
#   B_construction: the four clutter objects reported as machines; A_warehouse: 5 / 10 / 20 rack or wall segments.
# Results: results/pose_inspection/kinematic/<scene>/<cond>_<method>/seed*.json (existing runs are skipped).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
declare -A EXP=([pose]=pose_kf [pose_prune]=pose_kf_prune [nearest]=sel_nearest [nearest_prune]=sel_nearest_prune
                [gain_prune]=sel_gain_prune [rl]=rl_run3_it250 [rl_fd]=rl_run4_fd)
run() {  # scene cond method [override ...]
  local scene=$1 cond=$2 m=$3; shift 3
  local out=results/pose_inspection/kinematic/$scene/${cond}_$m
  if ls $out/seed*.json >/dev/null 2>&1; then echo "skip $scene/${cond}_$m (exists)"; return; fi
  echo "== $scene / ${cond}_$m  $(date +%H:%M:%S)"
  $PY scripts/pose_inspection/run_kinematic.py --exp exp/${EXP[$m]}.yaml --scene $scene --name ${cond}_$m \
      --set experiment.n_seeds=6 "$@" 2>&1 | grep -E "^done|seed [0-9]+:|Traceback|Error"
}
for m in rl pose gain_prune nearest_prune nearest pose_prune "$@"; do
  run B_construction gt $m
  run B_construction fd $m perception.false_detections.clutter=all
  run A_warehouse gt $m
  for k in 5 10 20; do run A_warehouse fd$k $m perception.false_detections.boxes=$k; done
done
echo "ablation finished $(date +%H:%M:%S)"
