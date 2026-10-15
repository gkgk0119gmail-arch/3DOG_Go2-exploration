#!/bin/bash
# Run experiment configs x scenes in Isaac Lab, one process per (scene, config), 6 seeds each by default.
#   scripts/pose_inspection/run_isaac_batch.sh "pose planar" "A_warehouse B_construction" [extra args...]
# Skips a (scene, config) whose results already exist (delete the folder to rerun). A run whose log is still
# (nearly) empty after START_TIMEOUT s is a Kit start-up deadlock: it is killed and started again (up to 3 tries).
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
START_TIMEOUT=${START_TIMEOUT:-600}
exps=$1; scenes=$2; shift 2
mkdir -p logs/pose_inspection
for scene in $scenes; do
  for exp in $exps; do
    out=results/pose_inspection/isaac/$scene/$exp
    if ls $out/seed*.json >/dev/null 2>&1; then echo "skip $scene/$exp (exists)"; continue; fi
    log=logs/pose_inspection/isaac_${scene}_${exp}.log
    for try in 1 2 3; do
      echo "== $scene / $exp  $(date +%H:%M:%S)  (try $try)"
      $PY -u scripts/pose_inspection/run_isaac.py --exp exp/$exp.yaml --scene $scene "$@" > $log 2>&1 &
      pid=$!
      t0=$SECONDS; hung=0
      while kill -0 $pid 2>/dev/null; do
        sleep 20
        if [ $((SECONDS - t0)) -gt $START_TIMEOUT ] && [ "$(stat -c %s $log)" -lt 2000 ]; then
          echo "   start-up hang (log $(stat -c %s $log) B after $START_TIMEOUT s): killing $pid"
          kill $pid; sleep 10; kill -9 $pid 2>/dev/null; hung=1; break
        fi
      done
      wait $pid 2>/dev/null
      [ $hung -eq 0 ] && break
    done
    grep -E "^done:|seed [0-9]+:|Traceback|Error" $log | tail -9
  done
done
echo "batch finished $(date +%H:%M:%S)"
