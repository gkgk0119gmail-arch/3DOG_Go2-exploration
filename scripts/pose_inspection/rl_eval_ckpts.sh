#!/bin/bash
# Evaluate several checkpoints of an RL run (validation scenes + hand-built scenes) after process $1 has exited.
#   scripts/pose_inspection/rl_eval_ckpts.sh <pid> <run dir> <it> [<it> ...]
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
pid=$1; run=$2; shift 2
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
for it in "$@"; do
  c=$(printf "%s/ckpt_%04d.pt" "$run" "$it")
  $PY scripts/pose_inspection/rl_eval.py "$c" --val 10 --envs 4 --handmade --out "$run/eval_ckpt$(printf %04d "$it")" \
    2>&1 | grep -E "^\| \*\*mean|A_warehouse \||B_construction \|"
done
echo "eval finished $(date +%H:%M:%S)"
