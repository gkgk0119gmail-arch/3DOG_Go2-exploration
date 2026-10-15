#!/bin/bash
# After process $1: evaluate checkpoints of a run on the validation pool + hand-built scenes and on the large sites.
#   scripts/pose_inspection/rl_eval_all.sh <pid> <run dir> <it> [<it> ...]
cd "$(dirname "$0")/../.."
PY=${PY:-$HOME/miniconda3/envs/isaac/bin/python}
pid=$1; run=$2; shift 2
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
for it in "$@"; do
  i=$(printf %04d "$it"); c="$run/ckpt_$i.pt"
  echo "== $c"
  $PY scripts/pose_inspection/rl_eval.py "$c" --val 10 --envs 4 --handmade --out "$run/eval_ckpt$i" 2>&1 | grep -E "^\| \*\*mean"
  $PY scripts/pose_inspection/rl_eval.py "$c" --pool test_site --val 6 --envs 4 --out "$run/eval_site_ckpt$i" 2>&1 | grep -E "^\| \*\*mean"
done
echo "eval finished $(date +%H:%M:%S)"
