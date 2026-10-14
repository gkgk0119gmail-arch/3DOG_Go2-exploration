#!/bin/bash
# Run the remaining Isaac experiments one after another once process $1 (an earlier queue) has exited.
#   scripts/pose_inspection/queue_master.sh <pid> "<exp list>|<scene list>" ["<exp list>|<scene list>" ...]
cd "$(dirname "$0")/../.."
pid=$1; shift
while kill -0 "$pid" 2>/dev/null; do sleep 30; done
for job in "$@"; do
  exps=${job%%|*}; scenes=${job##*|}
  echo "== job [$exps] on [$scenes] $(date +%H:%M:%S)"
  scripts/pose_inspection/run_isaac_batch.sh "$exps" "$scenes"
done
echo "master queue finished $(date +%H:%M:%S)"
