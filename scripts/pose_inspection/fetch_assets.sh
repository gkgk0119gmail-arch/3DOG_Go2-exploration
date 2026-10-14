#!/bin/bash
# Download the Isaac Sim 6.1 assets the POSE-inspection scenes need into the same local mirror the warehouse uses
# (assets/warehouse/https/..., git-ignored). Safe to re-run: existing files are skipped.
set -euo pipefail
SIM=$(cd "$(dirname "$0")/../.." && pwd)
HOST=omniverse-content-production.s3-us-west-2.amazonaws.com
MIRROR="$SIM/assets/warehouse/https/$HOST"
for PREFIX in "$@"; do
  keys=$(curl -s "https://$HOST/?list-type=2&prefix=$PREFIX&max-keys=1000" | grep -o '<Key>[^<]*</Key>' | sed 's/<[^>]*>//g' | grep -v '/\.thumbs/')
  for k in $keys; do
    out="$MIRROR/$k"
    [ -s "$out" ] && continue
    mkdir -p "$(dirname "$out")"
    curl -s --fail -o "$out" "https://$HOST/$k" && echo "got $k"
  done
done
