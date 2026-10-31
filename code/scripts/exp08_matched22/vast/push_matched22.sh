#!/usr/bin/env bash
# Laptop: upload code plus only the B checkpoints/evaluations needed by matched22.
set -euo pipefail
PORT="${1:?Usage: bash push_matched22.sh PORT HOST}"
HOST="${2:?Usage: bash push_matched22.sh PORT HOST}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
bash "$ROOT/code/scripts/exp08_09_full_lisa/vast/push.sh" "$PORT" "$HOST"
for seed in 42 43 44; do
  ssh -p "$PORT" "root@$HOST" "mkdir -p /workspace/FQW/outputs/full_raw/eval/xsB_s${seed} /workspace/FQW/outputs/full_raw/runs/xsB_s${seed}"
  rsync -az -e "ssh -p $PORT" \
    "$ROOT/outputs/full_raw/eval/xsB_s${seed}/" \
    "root@$HOST:/workspace/FQW/outputs/full_raw/eval/xsB_s${seed}/"
  rsync -az -e "ssh -p $PORT" --prune-empty-dirs \
    --include '*/' --include '*.pt' --include 'config.json' \
    --include 'final_test_per_window.csv' --exclude '*' \
    "$ROOT/outputs/full_raw/runs/xsB_s${seed}/" \
    "root@$HOST:/workspace/FQW/outputs/full_raw/runs/xsB_s${seed}/"
done
printf 'Matched22 code and B references uploaded.\n'
