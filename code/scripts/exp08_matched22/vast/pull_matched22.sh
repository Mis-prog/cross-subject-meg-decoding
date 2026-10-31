#!/usr/bin/env bash
# Laptop: regular results/checkpoints, plus the matched-control audit and timing logs.
set -euo pipefail
PORT="${1:?Usage: bash pull_matched22.sh PORT HOST}"
HOST="${2:?Usage: bash pull_matched22.sh PORT HOST}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
bash "$ROOT/code/scripts/exp08_09_full_lisa/vast/pull.sh" "$PORT" "$HOST"
mkdir -p "$ROOT/outputs/full_matched22" "$ROOT/logs/matched22"
rsync -az -e "ssh -p $PORT" --include '/matched22_inputs.json' --include '/timings.jsonl' \
  --include '/logs/***' --exclude '*' \
  "root@$HOST:/workspace/FQW/outputs/full_matched22/" "$ROOT/outputs/full_matched22/"
rsync -az -e "ssh -p $PORT" \
  "root@$HOST:/workspace/FQW/logs/matched22/" "$ROOT/logs/matched22/"
