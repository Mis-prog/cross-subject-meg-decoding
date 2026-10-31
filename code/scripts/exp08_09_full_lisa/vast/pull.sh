#!/usr/bin/env bash
# С ноутбука: забрать результаты серий — метрики, конфиги, ранги по окнам, сводки — и веса
# только завершённых обучений (есть filters_*.npz, который тренер пишет последним).
# Веса идущих прогонов, фильтры, логи и служебные папки остаются на сервере.
# Использование: bash code/scripts/exp08_09_full_lisa/vast/pull.sh <port> <host>
set -euo pipefail
PORT="$1"; HOST="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
R="root@$HOST:/workspace/FQW"
SSH="ssh -p $PORT -o ConnectTimeout=20"
FULL_RESULTS="$ROOT/experiments/exp08_09_full_lisa/results"
MATCHED_RESULTS="$ROOT/experiments/exp08_matched22/results"
mkdir -p "$ROOT/outputs" "$FULL_RESULTS" "$MATCHED_RESULTS"
rsync -az -e "$SSH" --prune-empty-dirs \
  --exclude 'dirprocess/' --exclude 'locks/' --exclude 'logs/' --exclude 'runs_failed/' --exclude 'filters/' \
  --include '*/' --include 'metrics.csv' --include 'config.json' --include 'final_test_per_window.csv' \
  --include 'eval/**' --exclude '*' \
  "$R/outputs/" "$ROOT/outputs/"
# Accept both the organized layout and results from servers running the earlier code.
if $SSH "root@$HOST" 'test -d /workspace/FQW/experiments/exp08_09_full_lisa/results'; then
  rsync -az -e "$SSH" "$R/experiments/exp08_09_full_lisa/results/" "$FULL_RESULTS/" || true
else
  rsync -az -e "$SSH" --exclude '/full_matched22/' --include 'full*/***' --exclude '*' \
    "$R/experiments/results/" "$FULL_RESULTS/" || true
fi
if $SSH "root@$HOST" 'test -d /workspace/FQW/experiments/exp08_matched22/results'; then
  rsync -az -e "$SSH" "$R/experiments/exp08_matched22/results/" "$MATCHED_RESULTS/" || true
else
  rsync -az -e "$SSH" "$R/experiments/results/full_matched22/" "$MATCHED_RESULTS/" || true
fi
for run in $($SSH "root@$HOST" 'cd /workspace/FQW/outputs && ls full*/runs/*/*/filters_*.npz 2>/dev/null | xargs -r -n1 dirname' 2>/dev/null); do
  mkdir -p "$ROOT/outputs/$run"
  rsync -az -e "$SSH" --include '*.pt' --exclude '*' "$R/outputs/$run/" "$ROOT/outputs/$run/"
done
echo "pulled $(date '+%H:%M:%S'): $(du -sh "$ROOT/outputs" | cut -f1) in outputs/"
