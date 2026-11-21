#!/usr/bin/env bash
# Run from any directory in the existing CUDA Python environment.
# Example: PYTHON=python bash run_server.sh train --all --fold 0 --canon /data/preprocessed
set -euo pipefail
FQW_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
FQW_PYTHON="${PYTHON:-python}"
FQW_LOG_DIR="${EXP10_LOG_DIR:-$FQW_ROOT/outputs/exp10_shared_matrix/logs}"
mkdir -p "$FQW_LOG_DIR"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
FQW_LOG="$FQW_LOG_DIR/$(date -u +%Y%m%dT%H%M%SZ)_$$.log"
cd "$FQW_ROOT"
"$FQW_PYTHON" -u code/scripts/exp10_shared_matrix/run_experiment.py "$@" 2>&1 | tee "$FQW_LOG"
