#!/usr/bin/env bash
# Server: run inside tmux. Stages: setup, data, check, pilot, run, summary.
# Defaults: raw WITHOUT ICA; one training at a time for a 16 GB GPU.
set -euo pipefail
FQW="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
STAGE="${1:?Usage: bash matched22.sh setup|data|check|pilot|run|summary}"
if [ "$STAGE" = setup ]; then
  bash "$FQW/code/scripts/exp08_09_full_lisa/vast/setup.sh"
  exit
fi
for d in /opt/miniforge3 /opt/miniforge; do
  if [ -x "$d/bin/conda" ]; then export PATH="$d/bin:$PATH"; break; fi
done
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate LISA
cd "$FQW"
mkdir -p "$FQW/logs/matched22"
case "$STAGE" in
  data)
    # Run again safely after an interrupted download. Conversion is required for LISA.
    python -u code/scripts/exp08_09_full_lisa/vast/download_nemar.py --workers "${DOWNLOAD_WORKERS:-8}" 2>&1 | tee "$FQW/logs/matched22/download.log"
    python code/scripts/exp08_09_full_lisa/vast/nemar_events_to_osf.py
    bash code/scripts/exp08_09_full_lisa/vast/prepare.sh raw "${PREP_JOBS:-4}" 2>&1 | tee "$FQW/logs/matched22/prepare.log"
    ;;
  check) python -u code/scripts/exp08_matched22/run_matched22.py check 2>&1 | tee "$FQW/logs/matched22/check.log" ;;
  pilot) python -u code/scripts/exp08_matched22/run_matched22.py run --seed 42 --procs 1 2>&1 | tee -a "$FQW/logs/matched22/series.log" ;;
  run) python -u code/scripts/exp08_matched22/run_matched22.py run --procs 1 2>&1 | tee -a "$FQW/logs/matched22/series.log" ;;
  summary) python -u code/scripts/exp08_matched22/run_matched22.py summarize ;;
  *) printf 'Unknown stage: %s\n' "$STAGE" >&2; exit 2 ;;
esac
