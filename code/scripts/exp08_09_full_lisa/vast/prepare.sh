#!/usr/bin/env bash
# На сервере: подготовка данных по пайплайну авторов, затем проверка канонических чисел.
#   bash prepare.sh clean 32   # с ICA авторов (нужен OSF)
#   bash prepare.sh raw 32     # без ICA; сырые данные можно взять с зеркала NEMAR (download_nemar.py)
# Если data/MASC-MEG уже скачан с NEMAR без ошибок, загрузка сырых данных с OSF пропускается.
set -euo pipefail
MODE="${1:-clean}"
N_JOBS="${2:-8}"
FQW=/workspace/FQW
LISA="$FQW/code/LISA"
# Не ставить base/bin впереди уже активированного env: иначе python в MFAligner окажется базовым.
if ! command -v conda >/dev/null 2>&1; then
  for d in /opt/miniforge3 /opt/miniforge; do [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break; done
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$LISA"
set -a; [ -f "$FQW/.env" ] && source "$FQW/.env"; set +a   # OSF_TOKEN, если нужен

stamp() { echo "=== $(date '+%F %T') $*"; }
NEMAR_OK=$(python3 -c "import json,sys; d=json.load(open('data/MASC-MEG/nemar_download.json')); print(int(not d['failed']))" 2>/dev/null || echo 0)
if [ "$MODE" = "clean" ] && [ "$NEMAR_OK" = "1" ]; then
  stamp "download ICA only (raw data from NEMAR)"
  conda activate LISA
  lisa-download-osf --project-ids 3wrft --output-dir "$LISA/data" --no-project-subdir --remote-subdir clean/
  conda deactivate
elif [ "$NEMAR_OK" != "1" ]; then
  stamp "download (OSF)"; bash scripts/download_dataset.sh
fi
stamp "mfa";     [ -d data/MASC-MEG/stimuli/combined_mfa ] || bash scripts/run_mfa.sh
stamp "prepare $MODE"; bash scripts/prepare_dataset.sh "$MODE" "$N_JOBS"
stamp "check"
conda activate LISA
cd "$FQW" && python code/scripts/exp08_09_full_lisa/full_lisa.py check
stamp "done"
