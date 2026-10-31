#!/usr/bin/env bash
# На сервере: окружения авторов (LISA с CUDA 11.8, MFAligner) + наши зависимости.
# Использование: bash /workspace/FQW/code/scripts/exp08_09_full_lisa/vast/setup.sh
set -euo pipefail
FQW=/workspace/FQW
LISA="$FQW/code/LISA"
LOG="$FQW/logs/full"; mkdir -p "$LOG"

if ! command -v conda >/dev/null 2>&1 && [ ! -x /opt/miniforge3/bin/conda ] && [ ! -x /opt/miniforge/bin/conda ]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p /opt/miniforge
fi
for d in /opt/miniforge3 /opt/miniforge; do [ -x "$d/bin/conda" ] && export PATH="$d/bin:$PATH" && break; done
CONDA_BASE="$(conda info --base)"
grep -q "conda.sh" ~/.bashrc 2>/dev/null || echo "source $CONDA_BASE/etc/profile.d/conda.sh" >> ~/.bashrc
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync tmux)
command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux)

cd "$LISA"
conda env list | awk '{print $1}' | grep -qx LISA || bash scripts/setup_main_env.sh cu118
conda env list | awk '{print $1}' | grep -qx MFAligner || bash scripts/setup_mfa_env.sh

source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate LISA
python -m pip install -q "mlflow==3.16.1" "pytest>=7.0"
python - <<'PY'
import torch, scipy.signal, lisa
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')
PY
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
free -g | head -2; nproc; df -h /workspace | tail -1
echo "setup OK"
