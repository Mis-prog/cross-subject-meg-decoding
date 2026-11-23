#!/bin/bash
#SBATCH --job-name=exp12
#SBATCH --account=proj_1910
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b|type_c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-11%2
#SBATCH --time=06:00:00
#SBATCH --output=exp12-%A_%a.log

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

canon=${1:?Pass the preprocessed directory}
out=${2:?Pass a separate output directory for this series}
epochs=${3:?Pass the fixed epoch limit}
phase=${4:?Pass pilot or full}
index=${SLURM_ARRAY_TASK_ID:?Submit this script as a job array}

case "$phase" in
    pilot|full) ;;
    *) printf 'Unknown phase: %s\n' "$phase" >&2; exit 2 ;;
esac
if (( index < 0 || index >= 12 )); then
    printf 'Array index outside 0–11: %s\n' "$index" >&2
    exit 2
fi

variants=(individual switch_p0p9)
task="${variants[index % 2]}_f$((index / 2))_s42"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"

runner=experiments/exp12_paired_shared_matrix/scripts/run_experiment.py
args=(--task "$task" --canon "$canon" --out "$out" --threads "$SLURM_CPUS_PER_TASK")
printf 'task=%s phase=%s epochs=%s node=%s\n' "$task" "$phase" "$epochs" "$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
time .venv/bin/python -u "$runner" train "${args[@]}" --nepoch "$epochs"
if [[ "$phase" == full ]]; then
    time .venv/bin/python -u "$runner" eval "${args[@]}"
fi
