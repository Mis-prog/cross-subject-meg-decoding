#!/bin/bash
#SBATCH --job-name=exp13
#SBATCH --account=proj_1910
#SBATCH --partition=rocky
#SBATCH --constraint=type_a|type_b|type_c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --array=0-5%2
#SBATCH --time=06:00:00
#SBATCH --output=exp13-%A_%a.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root with sbatch}"

canon=${1:-/home/isemenkov/shared-datasets/LISA/preprocessed}
out=${2:-outputs/exp13_shared_only}
fold=${SLURM_ARRAY_TASK_ID:?Submit this script as a job array}
threads=${SLURM_CPUS_PER_TASK:?Submit with --cpus-per-task}
if [[ ! "$fold" =~ ^[0-5]$ ]]; then
    printf 'Array index outside 0-5: %s\n' "$fold" >&2
    exit 2
fi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"

runner=code/scripts/exp10_shared_matrix/run_experiment.py
task="shared_only_f${fold}_s42"
args=(
    --protocol experiments/exp13_shared_only/protocol.json
    --task "$task"
    --canon "$canon"
    --out "$out"
    --threads "$threads"
)
printf 'task=%s canon=%s out=%s\n' "$task" "$canon" "$out"
time .venv/bin/python -u "$runner" train "${args[@]}"
time .venv/bin/python -u "$runner" eval "${args[@]}"
