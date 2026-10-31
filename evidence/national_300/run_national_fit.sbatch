#!/bin/bash
# Usage: sbatch --cpus-per-task=4 --job-name=nat4 evidence/national_300/run_national_fit.sbatch --tag cpus4 --seed 123
#SBATCH --partition=batch
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
${PYTHON:-python} -I evidence/national_300/run_national_fit.py "$@"
