#!/bin/bash
# Usage: sbatch evidence/regional_60/run_regional_fit.sbatch --seed 9101 --n_keep 5000
#SBATCH --job-name=reg_longref
#SBATCH --partition=batch
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
${PYTHON:-python} -I evidence/regional_60/run_regional_fit.py "$@"
