#!/bin/bash
# Phase 2 reference fit (CPU). Usage: sbatch evidence/regional_roi/run_reference_fit.sbatch [--mode cpu_pilot] [--seed 123]
#SBATCH --job-name=psis_ref
#SBATCH --partition=batch
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
${PYTHON:-python} -I evidence/regional_roi/run_reference_fit.py "$@"
