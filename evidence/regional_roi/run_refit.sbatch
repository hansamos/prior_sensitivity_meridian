#!/bin/bash
# Phase 3 refit array. Usage: sbatch --array=1-15 evidence/regional_roi/run_refit.sbatch   (one scenario per task)
#SBATCH --job-name=psis_refit
#SBATCH --partition=batch
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
${PYTHON:-python} -I evidence/regional_roi/run_refit.py --index "$SLURM_ARRAY_TASK_ID" "$@"
