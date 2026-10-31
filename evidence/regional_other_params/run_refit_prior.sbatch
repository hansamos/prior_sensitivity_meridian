#!/bin/bash
# Phase 3B refit array. Usage: sbatch --array=1-4 evidence/regional_other_params/run_refit_prior.sbatch --seed_base 3000 --out_root runs/phase3b_refits_A
#SBATCH --job-name=psis_3b
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
${PYTHON:-python} -I evidence/regional_other_params/run_refit_prior.py --index "$SLURM_ARRAY_TASK_ID" "$@"
