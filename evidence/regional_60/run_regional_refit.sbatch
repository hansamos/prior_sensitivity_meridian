#!/bin/bash
# Usage: sbatch --array=1-60 evidence/regional_60/run_regional_refit.sbatch --which A --out_root runs/regional_targets_A
#SBATCH --job-name=reg_tgt
#SBATCH --partition=batch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
${PYTHON:-python} -I evidence/regional_60/run_regional_refit.py --index "$SLURM_ARRAY_TASK_ID" "$@"
