#!/bin/bash
# Phase 3B single-field refits on upstream main (fork env). Usage:
#   sbatch --array=1,5,6 evidence/parity_main/run_refit_main.sbatch --seed_base 3000 --out_root runs/phase8_main/refits_A
#SBATCH --job-name=p8_refit
#SBATCH --partition=batch
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs/phase8_main
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
${PYTHON:-python} -I evidence/regional_other_params/run_refit_prior.py --index "$SLURM_ARRAY_TASK_ID" "$@"
