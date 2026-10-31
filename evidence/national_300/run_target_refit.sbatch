#!/bin/bash
# Usage: sbatch --array=1-300%100 evidence/national_300/run_target_refit.sbatch --seed_base 10000 --out_root runs/national_targets_A
#SBATCH --job-name=nat_tgt
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%A_%a_%x.out
#SBATCH --error=logs/%A_%a_%x.err
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
${PYTHON:-python} -I evidence/national_300/run_target_refit.py --index "$SLURM_ARRAY_TASK_ID" "$@"
