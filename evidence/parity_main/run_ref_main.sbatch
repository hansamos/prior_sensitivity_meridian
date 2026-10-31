#!/bin/bash
# Reference fit on upstream main (fork env). Same config as the Oct-27 reference: strong, seed 124, target_accept 0.95.
#SBATCH --job-name=p8_ref
#SBATCH --partition=batch
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=24G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

cd "${SLURM_SUBMIT_DIR:?submit from the repository root}" || exit 1
mkdir -p logs runs/phase8_main
unset PYTHONPATH
export JAX_PLATFORMS=cpu MERIDIAN_BACKEND=jax
git -C "${MERIDIAN_CLONE:?path of the Meridian clone}" rev-parse HEAD > runs/phase8_main/meridian_commit.txt
${PYTHON:-python} -I evidence/regional_roi/run_reference_fit.py --mode strong --seed 124 --target_accept 0.95 --out_root runs/phase8_main
