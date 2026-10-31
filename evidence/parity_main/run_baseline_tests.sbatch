#!/bin/bash
# Baseline pytest of the Meridian fork on upstream main, before any change. Usage: sbatch --export=ALL,BACKEND=jax|tensorflow evidence/parity_main/run_baseline_tests.sbatch
#SBATCH --job-name=mer_base
#SBATCH --partition=batch
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%j_%x.out
#SBATCH --error=logs/%j_%x.err

REPO="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "${MERIDIAN_CLONE:?path of the Meridian clone}" || exit 1
unset PYTHONPATH
export MERIDIAN_BACKEND=${BACKEND:?set BACKEND} JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
OUT="$REPO/evidence/parity_main/baseline_${BACKEND}"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/commit.txt"
${PYTHON:-python} -m pytest -n 16 -q -p no:cacheprovider \
  --junitxml="$OUT/junit.xml" 2>&1 | tail -n 200 > "$OUT/summary.txt"
