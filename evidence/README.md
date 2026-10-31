# Evidence

Scripts, result tables and figures behind the numbers in the top-level README. Not needed to use the method (see `notebooks/`). Every experiment compares the reweighted posterior with independent full refits under the new prior, on Meridian's simulated demo data (no client data).

## Layout

| Folder | Content |
|---|---|
| `common/` | Reweighting core (`prior_reweight.py`), data builders, fit settings (`fit_config.py`). |
| `national_300/` | National model, 300 priors (`roi_m`, `alpha_m`, `ec_m`), each refit once. Figures in `figures/blog/`. |
| `regional_60/` | Regional model (40 geos x 156 weeks), 60 priors (`roi_m`, `eta_m`), each refit once. Plan: `REGIONAL_PLAN.md`. |
| `regional_roi/` | Regional reference fit and 15 `roi_m` scenarios, two refits each (refit-to-refit noise). |
| `regional_other_params/` | Same reference, other parameters: `alpha_m`, `ec_m`, `eta_m`, `sigma`, joint and per-channel changes. |
| `heatmaps/` | k̂ / ESS / error heatmaps over the regional scenarios (reads saved CSVs). |
| `parity_main/` | Unit tests of the core; checks against Meridian `main`. |
| `docs/` | `PRIOR_INVENTORY.md` (which priors can be reweighted, and why); pinned package list of the original runs. |

Result tables are in each folder's `results/`, write-ups in `RESULTS*.md`. Those files and some docstrings use the project's working names:

| Name in the text | Folder |
|---|---|
| Phase 1 (core, tests) | `common/`, `parity_main/test_*.py` |
| Phase 2 (reference fit), Phase 3 (15 `roi_m` scenarios), Phase 4 (refit noise) | `regional_roi/` |
| Phase 3B (other parameters) | `regional_other_params/` |
| Phase 5 (heatmaps) | `heatmaps/` |
| Phase 8 (checks on `main`) | `parity_main/` |
| "Stage 1-4" | `national_300/`, `regional_60/` |

## Setup

```
pip install -r requirements.txt
```

Run everything from the repository root: `python -I evidence/<folder>/<script>.py`. Fits are written to `runs/` (untracked, about 2 GB of NetCDF). Run directories keep their original names (`runs/phase3_refits`, `runs/phase3b_refits_A`, `runs/national_targets_A`, ...); the analysis scripts take them as arguments. See each script's docstring for its arguments. Original package versions: `docs/environment_meridian_psis_freeze.txt`.

The `*.sbatch` files are for Slurm: adapt partition, time and memory, and submit from the repository root. They use `python` from your environment, or `PYTHON=/path/to/python sbatch ...`.

## Reproducing the experiments

**National, 300 priors.** Refit: about 1 minute on 4 cores.
```
python -I evidence/national_300/run_national_fit.py --seed 9001 --tag longref --n_keep 5000   # reference, 4 x 5000 draws
sbatch --array=1-300%100 evidence/national_300/run_target_refit.sbatch --seed_base 10000 --out_root runs/national_targets_A
python -I evidence/national_300/analyze_targets.py --ref runs/national/<longref run> \
    --refits_a runs/national_targets_A --refits_b runs/national_targets_B --out evidence/national_300/results
python -I evidence/national_300/make_blog_figures.py
```
The 300 priors are in `calibration/` (made by `calibrate_perturbations.py`).

**Regional, 60 priors.** Reference: about 40 minutes. Refits: median 23 minutes (8 to 95, depending on cluster load).
```
sbatch evidence/regional_60/run_regional_fit.sbatch --seed 9101 --n_keep 5000
sbatch --array=1-60 evidence/regional_60/run_regional_refit.sbatch --which A --out_root runs/regional_targets_A
sbatch --array=1-60 evidence/regional_60/run_regional_refit.sbatch --which B --out_root runs/regional_targets_B   # 6 targets have a second refit
python -I evidence/regional_60/analyze_regional.py --ref runs/regional/<longref run> \
    --refits_a runs/regional_targets_A --refits_b runs/regional_targets_B --out evidence/regional_60/results
python -I evidence/regional_60/make_regional_figures.py
```

**Regional, 15 `roi_m` scenarios and other parameters.** Reference: about 15 minutes on 16 cores. Refits: 9 to 19 minutes.
```
sbatch evidence/regional_roi/run_reference_fit.sbatch --mode strong --seed 124 --target_accept 0.95
sbatch --array=1-15 evidence/regional_roi/run_refit.sbatch                                                      # refits A, seeds 1001-1015
sbatch --array=1-15 evidence/regional_roi/run_refit.sbatch --seed_base 2000 --out_root runs/phase3_refits_B    # refits B
python -I evidence/regional_roi/analyze.py --ref runs/<reference run> --refits runs/phase3_refits --out evidence/regional_roi/results
python -I evidence/regional_roi/analyze_noise.py --ref runs/<reference run> --refits_a runs/phase3_refits --refits_b runs/phase3_refits_B --out evidence/regional_roi/results
sbatch --array=1-21 evidence/regional_other_params/run_refit_prior.sbatch --seed_base 3000 --out_root runs/phase3b_refits_A
sbatch --array=1-21 evidence/regional_other_params/run_refit_prior.sbatch --seed_base 4000 --out_root runs/phase3b_refits_B
python -I evidence/regional_other_params/analyze_prior.py --ref runs/<reference run> --refits_a runs/phase3b_refits_A --refits_b runs/phase3b_refits_B --out evidence/regional_other_params/results
```

**Unit tests of the core** (no MCMC, about 45 seconds).
```
python -I evidence/parity_main/test_prior_reweight.py
python -I evidence/parity_main/test_logprob_parity.py
```

**Checks against Meridian `main`.** These need a clone of google/meridian installed in editable mode (`pip install -e <clone>[dev]`), not the PyPI package.
- `parity_joint_fields.py`, `parity_joint_variants.py`: the joint log-density difference between a reference and a target model equals the sum of the per-field prior ratios (33 cases). Last output: `parity_main_jax.txt`.
- `run_ref_main.sbatch`, `run_refit_main.sbatch`, `compare_main.py`: rerun the reference fit and four scenarios on `main`, compare with the 2.1.0 runs (`compare_main.csv`). Set `MERIDIAN_CLONE` for the sbatch files.
- `run_baseline_tests.sbatch`: Meridian's own test suite on `main` (`baseline_jax/`, `baseline_tensorflow/`). The failures and collection errors come from optional extras that were not installed.

## Reproducibility check

Rerun from the new locations on the stored fits: `analyze_targets.py` (national) and `analyze_regional.py` reproduce `metrics_target_level.csv` exactly, and the 15 unit tests and 33 parity checks pass.

## Notes

- The 300-prior and 60-prior tests use a reference of 4 chains x 5000 draws. The notebooks use 4 x 1000 or fewer.
- `slope_m` cannot be reweighted: its default prior is `Deterministic(1.0)` (see `docs/PRIOR_INVENTORY.md`).
