# 15-scenario `roi_m` benchmark: PSIS and raw IS against a full refit

Reference: `ref_strong_ta0.95_seed124` (S = 4000, `reff` 0.378, `roi_m` LogNormal(0.2, 0.9)). Refits: Slurm array, 15 tasks, 16 CPUs each, seeds 1001-1015. Same data (40 geos x 156 weeks), model (AKS on, ROI parameterization) and MCMC (4 chains, 1000 adapt, 500 burn-in, 1000 kept) as the reference; only the `roi_m` prior differs. Code: `run_refit.py`, `analyze.py`, `noise_check.py`, `scenarios.py`. Tables: `results/scenario_level.csv`, `channel_level.csv`, `mean_error_in_mcse_units.csv`.

Error = estimate from the reference draws minus the refit estimate. "Unadjusted" = the reference posterior without reweighting. All summaries use the same midpoint-CDF weighted-quantile estimator.

## Refit quality
`roi_m`: R-hat max 1.001-1.017, bulk ESS min 409-2025, MCSE of mean max 0.009-0.054: adequate as comparison targets for the mean. Elsewhere: S11 has 124 divergences and an all-variable R-hat max of 1.298; S15 has 1.172; S07 1.077. Divergences per refit: 2-124 of 4000 (reference: 4). They have been associated with the high-ROI region, so tail results (q95) of divergent refits deserve caution.

## Results (mean absolute error over the 5 channels)

| Scenario | k̂ | status | PSIS ESS fraction | Mean: PSIS (unadjusted) | q95: PSIS (unadjusted) | Max abs mean error / MCSE: PSIS (unadjusted) |
|---|---|---|---|---|---|---|
| S01 sigma 0.40 | -0.15 | good | 0.21 | 0.014 (0.190) | 0.029 (0.503) | 0.9 (7.2) |
| S02 sigma 0.55 | -0.34 | good | 0.52 | 0.008 (0.120) | 0.034 (0.270) | 0.7 (6.7) |
| S03 sigma 0.70 | -0.42 | good | 0.83 | 0.011 (0.048) | 0.073 (0.113) | 1.1 (4.1) |
| S04 sigma 1.10 | 0.28 | good | 0.84 | 0.012 (0.037) | 0.033 (0.105) | 1.2 (3.3) |
| S05 sigma 1.30 | 0.42 | good | 0.52 | 0.018 (0.068) | 0.085 (0.198) | 2.6 (6.3) |
| S06 mu -0.20 | 0.21 | good | 0.70 | 0.025 (0.136) | 0.056 (0.179) | 1.4 (6.9) |
| S07 mu 0.00 | 0.07 | good | 0.92 | 0.034 (0.086) | 0.051 (0.113) | 1.8 (4.1) |
| S08 mu 0.40 | -0.07 | good | 0.93 | 0.018 (0.048) | 0.048 (0.080) | 1.1 (3.1) |
| S09 mu 0.60 | 0.02 | good | 0.77 | 0.029 (0.114) | 0.064 (0.183) | 1.9 (7.3) |
| S10 mu 0.80 | 0.12 | good | 0.58 | 0.019 (0.166) | 0.070 (0.239) | 1.2 (9.1) |
| S11 joint 0.0/0.6 | -0.34 | good | 0.64 | 0.053 (0.109) | 0.103 (0.331) | 2.2 (4.4) |
| S12 joint 0.4/0.6 | -0.25 | good | 0.47 | 0.023 (0.098) | 0.037 (0.197) | 1.8 (8.8) |
| S13 joint 0.6/0.6 | 0.05 | good | 0.28 | 0.012 (0.216) | 0.062 (0.144) | 0.7 (12.6) |
| **S14 stress high 0.8/0.35** | **1.31** | **fail** | 0.006 | 0.100 (0.530) | **0.530 (0.426)** | 0.8 (4.8) |
| **S15 stress low -0.4/0.35** | **0.88** | **warning** | 0.013 | 0.019 (0.466) | 0.050 (1.205) | 0.5 (3.8) |

For S14 and S15 the noise scale (`noise_check.py`) is inflated by the tiny ESS fraction, so a small ratio there means the comparison is uninformative, not that PSIS is accurate.

## Observations
1. In the 13 scenarios with k̂ ≤ 0.42 the PSIS mean error is 0.008-0.053 against 0.037-0.216 unadjusted. The largest single-channel mean error is at most 2.6 combined MCSE (11 of 13 at or below 2.0); the unadjusted reference is 3.1-12.6 MCSE off. The remaining discrepancy is at the scale of the Monte Carlo noise of the two fits (two valid fits already differed by up to 0.10).
2. Raw IS and PSIS are indistinguishable in these 13 cases (MAE of means within 0.001).
3. S14 (k̂ 1.31, PSIS ESS 23 of 4000): PSIS is worse than raw IS (mean 0.100 vs 0.070; q95 0.530 vs 0.356) and its q95 is worse than not reweighting (0.426). The diagnostic flagged it.
4. S15 (k̂ 0.88, ESS 53 of 4000): the mean error is small in absolute terms (0.019), but two independent refits agree to 0.002 on the mean (`RESULTS_noise.md`), so it is about 11x the refit-to-refit difference. Median and q95 are no better than raw IS.
5. k̂ ordered the errors at the extremes: the two scenarios with k̂ > 0.7 are the two where PSIS is least accurate or the check is uninformative; no "good" scenario had a large error.
6. Timing (16-CPU node for fits, 1 process for reweighting): reference fit 876 s, refits 528-1121 s; reweighting 0.010-0.019 s on repeat calls (first call 1.07 s, JAX compilation); reweighting plus summaries 0.014-0.018 s. The one-time reference fit (876 s) is excluded.

## Not established
- Whether the 0.01-0.05 residual errors are exactly noise or contain small systematic error (single seed; see `RESULTS_noise.md`).
- S11 and S15 have worse non-`roi_m` diagnostics, so a "good" result there leans on a less trustworthy refit.
- Prior-support-nested LogNormals on `roi_m` only, one dataset, one reference fit. Two failures do not establish a threshold's reliability.
