# PSIS-vs-refit error against refit-to-refit variability

Reference: `ref_strong_ta0.95_seed124`. For each of the 15 `roi_m` scenarios, two independent full refits with identical settings (4 chains x 1000 kept, 1000 adapt, 500 burn-in, `target_accept_prob` 0.95, AKS on): A seeds 1001-1015, B seeds 2001-2015. Code: `analyze_noise.py`. Tables: `results/channel_level_noise.csv`, `scenario_level_noise.csv`, `se_calibration_z_AB.csv`.

Statistics: posterior mean and equal-tailed 5/50/95% quantiles. Errors: |PSIS - A|, |PSIS - B|, |A - B|. Monte Carlo SEs: moving-block bootstrap within chains (block 50, 300 replicates). The PSIS bootstrap resamples (draw, weight) pairs and does not re-estimate the Pareto tail, so it understates the smoothing uncertainty.

## Refit quality
B: `roi_m` R-hat max 1.002-1.011, bulk ESS min 411-1712. A: R-hat max 1.017. Outside `roi_m`, several refits reach all-variable R-hat 1.10-1.30 and up to 124 divergences (A: S11, 124 divergences, R-hat 1.298; B: S01 1.101, S12 1.102, S10 77 divergences, S13 67).

## Calibration of the bootstrap SEs
z_AB = (A - B) / sqrt(se_A² + se_B²) over the 75 channel-by-scenario cells per statistic: sd 1.22 (mean), 1.21 (q05), 1.22 (q50), 1.07 (q95); 5-11% of cells with |z| > 2 (4.6% expected). The SEs are about 10-20% too small. Mean z_AB is within 0.15 of 0: no systematic A-B offset.

## Results (MAE over channels, ROI units)

Good scenarios (S01-S13, k̂ ≤ 0.42; 65 channel cells per statistic):

| Statistic | PSIS vs A | PSIS vs B | A vs B | cells with abs z (PSIS vs A) > 2, of 65 | cells with abs z (A vs B) > 2 |
|---|---|---|---|---|---|
| mean | 0.021 | 0.021 | 0.027 | 4 | 8 |
| q05 | 0.020 | 0.017 | 0.016 | 7 | 5 |
| q50 | 0.026 | 0.026 | 0.026 | 4 | 8 |
| q95 | 0.057 | 0.060 | 0.066 | 3 | 4 |

Borderline and failing scenarios (one each, 5 channels):

| | Statistic | PSIS vs A | PSIS vs B | A vs B |
|---|---|---|---|---|
| S15 (k̂ 0.88) | mean | 0.019 | 0.020 | 0.002 |
| | q50 | 0.031 | 0.035 | 0.006 |
| | q95 | 0.050 | 0.046 | 0.008 |
| S14 (k̂ 1.31) | mean | 0.100 | 0.106 | 0.007 |
| | q50 | 0.098 | 0.098 | 0.014 |
| | q95 | 0.530 | 0.518 | 0.025 |

## Observations
1. In the 13 good scenarios the PSIS-vs-refit error is about the refit-vs-refit difference for the mean and all three quantiles (pooled MAE ratios 0.8-1.25). The share of cells with |z| > 2, PSIS vs A (6%, 11%, 6%, 5% for mean/q05/q50/q95), is similar to A vs B (12%, 8%, 12%, 6%) once the SE underestimate is allowed for.
2. Individual cells vary: scenario-level ratios of PSIS-vs-A to A-vs-B MAE range from 0.3 to 5.4 (excluding S14, S15). Examples: S05 q05 (0.052 vs 0.010, max z 3.0), S11 (z up to 4.1 for q05 against A). Of 260 good-scenario cells, 18 have |z_PSIS,A| > 2.
3. S11 refit A is the odd one out: PSIS vs A is 0.053 on the mean, PSIS vs B 0.013, A vs B 0.041 (refit A had 124 divergences, R-hat 1.298; B had 12). Without S11 the pooled PSIS-vs-A errors are 0.019/0.019/0.025/0.053 (mean/q05/q50/q95) against A-vs-B 0.026/0.016/0.025/0.065.
4. For S15 and S14, PSIS error is far above refit-to-refit variability and PSIS agrees with neither refit: S15 mean 0.019 against 0.002 (about 11x; max z 2.9), S14 mean 0.100 against 0.007, q95 0.53 against 0.025 (21x).

## Interpretation
- For k̂ ≤ 0.42, PSIS is about as far from a fresh refit as two fresh refits are from each other. This supports "no detectable excess error over fit-to-fit noise in these cases", not "PSIS is at the MCMC noise floor" (one pair per scenario, SEs somewhat too small, exceptions above).
- For k̂ > 0.7 (two scenarios) the PSIS error is a multiple of the noise floor: the diagnostic did its job. One scenario per class does not locate the transition between 0.42 and 0.88.
- Not tested: k̂ in 0.5-0.7 (none in this grid), repeated reference fits (part of the PSIS error is that fit's own noise), other priors.
