# Other parameters: `alpha_m`, `ec_m`, joint and per-channel changes, `sigma`, `slope_m`, `eta_m`

Reference: `ref_strong_ta0.95_seed124` (`roi_m` LogNormal(0.2, 0.9), all other priors Meridian defaults). Each target changes one field (one common prior for all 5 channels, unless stated); `roi_m` prior, `ModelSpec`, data and MCMC (4 x 1000 kept, 1000 adapt, 500 burn-in, `target_accept_prob` 0.95, AKS on) are unchanged. Two independent refits per target (A, B). Targets were fixed before any k̂ or refit result, except AM5 and EA6, added afterwards from reference-draw k̂ alone (to cover k̂ 0.5-0.9). Code: `scenarios_other.py`, `run_refit_prior.py`, `analyze_prior.py`, `analyze_multi.py`. Tables: `results/`, `results_multi/`.

| Block | Refit A seeds | Refit B seeds |
|---|---|---|
| `alpha_m` | 3001-3005 | 4001-4005 |
| `ec_m` | 3006-3010 | 4006-4010 |
| `sigma` | 3011-3015 | 4011-4015 |
| `eta_m` | 3016-3021 | 4016-4021 |
| joint / per-channel | 6001-6005 | 7001-7005 |

Tables report the MAE over the 5 channels of the posterior mean in parameter units: PSIS vs A, PSIS vs B, A vs B (refit-to-refit noise), and no reweighting vs A. "Max z" is the largest channel error in combined bootstrap SEs. Quantile results and full tables are in the CSVs.

## `alpha_m`: Uniform(0,1) → Beta

Reference `alpha_m` posterior: channel means 0.41-0.84, sd 0.14-0.25. Refits: R-hat ≤ 1.008, bulk ESS ≥ 799; elsewhere `mu_t` mixes worst (all-variable R-hat ≤ 1.065, AM3 refit A 1.183), 7-123 divergences of 4000 (reference: 4).

| Target | k̂ | status | ESS frac | PSIS vs A | PSIS vs B | A vs B | no reweighting | max z |
|---|---|---|---|---|---|---|---|---|
| Beta(2,2) | -0.04 | good | 0.36 | 0.006 | 0.006 | 0.005 | 0.056 | 1.7 |
| Beta(5,2) | -0.10 | good | 0.15 | 0.004 | 0.003 | 0.001 | 0.097 | 1.2 |
| Beta(0.5,0.5) | 0.50 | good | 0.16 | 0.013 | 0.009 | 0.004 | 0.050 | 1.7 |
| Beta(1,3) | 0.78 | warning | 0.014 | 0.020 | 0.019 | 0.005 | 0.212 | 1.6 |
| Beta(2,8) | 2.78 | fail | 0.001 | 0.104 | 0.107 | 0.003 | 0.363 | 3.7 |

- The prior change is real: unadjusted error 0.05-0.36 against refit-to-refit 0.001-0.005.
- For k̂ ≤ 0.50, PSIS leaves 4-26% of the effect (0.003-0.013 against 0.050-0.097); in `alpha_m` units 0.013 is about 0.07 posterior sd. Beta(0.5,0.5) (k̂ 0.495, density ratio unbounded at 0 and 1) is 2-3x the refit level on the mean, median and q95 (max z 2.3).
- Beta(1,3) (k̂ 0.78, ESS about 55 draws): removes about 90% of the effect but leaves 3-7x the refit level (q05: 0.029 vs 0.004, z 3.0).
- Beta(2,8) (k̂ 2.78, ESS about 3 draws): removes about 70% of the effect yet is 35x the refit level (z 3.7-4.0); raw IS is no better (0.098).
- The k̂ order matches the order of excess error.

## `ec_m`: TruncatedNormal(0.8, 0.8; 0.1, 10) → TruncatedNormal, same bounds

Reference `ec_m` posterior: channel means 1.03-1.16, sd 0.57-0.60, largest draw 3.80, `reff` 0.99. Refits: R-hat 1.001-1.005, bulk ESS ≥ 1602 in 9 of 10. **EC4 refit A did not converge** (seed 3009: R-hat 1.130, bulk ESS 21, 315 divergences, 307 in chain 3). It is excluded and replaced by one new refit (seed 5009: R-hat 1.002, ESS 2626, 8 divergences), chosen because of the failure; refit B of EC4 was clean and agrees with it. Elsewhere all-variable R-hat ≤ 1.053, divergences 3-94.

| Target | k̂ | status | ESS frac | PSIS vs A | PSIS vs B | A vs B | no reweighting | max z |
|---|---|---|---|---|---|---|---|---|
| TN(1.0, 0.8) | 0.09 | good | 0.83 | 0.007 | 0.004 | 0.009 | 0.114 | 1.4 |
| TN(1.5, 0.5) | 0.28 | good | 0.11 | 0.023 | 0.030 | 0.012 | 0.394 | 2.5 |
| TN(0.4, 0.4) | 0.44 | good | 0.04 | 0.020 | 0.016 | 0.007 | 0.504 | 3.5 |
| TN(0.8, 2.0) | 0.80 | warning | 0.017 | 0.335 | 0.344 | 0.020 | 0.833 | 4.9 |
| TN(3.0, 0.5) | 4.14 | fail | ~0 | 0.759 | 0.752 | 0.011 | 1.881 | 6.4 |

- For k̂ ≤ 0.44, PSIS removes about 94-96% of the mean effect. TN(1.0, 0.8) (ESS 0.83) is at the refit level for the mean and all quantiles.
- **The other two "good" targets are 2-3x the refit level** (mean 0.020 vs 0.007 and 0.023 vs 0.012; ESS fractions 0.04 and 0.11, about 150 and 430 effective draws). The excess sits in a few channels and agrees with both refits (TN(0.4, 0.4) channel 2: +0.055 vs A, +0.043 vs B, about 0.07-0.09 posterior sd), so the PSIS estimate is the one that is off. k̂ did not separate these from the clean target.
- TN(0.8, 2.0) (k̂ 0.80): removes about 60% of the mean effect, 17x the refit level; q95 error 1.29. The refit posterior puts 9% of draws above 3.8 (the reference's largest) and 29% above 2.5 (reference: 2%): the reference holds no draws for much of the target.
- TN(3.0, 0.5) (k̂ 4.1): about 1 effective draw, mean error 70x the refit level; raw IS is no better (0.748).
- Hypothesis, untested: the errors at k̂ 0.28-0.44 may be sampling error of a PSIS estimator built from 150-430 effective draws of one reference fit, which the within-fit bootstrap understates.
- Over `alpha_m` and `ec_m` together (10 targets): no k̂ ≤ 0.5 case was as far off as the unadjusted reference, every k̂ ≥ 0.78 case was clearly above refit noise, and k̂ 0.28-0.44 with ESS fraction below about 0.1 can still leave errors a few times the refit noise.

## Joint and per-channel changes

All changed fields enter one PSIS call. `reff` is the minimum over the changed fields (a conservative choice, not derived from the weights; 0.64 for the joint cases, effect not tested). In every joint scenario the joint log ratio equals the sum of the single-factor ratios exactly (max difference 0.0), and in C2 it is exactly 0 on the four unchanged channels.

| ID | Change | k̂ | status | ESS frac |
|---|---|---|---|---|
| J1 | `alpha_m` Beta(2,2) and `ec_m` TN(1.0, 0.8) | 0.01 | good | 0.32 |
| J2 | `alpha_m` Beta(5,2) and `ec_m` TN(1.5, 0.5) | 0.88 | warning | 0.016 |
| J3 | `alpha_m` Beta(0.5,0.5) and `ec_m` TN(0.4, 0.4) | 0.75 | warning | 0.014 |
| C1 | `alpha_m` Beta per channel: (2,2), (5,2), (2,2), (1.5,1.5), (5,2) | -0.25 | good | 0.40 |
| C2 | `alpha_m` Beta(3,3) on channel 2 only, Uniform elsewhere | -1.59 | good | 0.80 |

Refits: changed fields R-hat ≤ 1.010, bulk ESS ≥ 344; all-variable R-hat 1.027-1.095 except J2 refit B (1.277, `mu_t`); J2's two refits agree on the changed fields (A vs B 0.001 and 0.009). Divergences 4-97.

| Scenario | field | k̂ | PSIS vs A | PSIS vs B | A vs B | no reweighting | max z |
|---|---|---|---|---|---|---|---|
| J1 | `alpha_m` | 0.01 | 0.005 | 0.006 | 0.007 | 0.052 | 1.3 |
| J1 | `ec_m` | 0.01 | 0.015 | 0.031 | 0.018 | 0.080 | 1.2 |
| C1 | `alpha_m` | -0.25 | 0.002 | 0.003 | 0.002 | 0.050 | 1.0 |
| C2 | `alpha_m` | -1.59 | 0.005 | 0.005 | 0.004 | 0.016 | 1.4 |
| J2 | `alpha_m` | 0.88 | 0.013 | 0.013 | 0.001 | 0.094 | 1.5 |
| J2 | `ec_m` | 0.88 | 0.064 | 0.062 | 0.009 | 0.361 | 2.1 |
| J3 | `alpha_m` | 0.75 | 0.031 | 0.021 | 0.009 | 0.032 | 1.7 |
| J3 | `ec_m` | 0.75 | 0.028 | 0.026 | 0.005 | 0.490 | 1.9 |

- J1, C1, C2 (k̂ ≤ 0.01) are at the refit level for the mean in every field; one exception among the quantiles, J1 `ec_m` q05 (0.034 vs 0.014, no better than not reweighting, z 2.8). A single-channel change (C2) and a per-channel specification (C1) are handled correctly.
- J2 (k̂ 0.88): means improve a lot but stay 7-13x above the refit level (`ec_m` 0.064 vs 0.009, an 82% reduction of 0.361). The `ec_m` q95 is worse than not reweighting (0.122 vs 0.112) and than raw IS (0.088).
- J3 (k̂ 0.75): `ec_m` mean improves by 94% (0.490 → 0.028), but the `alpha_m` mean (0.031) is no better than not reweighting (0.032) and its q05 is worse (0.101 vs 0.053; raw IS 0.100, so not a smoothing effect).
- The warning-level k̂ of J2 and J3 comes from combining changes that were each good alone (single-factor k̂ -0.10 and 0.28 for J2; 0.49 and 0.44 for J3).

## `sigma`: HalfNormal(5) → HalfNormal / LogNormal

Scalar `sigma` (`unique_sigma_for_each_geo=False`). The reference posterior is tight (mean 0.784, sd 0.0071, `reff` 1.0), so HalfNormal(5) is essentially flat across it: only priors with a width of about 0.01-0.04 move it, hence LogNormal targets of that width plus one weak control. Refits: R-hat ≤ 1.004, bulk ESS ≥ 4714; all-variable R-hat ≤ 1.115, divergences 1-61.

| Target | k̂ | status | ESS frac | PSIS vs A | PSIS vs B | A vs B | no reweighting | max z |
|---|---|---|---|---|---|---|---|---|
| HalfNormal(1.0), control | -0.01 | good | 1.00 | 0.00010 | 0.00017 | 0.00008 | 0.00014 | 0.8 |
| LogNormal(log 0.8, 0.05) | -1.70 | good | 1.00 | 0.00003 | 0.00018 | 0.00015 | 0.00042 | 0.3 |
| LogNormal(log 0.8, 0.02) | -1.56 | good | 0.87 | 0.00016 | 0.00013 | 0.00003 | 0.00253 | 1.3 |
| LogNormal(log 0.85, 0.02) | 0.42 | good | 0.13 | 0.00007 | 0.00014 | 0.00007 | 0.01100 | 0.1 |
| LogNormal(log 0.95, 0.02) | 1.33 | fail | 0.001 | 0.00287 | 0.00279 | 0.00007 | 0.02734 | 1.4 |

- The control behaves as a null: reweighting changes nothing beyond Monte Carlo noise.
- For k̂ ≤ 0.42, PSIS recovers the refit mean to within 0.0002; for the largest shift (1.5 posterior sd, 0.0110 unadjusted) the error is under 1% of it. At k̂ 0.42 (ESS 0.13) the mean is still at the refit level while the q95 error is 0.0034 (z 1.4, 26x the A-vs-B level).
- The failing target (k̂ 1.33, about 4 effective draws): removes about 90% of the mean shift yet stays 41x the refit level; q95 z 10.4; the PSIS bootstrap SE of the mean (0.0021) is about 20x the refit SEs. The diagnostic flagged it.
- `sigma` is far better determined by the likelihood than the treatment-effect priors, so these small absolute errors do not transfer to weakly identified parameters.

## `slope_m`: not reweightable under the default

In the reference fit `slope_m` is `Deterministic(1.0)`: all 4000 x 5 draws equal exactly 1.0. A point mass has no density, so the factor is rejected (`ValueError: ... reference prior is Deterministic (a point mass) ...`), for a deterministic and for a stochastic target alike; a stochastic target would also violate the support condition. No reach-and-frequency channels exist in this dataset, so `slope_rf` was not testable. A stochastic slope prior needs a new reference model.

## `eta_m`: HalfNormal(1) → HalfNormal / LogNormal

Reference `eta_m` posterior: channel means 0.53-0.93, sd 0.43-0.55, largest draw 3.39, bulk ESS ≥ 833, `reff` 0.38. Refits: R-hat 1.001-1.012, bulk ESS 502-3017 (lowest for EA2, MCSE of the mean 0.036-0.038); all-variable R-hat 1.022-1.282 (EA2 refit B), divergences 0-35.

| Target | k̂ | status | ESS frac | PSIS vs A | PSIS vs B | A vs B | no reweighting | max z |
|---|---|---|---|---|---|---|---|---|
| HalfNormal(0.5) | -0.02 | good | 0.18 | 0.011 | 0.009 | 0.009 | 0.285 | 1.0 |
| LogNormal(0, 0.5) | -0.44 | good | 0.13 | 0.017 | 0.017 | 0.007 | 0.276 | 1.8 |
| HalfNormal(2.0) | 0.44 | good | 0.23 | **0.169** | **0.139** | 0.038 | 0.397 | **6.6** |
| HalfNormal(0.35) | 0.58 | good | 0.03 | 0.018 | 0.015 | 0.006 | 0.407 | 2.0 |
| HalfNormal(0.2) | 3.54 | fail | 0.003 | 0.031 | 0.032 | 0.002 | 0.540 | 1.5 |
| LogNormal(log 2, 0.3) | 5.60 | fail | 0.002 | 0.519 | 0.509 | 0.012 | 1.130 | 11.2 |

- HalfNormal(0.5) and LogNormal(0, 0.5) remove 94-96% of the shift and are close to the refit level.
- **HalfNormal(2.0) (k̂ 0.44, "good") is wrong by a large, one-directional amount**: mean error 0.169 vs A and 0.139 vs B (4x refit level), only about 57% of the shift recovered, all five channels under-estimated (z -2.6 to -6.6), q95 error 0.467 vs 0.145 refit-to-refit (z 8.9). The wider prior pulls the posterior into a region the reference barely visited: in refit A, 18-29% of draws per channel exceed the reference's 95th percentile and 2-6% exceed its maximum (reference max 2.83-3.39, refit max 5.9-7.2). The ESS fraction (0.23) did not flag it.
- HalfNormal(0.35) (k̂ 0.58, about 110 effective draws): removes about 96% of the shift, 3x the refit level; channel 2 is off by -0.040 against both refits (z -2.0).
- HalfNormal(0.2) (k̂ 3.54, 12 effective draws): flagged as a failure, yet the mean is within 0.031 of the refit (large own bootstrap SE); quantile errors are 0.038-0.056 (q95 z 4.2). Plausibly because the target concentrates mass near zero, which the reference covers (untested).
- LogNormal(log 2, 0.3) (k̂ 5.6): clear failure (mean error 43x refit level, q95 z 37).
- Across the 80 good-target cells, 16 have |z| > 2 against refit A, versus 5 for A vs B; most come from HalfNormal(2.0).
- Untested ideas: a guard on how well the reference covers the region the target upweights (e.g. the share of weight carried by the top 5% of draws by parameter value); reference-fit sampling error in the `eta_m` upper tail (`reff` 0.38) probably contributes.

## Caveats
Few targets per field, one dataset, one reference fit, one refit pair each (EC4: the second attempt of refit A), one common prior across channels except C1 and C2. Different prior families per channel for one field, and channel-specific `ec_m` changes, were not run.
