# Reference fits (regional model)

Meridian 2.1.0, JAX CPU, `media_prior_type="roi"`, `roi_m ~ LogNormal(0.2, 0.9)`. Three fits: a CPU pilot, and two full fits (40 geos x 156 weeks, AKS on, 66 knots, 4 chains, 1000 adapt, 500 burn-in, 1000 kept = 4000 draws).

| | pilot (3 geos x 80 weeks; 2 chains, 300 kept) | strong A (seed 123, target_accept 0.85) | strong B (seed 124, target_accept 0.95) |
|---|---|---|---|
| Fit time (incl. JIT) | 37 s, 8 CPU | 577 s, 16 CPU | 876 s, 16 CPU (15 min total job) |
| `roi_m` R-hat max | 1.019 | 1.006 | 1.004 |
| `roi_m` bulk / tail ESS min | 198 / 302 | 526 / 773 | 705 / 1041 |
| `roi_m` MCSE of mean, max | 0.111 | 0.038 | 0.034 |
| `reff` for PSIS | 0.54 | 0.22 | 0.38 |
| Worst variable (R-hat / bulk ESS) | `knot_values`, `mu_t`: 1.030 / 88 | `mu_t`: 1.048 / 86 | `mu_t`: 1.030 / 92 |
| Divergences | 8 / 600 | 120 / 4000 (per chain 54, 9, 22, 35) | 4 / 4000 (0, 3, 0, 1) |
| Step size / mean `n_steps` | 0.040 / - | 0.029 / 124 | 0.016 / 255 |

The pilot uses different data, so its posteriors are not comparable.

`target_accept_prob` goes through `dual_averaging_kwargs`, a static jit argument in Meridian 2.1.0 (JAX): a plain dict raises "Non-hashable static arguments are not supported", so `fit_config.HashableDict` is used.

## Observations
- In both strong fits `n_steps` is 127 in about 97-98% of iterations. A probe on the pilot data showed this is NUTS stopping on its U-turn criterion at tree depth 7, not Meridian's depth cap of 10; iterations with fewer steps are essentially the divergent ones.
- Divergent draws have higher `roi_m` than the rest: A, Channel0 2.06 vs 1.60 and Channel2 1.56 vs 1.22; B, Channel0 3.55 vs 1.66 (4 divergent draws). An association in these fits, not shown to be a bias.
- A minus B on `roi_m`: all 20 differences (5 channels x mean, q05, q50, q95) are within 2.4 combined MCSEs (largest: Channel4 median, -2.3). Mean differences reach 0.10 (about 6% of the posterior SD) between two valid fits, a first indication of the fit-to-fit variability that PSIS-vs-refit errors must be judged against (measured properly in `RESULTS_noise.md`).

## Assessment
- B is the reference used for the `roi_m` benchmark: R-hat 1.004, ESS ≥ 705, 0.1% divergences. It is not adequate for claims on time or geo effects (`mu_t`, `knot_values`, `tau_g`: R-hat up to 1.03, ESS about 92-160).
- A has too many divergences for tail-sensitive prior changes.
- Refits must use B's MCMC settings. One fit is about 15 minutes on 16 CPUs.
