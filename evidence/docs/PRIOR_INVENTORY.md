# Which Meridian priors can be reweighted

Established on Meridian 2.1.0, `media_prior_type="roi"`, by inspecting the joint distribution of a model built on the demo data (3 geos x 80 weeks, 5 media channels, 1 organic channel, 2 controls, 1 non-media treatment): [introspect_joint_dist.py](introspect_joint_dist.py), raw output [introspection.json](introspection.json). Meridian `main` differs from 2.1.0 (for example the changepoint priors `zeta_m`, `zeta_rf`); this file covers 2.1.0 only.

## What shapes the weights

1. **Every prior reaches the model wrapped in `BatchBroadcast`** to the channel / geo / knot count. A scalar prior gives a node of batch shape `[J]`, so `log_prob` returns one value per channel: reduce over the last axis, keep the draw axis. Channel-specific priors are possible by passing a batch-shaped distribution.
2. **`slope_m` and `slope_om` are `Deterministic(1.0)`**, not random variables. A point mass has no density, so they cannot enter the ratio.
3. **Deterministic nodes never enter the ratio**: `beta_m`, `beta_gm`, `beta_om`, `beta_gom`, `gamma_n`, `gamma_gn`, `gamma_gc`, `tau_g`, `mu_t`. Under `roi`, the sampled variable is `roi_m` and `beta_m` is derived from it.
4. **Fixed N(0,1) latents** (`beta_gm_dev`, `beta_gom_dev`, `gamma_gc_dev`, `gamma_gn_dev`) are not `PriorDistribution` fields and cancel in the ratio.
5. **Support:** the new prior's support must lie inside the reference prior's. A wider `ec_m` truncation, for example, needs posterior mass the reference draws cannot contain.
6. **Prior substitution:** when `kpi_type="non_revenue"`, `revenue_per_kpi` is missing and the default ROI prior is used, `context.py` replaces the `roi_m` / `roi_rf` prior with a total-media-contribution prior. The ratio must use the prior in force, not `ModelSpec.prior.roi_m`. The demo data supplies `revenue_per_conversion`, so this does not trigger.
7. **National models (1 geo)** turn `tau_g_excl_baseline`, `eta_om` and `xi_*` into `Deterministic(0)`: not reweightable.
8. **`sigma`** is a scalar unless `unique_sigma_for_each_geo=True` (then `[n_geos]`).
9. **With `enable_aks=True`** the number of knots is chosen from the data. A prior change does not alter it, but the reference fit and the refit must use the same selection.

## Parameters

| Parameter | Default prior | Reweightable | Evidence |
|---|---|---|---|
| `roi_m` | LogNormal(0.2, 0.9) | yes | `log_prob` parity ~1e-15; 15 scenarios x 2 refits: k̂ ≤ 0.42 close to refit noise, k̂ 0.88 and 1.31 clearly above it (`regional_roi/RESULTS_benchmark.md`, `RESULTS_noise.md`); `national_300/`, `regional_60/` |
| `alpha_m` | Uniform(0, 1) | yes, same-support change (e.g. Beta) | 5 targets, 2 refits each: k̂ ≤ 0.50 close to refit noise, k̂ 0.78 and 2.78 clearly above it |
| `ec_m` | TruncatedNormal(0.8, 0.8, 0.1, 10) | yes, keep the truncation bounds | 5 targets, 2 refits each: clean at k̂ 0.09, 2-3x refit noise at k̂ 0.28-0.44 with ESS fraction ≤ 0.11, failing at k̂ ≥ 0.80 |
| `eta_m` | HalfNormal(1) | partly | 6 targets, 2 refits each: k̂ ≤ 0.0 and 0.58 close to refit noise; HalfNormal(2.0) (k̂ 0.44) under-recovered the shift by 43% because the reference lacks the upper tail |
| `sigma` | HalfNormal(5) | yes | 5 targets, 2 refits each: k̂ ≤ 0.42 at refit noise for the mean, k̂ 1.33 failing |
| `alpha_m` + `ec_m` together | | yes, sum the log ratios | joint ratio equals the sum of the single-factor ratios exactly; at refit noise for k̂ ≈ 0, limited at k̂ 0.75-0.88 |
| channel-specific `alpha_m` | | yes | 2 scenarios, both at refit noise; unchanged channels contribute exactly 0 |
| `slope_m`, `slope_om` | Deterministic(1.0) | **no** | all reference draws are exactly 1.0 |
| `alpha_om`, `ec_om`, `eta_om`, `contribution_om`, `contribution_n`, `xi_n`, `gamma_c`, `xi_c`, `tau_g_excl_baseline`, `knot_values` | Uniform / TruncatedNormal / HalfNormal / Beta(1, 99) / TruncatedNormal(0, 0.1, -1, 1) / Normal(0, 5) | random variables, not benchmarked | |
| `*_rf`, `*_orf` | | not tested (no reach-and-frequency data); `slope_rf`, `slope_orf` are stochastic by source default | |
| prior-type switches (e.g. ROI to mROI), `adstock_decay_spec`, `saturation_spec`, `max_lag`, `hill_before_adstock`, `knots`, `enable_aks`, `media_effects_dist`, `unique_sigma_for_each_geo`, data | | **no**: the model structure or the sampled variable changes | |

Details of the evidence rows are in `regional_other_params/RESULTS.md`.

## Other treatment parameterizations

Each is a separate reference model: the sampled treatment node changes, the rest stays.

| `media_prior_type` | Sampled node | Default | `beta_m` |
|---|---|---|---|
| `roi` | `roi_m` | LogNormal(0.2, 0.9) | deterministic |
| `mroi` | `mroi_m` | LogNormal(0, 0.5) | deterministic |
| `contribution` | `contribution_m` | Beta(1, 99) | deterministic |
| `coefficient` | `beta_m` | HalfNormal(5) | random variable |

With `organic_media_prior_type="coefficient"` and `non_media_treatments_prior_type="coefficient"`, `beta_om` is HalfNormal(5) and `gamma_n` is Normal(0, 5), both sampled. Meridian warns that coefficient priors are not recommended (`spec.py:655`).

## What the posterior stores

`InferenceData.posterior` has `roi_m`, `alpha_m`, `ec_m`, `eta_m`, `slope_m`, `beta_m`, `beta_gm` (media); `alpha_om`, `ec_om`, `eta_om`, `slope_om`, `contribution_om`, `beta_om`, `beta_gom` (organic); `contribution_n`, `xi_n`, `gamma_n`, `gamma_gn` (non-media); `gamma_c`, `xi_c`, `gamma_gc` (controls); `sigma`, `knot_values`, `mu_t`, `tau_g`. Dims are `(chain, draw, <channel/geo/knot/time>)`; `sigma` is `(chain, draw)`.

- **Not stored:** `tau_g_excl_baseline` (only `tau_g` over all geos is) and the N(0,1) latents. The latents cancel. A `tau_g_excl_baseline` ratio would need the draws rebuilt from `tau_g`; not checked.
- `slope_m` is stored (constant 1.0), which does not make it reweightable.
- `sample_stats` has `diverging`, `lp`, `n_steps`, `step_size`. The `InferenceData` also carries a `trace` group outside the ArviZ schema, so only `posterior` and `sample_stats` are saved to NetCDF.

## Not verified

- Rebuilding `tau_g_excl_baseline` from `tau_g`.
- Reach-and-frequency (`*_rf`, `*_orf`) behaviour.
- Joint or conditional prior factors: everything here is an independent marginal factor.
