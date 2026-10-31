# Prior sensitivity in Google Meridian, without refitting

Checking a prior usually means refitting once per alternative. Instead: **fit once, then reweight the saved MCMC draws** (Pareto-smoothed importance sampling). Pareto k̂ and the effective sample size (ESS) say when to trust the result and when to refit.

| Pareto k̂ | verdict |
|---|---|
| < 0.5 | use the reweighted posterior |
| 0.5 – 0.7 | check it |
| ≥ 0.7 | refit |

## Try it

Two self-contained notebooks. Each fits Meridian's demo model once, then reweights it.

| Notebook | Model | Open |
|---|---|---|
| [notebooks/national.ipynb](notebooks/national.ipynb) | national: `roi_m`, `alpha_m`, `ec_m` | [Colab](https://colab.research.google.com/github/hansamos/prior_sensitivity_meridian/blob/main/notebooks/national.ipynb) |
| [notebooks/regional.ipynb](notebooks/regional.ipynb) | geo-level: `roi_m`, `eta_m` | [Colab](https://colab.research.google.com/github/hansamos/prior_sensitivity_meridian/blob/main/notebooks/regional.ipynb) |

```
pip install -r requirements.txt
jupyter lab notebooks/
```

The core of the method:

```python
log_w = (new_prior.logpdf(roi) - reference_prior.logpdf(roi)).sum(axis=1)   # one number per draw
log_w, k_hat = az.psislw(log_w)                                             # smooth the weights
w = np.exp(log_w); w /= w.sum()
new_roi_mean = (w[:, None] * roi).sum(axis=0)                               # posterior mean under the new prior
```

## Evidence

Reweighted posteriors against independent full refits, Meridian 2.1.0 (JAX), simulated demo data:

| Test | Priors | KL to the refit, median |
|---|---|---|
| National: `roi_m`, `alpha_m`, `ec_m`, 5 tiers of distance from the reference prior | 300 | **0.0056** for k̂ < 0.5 (144 priors), 0.023 for 0.5 – 0.7 (34), 0.23 for ≥ 0.7 (122). Two refits of one prior differ by 0.0086. |
| Regional (40 geos x 156 weeks): `roi_m`, `eta_m`, 3 tiers | 60 | **0.0057** for k̂ < 0.5 (25), 0.018 for 0.5 – 0.7 (6), 0.060 for ≥ 0.7 (29). Two refits differ by 0.011. |
| Regional, `roi_m`, two refits each | 15 | The 13 priors with k̂ ≤ 0.42 are about as far from a refit as two refits are from each other. The two stress cases (k̂ 0.88, 1.31) are 11x and 14x off. |

Limits: simulated data, one reference fit per experiment, few parameters. k̂ alone is not always enough: a few targets with k̂ < 0.5 but a small ESS were off by 2 to 3 times the refit noise. Scripts, tables and figures: [evidence/](evidence/README.md).

## Scope

Only the prior changes: same data, model structure and likelihood. The new prior's support must lie inside the reference prior's. Parameters with a point-mass prior (Meridian's default `slope_m`) cannot be reweighted.

## Reference

Vehtari, A., Simpson, D., Gelman, A., Yao, Y., Gabry, J. (2024). Pareto smoothed importance sampling. *Journal of Machine Learning Research*, 25(72), 1-58.

## License

Apache 2.0, see [LICENSE](LICENSE).
