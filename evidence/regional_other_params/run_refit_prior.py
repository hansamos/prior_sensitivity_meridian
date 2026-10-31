"""Phase 3B: one full refit under a target prior on one non-ROI field (one scenario per array task).

Usage: python -I evidence/regional_other_params/run_refit_prior.py --index 1..N --seed_base 3000 --out_root runs/phase3b_refits_A
Same data, ModelSpec and MCMC settings as the reference fit (evidence/regional_roi/run_reference_fit.py, strong,
target_accept 0.95). roi_m stays LogNormal(0.2, 0.9); only the scenario's field gets the target prior.
"""
import argparse
import datetime
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "regional_other_params"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import arviz as az
import numpy as np
import pandas as pd

from meridian import constants
from meridian.model import model, prior_distribution, spec
import tensorflow_probability.substrates.jax as tfp_jax
from demo_data import build_data
from prior_reweight import relative_efficiency
from fit_config import BASE_MU, BASE_SIGMA, HashableDict, MODES, SMOKE_MCMC, hardware, versions
from scenarios_other import MULTI_SCENARIOS, SCENARIOS_3B, make


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True, help="1-based index into SCENARIOS_3B")
    ap.add_argument("--seed_base", type=int, required=True)
    ap.add_argument("--mode", default="strong", choices=MODES)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--registry", default="single", choices=("single", "multi"),
                    help="single: SCENARIOS_3B (one field each); multi: MULTI_SCENARIOS (joint / channel-specific)")
    ap.add_argument("--out_root", required=True)
    a = ap.parse_args()

    if a.registry == "single":
        scenario, field, target_spec, role = SCENARIOS_3B[a.index - 1]
        targets = {field: target_spec}
    else:
        scenario, targets, role = MULTI_SCENARIOS[a.index - 1]
        field = "+".join(targets)
    seed = a.seed_base + a.index
    cfg = MODES[a.mode]
    mcmc = SMOKE_MCMC if a.smoke else cfg["mcmc"]
    run_id = f"{scenario}{'_SMOKE' if a.smoke else ''}_seed{seed}"
    out = os.path.abspath(os.path.join(a.out_root, run_id))
    os.makedirs(out, exist_ok=False)
    print("run_id:", run_id, "fields:", field, "targets:", targets, "\nout:", out, flush=True)

    data = build_data(cfg["n_geos"], cfg["n_times"])
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp_jax.distributions.LogNormal(np.float64(BASE_MU), np.float64(BASE_SIGMA), name=constants.ROI_M),
        **{f: make(sp, name=f) for f, sp in targets.items()})
    mmm = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(prior=prior, enable_aks=cfg["enable_aks"], media_prior_type="roi"))

    t0 = time.perf_counter()
    mmm.sample_posterior(**mcmc, seed=seed,
                         dual_averaging_kwargs=HashableDict(target_accept_prob=a.target_accept))
    fit_seconds = time.perf_counter() - t0
    print(f"fit seconds: {fit_seconds:.1f}", flush=True)

    idata = mmm.inference_data
    post, stats = idata.posterior, idata.sample_stats
    rhat, ess_b = az.rhat(post), az.ess(post, method="bulk")
    ess_t, mcse = az.ess(post, method="tail"), az.mcse(post, method="mean")
    diag = pd.DataFrame([dict(
        variable=v, rhat_max=float(rhat[v].max()), ess_bulk_min=float(ess_b[v].min()),
        ess_tail_min=float(ess_t[v].min()), mcse_mean_max=float(mcse[v].max())) for v in post.data_vars])
    diag.to_csv(os.path.join(out, "reference_diagnostics.csv"), index=False)

    # per-channel summary and diagnostics of the changed field (and roi_m, for context)
    summ = []
    for v in sorted(set(targets) | {constants.ROI_M}):
        x = np.asarray(post[v])
        if x.ndim == 2:  # scalar variable such as sigma: (chain, draw) -> one "channel"
            x = x[..., None]
        for j in range(x.shape[-1]):
            summ.append(dict(variable=v, channel=j, mean=x[..., j].mean(), sd=x[..., j].std(ddof=1),
                             q05=np.quantile(x[..., j], .05), q50=np.quantile(x[..., j], .5),
                             q95=np.quantile(x[..., j], .95), rhat=float(np.atleast_1d(np.asarray(rhat[v]))[j]),
                             ess_bulk=float(np.atleast_1d(np.asarray(ess_b[v]))[j]), ess_tail=float(np.atleast_1d(np.asarray(ess_t[v]))[j]),
                             mcse_mean=float(np.atleast_1d(np.asarray(mcse[v]))[j])))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(out, "field_summary.csv"), index=False)
    fx = summ[summ.variable.isin(list(targets))]

    sampler = {}
    for v in stats.data_vars:
        arr = np.asarray(stats[v])
        if arr.dtype == bool or "diverg" in v.lower():
            sampler[v] = dict(n_true=int(arr.sum()), n_total=int(arr.size))
        elif np.issubdtype(arr.dtype, np.number):
            sampler[v] = dict(mean=float(np.nanmean(arr)), min=float(np.nanmin(arr)), max=float(np.nanmax(arr)))
    div = np.asarray(stats["diverging"]).sum(axis=1).tolist() if "diverging" in stats else None

    saved = {}
    for g in ("posterior", "sample_stats"):
        try:
            getattr(idata, g).to_netcdf(os.path.join(out, f"{g}.nc"))
            saved[g] = "ok"
        except Exception as e:  # noqa: BLE001
            saved[g] = f"FAILED: {type(e).__name__}: {e}"

    meta = dict(
        run_id=run_id, kind="target_refit_3b", smoke=a.smoke, scenario=scenario, field=field, role=role,
        target_prior={f: dict(kind=sp[0], params=np.asarray(sp[1], dtype=float).tolist()) for f, sp in targets.items()},
        fixed_priors=dict(roi_m=f"LogNormal(mu={BASE_MU}, sigma={BASE_SIGMA})"),
        mode=a.mode, seed=seed, mcmc=mcmc, target_accept_prob=a.target_accept, enable_aks=cfg["enable_aks"],
        data=dict(n_geos=int(data.kpi.shape[0]), n_times=int(data.kpi.shape[1])),
        n_retained_total=int(np.asarray(post[constants.ROI_M]).shape[0] * np.asarray(post[constants.ROI_M]).shape[1]),
        fit_seconds=fit_seconds, fit_includes_jit_compilation=True,
        field_diag=dict(rhat_max=float(fx.rhat.max()), ess_bulk_min=float(fx.ess_bulk.min()),
                        ess_tail_min=float(fx.ess_tail.min()), mcse_mean_max=float(fx.mcse_mean.max())),
        all_variables=dict(rhat_max=float(diag.rhat_max.max()), ess_bulk_min=float(diag.ess_bulk_min.min()),
                           worst_rhat_variable=str(diag.loc[diag.rhat_max.idxmax(), "variable"])),
        divergences_per_chain=div, sampler_stats=sampler, saved=saved,
        versions=versions(), hardware=hardware(), finished=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps({k: meta[k] for k in ("scenario", "fit_seconds", "field_diag", "divergences_per_chain", "saved")}, indent=1))
    print("DONE", out)


if __name__ == "__main__":
    main()
