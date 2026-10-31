"""Fit the ROI-prior reference model once and save chain-aware diagnostics.

Usage: python -I evidence/regional_roi/run_reference_fit.py --mode cpu_pilot --seed 123 [--smoke]
Writes runs/<run_id>/ with run_metadata.json, posterior_variables.csv,
reference_diagnostics.csv, roi_m_summary.csv, posterior.nc and sample_stats.nc.
No benchmark numbers are produced here; this only characterises the reference fit.
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
sys.path.insert(0, os.path.join(HERE, "..", "common"))

import arviz as az
import jax
import numpy as np
import pandas as pd

from meridian import constants
from meridian.model import model, prior_distribution, spec
import tensorflow_probability.substrates.jax as tfp_jax
from demo_data import build_data
from fit_config import BASE_MU, BASE_SIGMA, HashableDict, MODES, SMOKE_MCMC, hardware, versions
from prior_reweight import relative_efficiency

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="cpu_pilot", choices=MODES)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--smoke", action="store_true", help="tiny MCMC to test this script only")
    ap.add_argument("--target_accept", type=float, default=None,
                    help="NUTS target_accept_prob via dual_averaging_kwargs (Meridian/TFP default 0.85)")
    ap.add_argument("--out_root", default=os.path.join(HERE, "..", "..", "runs"))
    a = ap.parse_args()

    cfg = MODES[a.mode]
    mcmc = SMOKE_MCMC if a.smoke else cfg["mcmc"]
    sample_kwargs = {}
    if a.target_accept is not None:
        sample_kwargs["dual_averaging_kwargs"] = HashableDict(target_accept_prob=a.target_accept)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ta = f"_ta{a.target_accept}" if a.target_accept is not None else ""
    run_id = f"ref_{a.mode}{'_SMOKE' if a.smoke else ''}{ta}_seed{a.seed}_{stamp}"
    out = os.path.abspath(os.path.join(a.out_root, run_id))
    os.makedirs(out, exist_ok=False)
    print("run_id:", run_id, "\nout:", out, flush=True)

    data = build_data(cfg["n_geos"], cfg["n_times"])
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp_jax.distributions.LogNormal(np.float64(BASE_MU), np.float64(BASE_SIGMA), name=constants.ROI_M))
    mmm = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(prior=prior, enable_aks=cfg["enable_aks"], media_prior_type="roi"))

    t0 = time.perf_counter()
    mmm.sample_posterior(**mcmc, seed=a.seed, **sample_kwargs)
    fit_seconds = time.perf_counter() - t0
    print(f"fit seconds: {fit_seconds:.1f}", flush=True)

    idata = mmm.inference_data
    post, stats = idata.posterior, idata.sample_stats

    # ---- variable inventory (resolves the 'posterior names/dims' item in PRIOR_INVENTORY.md)
    rows = [dict(group="posterior", name=v, dims=",".join(post[v].dims), shape=list(post[v].shape))
            for v in post.data_vars]
    rows += [dict(group="sample_stats", name=v, dims=",".join(stats[v].dims), shape=list(stats[v].shape))
             for v in stats.data_vars]
    pd.DataFrame(rows).to_csv(os.path.join(out, "posterior_variables.csv"), index=False)

    # ---- chain-aware diagnostics BEFORE flattening, for every posterior variable
    rhat = az.rhat(post)
    ess_bulk = az.ess(post, method="bulk")
    ess_tail = az.ess(post, method="tail")
    mcse_mean = az.mcse(post, method="mean")
    diag = []
    for v in post.data_vars:
        diag.append(dict(
            variable=v, n_elements=int(np.prod(post[v].shape[2:])) if post[v].ndim > 2 else 1,
            rhat_max=float(rhat[v].max()), ess_bulk_min=float(ess_bulk[v].min()),
            ess_tail_min=float(ess_tail[v].min()), mcse_mean_max=float(mcse_mean[v].max())))
    diag = pd.DataFrame(diag)
    diag.to_csv(os.path.join(out, "reference_diagnostics.csv"), index=False)

    # ---- roi_m: per-channel summary, diagnostics, reff
    roi = post[constants.ROI_M]
    assert roi.dims[:2] == ("chain", "draw"), roi.dims
    n_chain, n_draw = roi.shape[:2]
    roi_np = np.asarray(roi)
    ch = [str(c) for c in roi.coords["media_channel"].values]
    roi_tbl = pd.DataFrame(dict(
        channel=ch, mean=roi_np.mean((0, 1)), sd=roi_np.std((0, 1), ddof=1),
        q05=np.quantile(roi_np, 0.05, axis=(0, 1)), q50=np.quantile(roi_np, 0.5, axis=(0, 1)),
        q95=np.quantile(roi_np, 0.95, axis=(0, 1)),
        rhat=np.asarray(rhat[constants.ROI_M]), ess_bulk=np.asarray(ess_bulk[constants.ROI_M]),
        ess_tail=np.asarray(ess_tail[constants.ROI_M]), mcse_mean=np.asarray(mcse_mean[constants.ROI_M])))
    roi_tbl.to_csv(os.path.join(out, "roi_m_summary.csv"), index=False)
    reff_roi = relative_efficiency([roi_np])

    # ---- sampler diagnostics, whatever this version exposes
    sampler = {}
    for v in stats.data_vars:
        arr = np.asarray(stats[v])
        if arr.dtype == bool or "diverg" in v.lower():
            sampler[v] = dict(n_true=int(arr.sum()), n_total=int(arr.size))
        elif np.issubdtype(arr.dtype, np.number):
            sampler[v] = dict(mean=float(np.nanmean(arr)), min=float(np.nanmin(arr)), max=float(np.nanmax(arr)))

    # ---- save draws (posterior + sample_stats only; the extra 'trace' group is not in the arviz schema)
    saved = {}
    for g in ("posterior", "sample_stats"):
        try:
            getattr(idata, g).to_netcdf(os.path.join(out, f"{g}.nc"))
            saved[g] = "ok"
        except Exception as e:  # noqa: BLE001
            saved[g] = f"FAILED: {type(e).__name__}: {e}"

    meta = dict(
        run_id=run_id, kind="reference_fit", smoke=a.smoke, mode=a.mode, seed=a.seed,
        mcmc=mcmc, sample_kwargs=sample_kwargs, target_accept_prob=(a.target_accept if a.target_accept is not None else 0.85),
        target_accept_is_default=(a.target_accept is None), enable_aks=cfg["enable_aks"],
        data=dict(n_geos=int(data.kpi.shape[0]), n_times=int(data.kpi.shape[1]),
                  geos=[str(g) for g in data.kpi.coords["geo"].values],
                  time_first=str(data.kpi.coords["time"].values[0]), time_last=str(data.kpi.coords["time"].values[-1]),
                  media_channels=ch),
        reference_prior=dict(roi_m=f"LogNormal(mu={BASE_MU}, sigma={BASE_SIGMA})", media_prior_type="roi"),
        n_chains=int(n_chain), n_draws_per_chain=int(n_draw), n_retained_total=int(n_chain * n_draw),
        fit_seconds=fit_seconds, fit_includes_jit_compilation=True,
        roi_m=dict(rhat_max=float(roi_tbl.rhat.max()), ess_bulk_min=float(roi_tbl.ess_bulk.min()),
                   ess_tail_min=float(roi_tbl.ess_tail.min()), mcse_mean_max=float(roi_tbl.mcse_mean.max()),
                   reff_for_psis=reff_roi),
        all_variables=dict(rhat_max=float(diag.rhat_max.max()), ess_bulk_min=float(diag.ess_bulk_min.min()),
                           ess_tail_min=float(diag.ess_tail_min.min()),
                           worst_rhat_variable=str(diag.loc[diag.rhat_max.idxmax(), "variable"]),
                           worst_ess_variable=str(diag.loc[diag.ess_bulk_min.idxmin(), "variable"])),
        sampler_stats=sampler, saved=saved, versions=versions(), hardware=hardware(),
    )
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps({k: meta[k] for k in ("n_retained_total", "fit_seconds", "roi_m", "all_variables", "saved")}, indent=1))
    print("DONE", out)


if __name__ == "__main__":
    main()
