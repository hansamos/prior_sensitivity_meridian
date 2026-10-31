"""Phase 3: one full refit under a target ROI prior (one scenario per Slurm array task).

Usage: python -I evidence/regional_roi/run_refit.py --index 1..15 [--mode strong] [--target_accept 0.95] [--smoke]
Same data, model structure and MCMC settings as the reference fit (evidence/regional_roi/run_reference_fit.py);
only the roi_m prior differs. Writes runs/phase3_refits/<scenario>_seed<seed>/ with
run_metadata.json, reference-style diagnostics, roi_m_summary.csv, posterior.nc, sample_stats.nc.
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
for sub in ("common", "regional_roi"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import arviz as az
import numpy as np
import pandas as pd

from meridian import constants
from meridian.model import model, prior_distribution, spec
import tensorflow_probability.substrates.jax as tfp_jax
from demo_data import build_data
from prior_reweight import relative_efficiency
from fit_config import HashableDict, MODES, SMOKE_MCMC, hardware, versions
from scenarios import REFIT_SEED_BASE, SCENARIOS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True, help="1-based scenario index")
    ap.add_argument("--mode", default="strong", choices=MODES)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true", help="tiny MCMC to test this script only")
    ap.add_argument("--seed_base", type=int, default=REFIT_SEED_BASE,
                    help="seed = seed_base + index; Phase 3 refits A used 1000, Phase 4 refits B use 2000")
    ap.add_argument("--out_root", default=os.path.join(HERE, "..", "..", "runs", "phase3_refits"))
    a = ap.parse_args()

    scenario, mu, sigma, family = SCENARIOS[a.index - 1]
    seed = a.seed_base + a.index
    cfg = MODES[a.mode]
    mcmc = SMOKE_MCMC if a.smoke else cfg["mcmc"]
    run_id = f"{scenario}{'_SMOKE' if a.smoke else ''}_seed{seed}"
    out = os.path.abspath(os.path.join(a.out_root, run_id))
    os.makedirs(out, exist_ok=False)
    print("run_id:", run_id, "mu, sigma:", mu, sigma, "\nout:", out, flush=True)

    data = build_data(cfg["n_geos"], cfg["n_times"])
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp_jax.distributions.LogNormal(np.float64(mu), np.float64(sigma), name=constants.ROI_M))
    mmm = model.Meridian(
        input_data=data,
        model_spec=spec.ModelSpec(prior=prior, enable_aks=cfg["enable_aks"], media_prior_type="roi"))

    sample_kwargs = dict(dual_averaging_kwargs=HashableDict(target_accept_prob=a.target_accept))
    t0 = time.perf_counter()
    mmm.sample_posterior(**mcmc, seed=seed, **sample_kwargs)
    fit_seconds = time.perf_counter() - t0
    print(f"fit seconds: {fit_seconds:.1f}", flush=True)

    idata = mmm.inference_data
    post, stats = idata.posterior, idata.sample_stats

    rhat = az.rhat(post)
    ess_bulk = az.ess(post, method="bulk")
    ess_tail = az.ess(post, method="tail")
    mcse_mean = az.mcse(post, method="mean")
    diag = pd.DataFrame([dict(
        variable=v, rhat_max=float(rhat[v].max()), ess_bulk_min=float(ess_bulk[v].min()),
        ess_tail_min=float(ess_tail[v].min()), mcse_mean_max=float(mcse_mean[v].max()))
        for v in post.data_vars])
    diag.to_csv(os.path.join(out, "reference_diagnostics.csv"), index=False)

    roi_np = np.asarray(post[constants.ROI_M])
    assert post[constants.ROI_M].dims[:2] == ("chain", "draw"), post[constants.ROI_M].dims
    ch = [str(c) for c in post[constants.ROI_M].coords["media_channel"].values]
    roi_tbl = pd.DataFrame(dict(
        channel=ch, mean=roi_np.mean((0, 1)), sd=roi_np.std((0, 1), ddof=1),
        q05=np.quantile(roi_np, 0.05, axis=(0, 1)), q50=np.quantile(roi_np, 0.5, axis=(0, 1)),
        q95=np.quantile(roi_np, 0.95, axis=(0, 1)),
        rhat=np.asarray(rhat[constants.ROI_M]), ess_bulk=np.asarray(ess_bulk[constants.ROI_M]),
        ess_tail=np.asarray(ess_tail[constants.ROI_M]), mcse_mean=np.asarray(mcse_mean[constants.ROI_M])))
    roi_tbl.to_csv(os.path.join(out, "roi_m_summary.csv"), index=False)

    sampler = {}
    for v in stats.data_vars:
        arr = np.asarray(stats[v])
        if arr.dtype == bool or "diverg" in v.lower():
            sampler[v] = dict(n_true=int(arr.sum()), n_total=int(arr.size))
        elif np.issubdtype(arr.dtype, np.number):
            sampler[v] = dict(mean=float(np.nanmean(arr)), min=float(np.nanmin(arr)), max=float(np.nanmax(arr)))
    div_per_chain = np.asarray(stats["diverging"]).sum(axis=1).tolist() if "diverging" in stats else None

    saved = {}
    for g in ("posterior", "sample_stats"):
        try:
            getattr(idata, g).to_netcdf(os.path.join(out, f"{g}.nc"))
            saved[g] = "ok"
        except Exception as e:  # noqa: BLE001
            saved[g] = f"FAILED: {type(e).__name__}: {e}"

    meta = dict(
        run_id=run_id, kind="target_refit", smoke=a.smoke, scenario=scenario, family=family,
        target_prior=dict(roi_m=f"LogNormal(mu={mu}, sigma={sigma})"),
        mode=a.mode, seed=seed, mcmc=mcmc, target_accept_prob=a.target_accept, enable_aks=cfg["enable_aks"],
        data=dict(n_geos=int(data.kpi.shape[0]), n_times=int(data.kpi.shape[1])),
        n_chains=int(roi_np.shape[0]), n_draws_per_chain=int(roi_np.shape[1]),
        n_retained_total=int(roi_np.shape[0] * roi_np.shape[1]),
        fit_seconds=fit_seconds, fit_includes_jit_compilation=True,
        roi_m=dict(rhat_max=float(roi_tbl.rhat.max()), ess_bulk_min=float(roi_tbl.ess_bulk.min()),
                   ess_tail_min=float(roi_tbl.ess_tail.min()), mcse_mean_max=float(roi_tbl.mcse_mean.max()),
                   reff=relative_efficiency([roi_np])),
        all_variables=dict(rhat_max=float(diag.rhat_max.max()), ess_bulk_min=float(diag.ess_bulk_min.min()),
                           worst_rhat_variable=str(diag.loc[diag.rhat_max.idxmax(), "variable"])),
        divergences_per_chain=div_per_chain, sampler_stats=sampler, saved=saved,
        versions=versions(), hardware=hardware(),
        finished=datetime.datetime.now().isoformat(timespec="seconds"),
    )
    with open(os.path.join(out, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=1, default=str)
    print(json.dumps({k: meta[k] for k in ("scenario", "fit_seconds", "roi_m", "divergences_per_chain", "saved")}, indent=1))
    print("DONE", out)


if __name__ == "__main__":
    main()
