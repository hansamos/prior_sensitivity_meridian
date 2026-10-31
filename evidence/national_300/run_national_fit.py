"""National reference-style fit (timing run). Same MCMC settings as the 40-geo strong reference:
4 chains, 1000 adapt, 500 burn-in, 1000 kept, target_accept 0.95, ROI LogNormal(0.2, 0.9), AKS on.
Usage: python -I evidence/national_300/run_national_fit.py --seed 123 --tag cpus4 [--smoke]
"""
import argparse, json, os, sys, time, datetime

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "national_300"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import arviz as az
import numpy as np
import pandas as pd
from meridian import constants
from meridian.model import model, prior_distribution, spec
import tensorflow_probability.substrates.jax as tfp_jax
from demo_data_national import build_national_data
from fit_config import BASE_MU, BASE_SIGMA, HashableDict, MODES, SMOKE_MCMC, hardware, versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tag", default="")
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n_keep", type=int, default=None, help="override kept draws per chain (long reference)")
    ap.add_argument("--out_root", default=os.path.join(HERE, "..", "..", "runs", "national"))
    a = ap.parse_args()
    mcmc = dict(SMOKE_MCMC if a.smoke else MODES["strong"]["mcmc"])
    if a.n_keep:
        mcmc["n_keep"] = a.n_keep
    run_id = f"nat_{a.tag}{'_SMOKE' if a.smoke else ''}_seed{a.seed}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    out = os.path.abspath(os.path.join(a.out_root, run_id)); os.makedirs(out, exist_ok=False)
    print("run_id:", run_id, flush=True)

    data = build_national_data()
    t_build = time.perf_counter()
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp_jax.distributions.LogNormal(np.float64(BASE_MU), np.float64(BASE_SIGMA), name=constants.ROI_M))
    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(prior=prior, enable_aks=True,
                                                                     media_prior_type="roi"))
    n_knots = None
    try:
        n_knots = int(mmm.knot_info.n_knots)
    except Exception as e:  # noqa: BLE001
        n_knots = f"unavailable: {type(e).__name__}"
    t0 = time.perf_counter()
    mmm.sample_posterior(**mcmc, seed=a.seed, dual_averaging_kwargs=HashableDict(target_accept_prob=a.target_accept))
    fit_seconds = time.perf_counter() - t0
    print(f"fit seconds: {fit_seconds:.1f}  knots: {n_knots}", flush=True)

    post, stats = mmm.inference_data.posterior, mmm.inference_data.sample_stats
    rh, eb, et, mc = az.rhat(post), az.ess(post, method="bulk"), az.ess(post, method="tail"), az.mcse(post, method="mean")
    diag = pd.DataFrame([dict(variable=v, rhat_max=float(rh[v].max()), ess_bulk_min=float(eb[v].min()),
                              ess_tail_min=float(et[v].min()), mcse_mean_max=float(mc[v].max()))
                         for v in post.data_vars])
    diag.to_csv(os.path.join(out, "diagnostics.csv"), index=False)
    div = np.asarray(stats["diverging"]).sum(axis=1).tolist() if "diverging" in stats else None
    for g in ("posterior", "sample_stats"):
        getattr(mmm.inference_data, g).to_netcdf(os.path.join(out, f"{g}.nc"))
    key = diag[diag.variable.isin(["roi_m", "alpha_m", "ec_m", "sigma"])].set_index("variable").to_dict("index")
    meta = dict(run_id=run_id, kind="national_timing", smoke=a.smoke, seed=a.seed, tag=a.tag, mcmc=mcmc,
                target_accept_prob=a.target_accept, enable_aks=True, n_knots=n_knots,
                data=dict(n_geos=int(data.kpi.shape[0]), n_times=int(data.kpi.shape[1])),
                fit_seconds=fit_seconds, fit_includes_jit_compilation=True,
                model_build_seconds_before_fit=t0 - t_build, key_diagnostics=key,
                all_variables=dict(rhat_max=float(diag.rhat_max.max()), worst_rhat_variable=str(diag.loc[diag.rhat_max.idxmax(), "variable"]),
                                   ess_bulk_min=float(diag.ess_bulk_min.min()), worst_ess_variable=str(diag.loc[diag.ess_bulk_min.idxmin(), "variable"])),
                divergences_per_chain=div, versions=versions(), hardware=hardware())
    json.dump(meta, open(os.path.join(out, "run_metadata.json"), "w"), indent=1, default=str)
    print(json.dumps({k: meta[k] for k in ("fit_seconds", "n_knots", "key_diagnostics", "all_variables", "divergences_per_chain")}, indent=1, default=str))
    print("DONE", out)


if __name__ == "__main__":
    main()
