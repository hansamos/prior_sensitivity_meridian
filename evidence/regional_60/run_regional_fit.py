"""Regional long reference fit (40 geos, 156 weeks): same model, priors and MCMC as the regional reference except n_keep.
Usage: python -I evidence/regional_60/run_regional_fit.py --seed 9101 --n_keep 5000 [--smoke]
Reference priors: roi_m LogNormal(0.2, 0.9), everything else Meridian defaults (eta_m HalfNormal(1)).
"""
import argparse, datetime, json, os, sys, time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common",):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import arviz as az
import numpy as np
import pandas as pd
from meridian import constants
from meridian.model import model, prior_distribution, spec
import tensorflow_probability.substrates.jax as tfp_jax
from demo_data import build_data
from fit_config import BASE_MU, BASE_SIGMA, HashableDict, MODES, SMOKE_MCMC, hardware, versions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=9101)
    ap.add_argument("--n_keep", type=int, default=5000)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out_root", default=os.path.join(HERE, "..", "..", "runs", "regional"))
    a = ap.parse_args()
    cfg = MODES["strong"]
    mcmc = dict(SMOKE_MCMC if a.smoke else cfg["mcmc"])
    if not a.smoke:
        mcmc["n_keep"] = a.n_keep
    run_id = f"reg_longref{'_SMOKE' if a.smoke else ''}_seed{a.seed}_{datetime.datetime.now():%Y%m%d-%H%M%S}"
    out = os.path.abspath(os.path.join(a.out_root, run_id)); os.makedirs(out, exist_ok=False)
    print("run_id:", run_id, "mcmc:", mcmc, flush=True)
    data = build_data(cfg["n_geos"], cfg["n_times"])
    prior = prior_distribution.PriorDistribution(
        roi_m=tfp_jax.distributions.LogNormal(np.float64(BASE_MU), np.float64(BASE_SIGMA), name=constants.ROI_M))
    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(prior=prior, enable_aks=cfg["enable_aks"],
                                                                     media_prior_type="roi"))
    t0 = time.perf_counter()
    mmm.sample_posterior(**mcmc, seed=a.seed, dual_averaging_kwargs=HashableDict(target_accept_prob=a.target_accept))
    fit_seconds = time.perf_counter() - t0
    print(f"fit seconds: {fit_seconds:.1f}", flush=True)
    idata = mmm.inference_data
    post = idata.posterior
    rh, eb, et, mc = az.rhat(post), az.ess(post, method="bulk"), az.ess(post, method="tail"), az.mcse(post, method="mean")
    diag = pd.DataFrame([dict(variable=v, rhat_max=float(rh[v].max()), ess_bulk_min=float(eb[v].min()),
                              ess_tail_min=float(et[v].min()), mcse_mean_max=float(mc[v].max())) for v in post.data_vars])
    diag.to_csv(os.path.join(out, "diagnostics.csv"), index=False)
    post[["roi_m", "eta_m", "alpha_m", "ec_m"]].to_netcdf(os.path.join(out, "posterior.nc"))
    div = np.asarray(idata.sample_stats["diverging"]).sum(axis=1).tolist()
    meta = dict(run_id=run_id, seed=a.seed, mcmc=mcmc, target_accept_prob=a.target_accept, fit_seconds=fit_seconds,
                n_geos=int(data.kpi.shape[0]), n_times=int(data.kpi.shape[1]), divergences_per_chain=div,
                worst=dict(rhat_max=float(diag.rhat_max.max()), worst_rhat_variable=str(diag.loc[diag.rhat_max.idxmax(), "variable"]),
                           ess_bulk_min=float(diag.ess_bulk_min.min()), worst_ess_variable=str(diag.loc[diag.ess_bulk_min.idxmin(), "variable"])),
                key={v: diag[diag.variable == v].iloc[0].to_dict() for v in ("roi_m", "eta_m", "alpha_m", "ec_m")},
                versions=versions(), hardware=hardware(), finished=datetime.datetime.now().isoformat())
    json.dump(meta, open(os.path.join(out, "run_metadata.json"), "w"), indent=1, default=str)
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
