"""Stage 3: one national refit for a chosen perturbed prior (one target per Slurm array task).

Usage: python -I evidence/national_300/run_target_refit.py --index 1..300 --seed_base 10000 --out_root runs/national_targets_A
Target i = row i (1-based) of evidence/national_300/calibration/chosen_targets.csv. Same data/ModelSpec/MCMC as the national reference
fit (4 chains, 1000 adapt, 500 burn-in, 1000 kept, target_accept 0.95, AKS on); roi_m stays LogNormal(0.2, 0.9) unless it
is the perturbed parameter. Saves only roi_m, alpha_m, ec_m draws plus diagnostics.
"""
import argparse, datetime, json, os, sys, time

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
from fit_config import BASE_MU, BASE_SIGMA, HashableDict, MODES, hardware

tfd = tfp_jax.distributions
f64 = np.float64


def make_prior(row):
    if row.param == "roi_m":
        return tfd.LogNormal(f64(row.mu), f64(row.sigma), name="roi_m")
    if row.param == "alpha_m":
        return tfd.Beta(f64(row.a), f64(row.b), name="alpha_m")
    return tfd.TruncatedNormal(f64(row.loc), f64(row.scale), f64(0.1), f64(10.0), name="ec_m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--seed_base", type=int, required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--targets", default=os.path.join(HERE, "calibration", "chosen_targets.csv"))
    ap.add_argument("--target_accept", type=float, default=0.95)
    a = ap.parse_args()
    row = next(pd.read_csv(a.targets).iloc[[a.index - 1]].itertuples(index=False))
    seed = a.seed_base + a.index
    out = os.path.abspath(os.path.join(a.out_root, f"{row.target_id}_seed{seed}")); os.makedirs(out, exist_ok=False)
    print("target:", row.target_id, "param:", row.param, "KL:", row.KL, flush=True)

    kw = {"roi_m": tfd.LogNormal(f64(BASE_MU), f64(BASE_SIGMA), name="roi_m")}
    kw[row.param] = make_prior(row)
    data = build_national_data()
    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(**kw), enable_aks=True, media_prior_type="roi"))
    mcmc = MODES["strong"]["mcmc"]
    t0 = time.perf_counter()
    mmm.sample_posterior(**mcmc, seed=seed, dual_averaging_kwargs=HashableDict(target_accept_prob=a.target_accept))
    fit_seconds = time.perf_counter() - t0

    idata = mmm.inference_data
    post = idata.posterior
    rh, eb, et, mc = az.rhat(post), az.ess(post, method="bulk"), az.ess(post, method="tail"), az.mcse(post, method="mean")
    keep = ["roi_m", "alpha_m", "ec_m"]
    diag = {v: dict(rhat_max=float(rh[v].max()), ess_bulk_min=float(eb[v].min()), ess_tail_min=float(et[v].min()),
                    mcse_mean_max=float(mc[v].max())) for v in keep}
    allv = dict(rhat_max=float(max(float(rh[v].max()) for v in post.data_vars)),
                ess_bulk_min=float(min(float(eb[v].min()) for v in post.data_vars)))
    post[keep].to_netcdf(os.path.join(out, "posterior_media.nc"))
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    meta = dict(target_id=row.target_id, param=row.param, tier=int(row.tier), seed=seed, index=a.index,
                target_row=row._asdict(), fit_seconds=fit_seconds, key_diagnostics=diag, all_variables=allv,
                divergences=div, mcmc=mcmc, hardware=hardware(), finished=datetime.datetime.now().isoformat())
    json.dump(meta, open(os.path.join(out, "run_metadata.json"), "w"), indent=1, default=str)
    print("DONE", out, f"{fit_seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
