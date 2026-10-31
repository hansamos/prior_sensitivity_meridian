"""Regional test, step 2: one full refit (40 geos, 156 weeks) for one target prior (one target per Slurm array task).

Usage: python -I evidence/regional_60/run_regional_refit.py --index 1..60 --which A|B --out_root runs/regional_targets_A
Target i = row i of evidence/regional_60/calibration/regional_targets.csv. Same data, ModelSpec and MCMC as the regional reference fit
(4 chains, 1000 adapt, 500 burn-in, 1000 kept, target_accept 0.95, AKS on); roi_m stays LogNormal(0.2, 0.9) and eta_m stays
HalfNormal(1) unless it is the perturbed parameter. Saves roi_m, eta_m, alpha_m, ec_m draws plus diagnostics.
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
from meridian.model import model, prior_distribution, spec
import tensorflow_probability.substrates.jax as tfp_jax
from demo_data import build_data
from fit_config import BASE_MU, BASE_SIGMA, HashableDict, MODES, hardware

tfd = tfp_jax.distributions
f64 = np.float64
KEEP = ["roi_m", "eta_m", "alpha_m", "ec_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--which", default="A", choices=("A", "B"))
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--targets", default=os.path.join(HERE, "calibration", "regional_targets.csv"))
    ap.add_argument("--target_accept", type=float, default=0.95)
    a = ap.parse_args()
    row = next(pd.read_csv(a.targets).iloc[[a.index - 1]].itertuples(index=False))
    assert row.index == a.index
    if a.which == "B" and not row.second_refit:
        print("not a second-refit target; nothing to do", row.target_id); return
    seed = int(row.seed_A if a.which == "A" else row.seed_B)
    out = os.path.abspath(os.path.join(a.out_root, f"{row.target_id}_seed{seed}")); os.makedirs(out, exist_ok=False)
    print("target:", row.target_id, "param:", row.param, "KL:", row.KL, "seed:", seed, flush=True)

    kw = {"roi_m": tfd.LogNormal(f64(BASE_MU), f64(BASE_SIGMA), name="roi_m")}
    if row.param == "roi_m":
        kw["roi_m"] = tfd.LogNormal(f64(row.p1), f64(row.p2), name="roi_m")
    else:
        kw["eta_m"] = tfd.HalfNormal(f64(row.p1), name="eta_m")
    cfg = MODES["strong"]
    data = build_data(cfg["n_geos"], cfg["n_times"])
    mmm = model.Meridian(input_data=data, model_spec=spec.ModelSpec(
        prior=prior_distribution.PriorDistribution(**kw), enable_aks=cfg["enable_aks"], media_prior_type="roi"))
    t0 = time.perf_counter()
    mmm.sample_posterior(**cfg["mcmc"], seed=seed, dual_averaging_kwargs=HashableDict(target_accept_prob=a.target_accept))
    fit_seconds = time.perf_counter() - t0

    idata = mmm.inference_data
    post = idata.posterior
    rh, eb, et, mc = az.rhat(post), az.ess(post, method="bulk"), az.ess(post, method="tail"), az.mcse(post, method="mean")
    diag = {v: dict(rhat_max=float(rh[v].max()), ess_bulk_min=float(eb[v].min()), ess_tail_min=float(et[v].min()),
                    mcse_mean_max=float(mc[v].max())) for v in KEEP}
    allv = dict(rhat_max=float(max(float(rh[v].max()) for v in post.data_vars)),
                ess_bulk_min=float(min(float(eb[v].min()) for v in post.data_vars)))
    post[KEEP].to_netcdf(os.path.join(out, "posterior_media.nc"))
    div = int(np.asarray(idata.sample_stats["diverging"]).sum())
    meta = dict(target_id=row.target_id, param=row.param, tier=int(row.tier), which=a.which, seed=seed, index=a.index,
                target_row=row._asdict(), fit_seconds=fit_seconds, key_diagnostics=diag, all_variables=allv,
                divergences=div, mcmc=cfg["mcmc"], n_geos=int(data.kpi.shape[0]), n_times=int(data.kpi.shape[1]),
                hardware=hardware(), finished=datetime.datetime.now().isoformat())
    json.dump(meta, open(os.path.join(out, "run_metadata.json"), "w"), indent=1, default=str)
    print("DONE", out, f"{fit_seconds:.0f}s", flush=True)


if __name__ == "__main__":
    main()
