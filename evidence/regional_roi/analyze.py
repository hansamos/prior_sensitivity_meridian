"""Phase 3 analysis: raw IS / PSIS from the reference fit versus the 15 full refits.

Usage: python -I evidence/regional_roi/analyze.py --ref runs/<reference run> --refits runs/phase3_refits --out evidence/regional_roi/results
Reads posterior.nc (+ run_metadata.json) only; refits MUST be done. Writes CSVs, no prose.
All summaries use one estimator (midpoint-CDF weighted quantile, uniform weights for refits).
"""
import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "regional_roi"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import numpy as np
import pandas as pd
import xarray as xr

import meridian.backend  # noqa: F401  (enables JAX x64, required by reweight)
import tensorflow_probability.substrates.jax as tfp_jax
from prior_reweight import PriorFactor, reweight, relative_efficiency
from scenarios import BASE_MU, BASE_SIGMA, SCENARIOS

tfd = tfp_jax.distributions
STATS = ("mean", "q05", "q50", "q95")


def wquantile(x, w, q):
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cdf = (np.cumsum(ws) - 0.5 * ws) / np.sum(ws)
    return np.interp(q, cdf, xs)


def summarize(draws, w):
    return {j: dict(mean=float(np.sum(w * draws[:, j]) / np.sum(w)),
                    q05=wquantile(draws[:, j], w, 0.05), q50=wquantile(draws[:, j], w, 0.5),
                    q95=wquantile(draws[:, j], w, 0.95)) for j in range(draws.shape[1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--refits", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    post = xr.open_dataset(os.path.join(a.ref, "posterior.nc"))
    roi = np.asarray(post["roi_m"])                       # (chain, draw, channel)
    channels = [str(c) for c in post["roi_m"].coords["media_channel"].values]
    ref_meta = json.load(open(os.path.join(a.ref, "run_metadata.json")))
    n_ch, n_dr, J = roi.shape
    flat = roi.reshape(-1, J)                             # chain-major
    reff = relative_efficiency([roi])
    S = flat.shape[0]
    print(f"reference {os.path.basename(a.ref)}: S={S}, J={J}, reff={reff:.3f}")

    ref_prior = tfd.LogNormal(np.float64(BASE_MU), np.float64(BASE_SIGMA))
    scen_rows, chan_rows = [], []
    for idx, (name, mu, sigma, family) in enumerate(SCENARIOS, start=1):
        dirs = sorted(glob.glob(os.path.join(a.refits, f"{name}_seed*")))
        dirs = [d for d in dirs if "_SMOKE" not in d]
        if len(dirs) != 1 or not os.path.exists(os.path.join(dirs[0], "run_metadata.json")):
            print(f"  {name}: refit missing or ambiguous ({dirs}); skipped")
            continue
        meta = json.load(open(os.path.join(dirs[0], "run_metadata.json")))
        rd = np.asarray(xr.open_dataset(os.path.join(dirs[0], "posterior.nc"))["roi_m"])
        rflat = rd.reshape(-1, J)

        # cold first call for scenario 1 includes any JAX compile; repeat timing is taken after
        target = tfd.LogNormal(np.float64(mu), np.float64(sigma))
        factor = PriorFactor("roi_m", ref_prior, target)
        t0 = time.perf_counter()
        res = reweight({"roi_m": flat}, [factor], reff=reff)
        core_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        res = reweight({"roi_m": flat}, [factor], reff=reff)
        core_repeat_s = time.perf_counter() - t0

        raw = res.raw_weights
        psis = res.psis_weights if res.psis_weights is not None else res.raw_weights
        t0 = time.perf_counter()
        s_raw, s_psis = summarize(flat, raw), summarize(flat, psis)
        summary_s = time.perf_counter() - t0
        s_ref = summarize(rflat, np.full(rflat.shape[0], 1.0 / rflat.shape[0]))
        s_base = summarize(flat, np.full(S, 1.0 / S))

        for j, ch in enumerate(channels):
            for st in STATS:
                chan_rows.append(dict(
                    scenario=name, channel=ch, stat=st, refit=s_ref[j][st], raw_is=s_raw[j][st],
                    psis=s_psis[j][st], reference_unweighted=s_base[j][st],
                    err_raw=s_raw[j][st] - s_ref[j][st], err_psis=s_psis[j][st] - s_ref[j][st],
                    err_noadjust=s_base[j][st] - s_ref[j][st]))

        sd_ref = rflat.std(axis=0, ddof=1)
        row = dict(
            scenario=name, family=family, mu=mu, sigma=sigma, status=res.status,
            pareto_k=res.pareto_k, k_threshold=res.k_threshold, raw_weight_ess=res.raw_ess,
            psis_weight_ess=res.psis_ess, ess_fraction=res.ess_fraction, n_zero_weight=res.n_zero_weight,
            refit_rhat_max=meta["roi_m"]["rhat_max"], refit_ess_bulk_min=meta["roi_m"]["ess_bulk_min"],
            refit_mcse_mean_max=meta["roi_m"]["mcse_mean_max"],
            refit_divergences=int(sum(meta["divergences_per_chain"] or [])),
            refit_seed=meta["seed"], refit_seconds=meta["fit_seconds"],
            reference_fit_seconds=ref_meta["fit_seconds"],
            core_reweight_seconds_cold=core_s, core_reweight_seconds_repeat=core_repeat_s,
            summary_seconds=summary_s,
            end_to_end_seconds=core_repeat_s + summary_s)
        for st in STATS:
            e_raw = np.array([s_raw[j][st] - s_ref[j][st] for j in range(J)])
            e_psis = np.array([s_psis[j][st] - s_ref[j][st] for j in range(J)])
            e_no = np.array([s_base[j][st] - s_ref[j][st] for j in range(J)])
            row[f"mae_{st}_raw"], row[f"mae_{st}_psis"], row[f"mae_{st}_noadjust"] = (
                np.abs(e_raw).mean(), np.abs(e_psis).mean(), np.abs(e_no).mean())
            row[f"max_abs_{st}_psis"] = np.abs(e_psis).max()
            row[f"scaled_mae_{st}_psis"] = np.mean(np.abs(e_psis) / sd_ref)   # error in refit-posterior SDs
        scen_rows.append(row)
        print(f"  {name}: k={res.pareto_k:.3f} status={res.status} ess_frac={res.ess_fraction:.3f} "
              f"mae_mean raw/psis={row['mae_mean_raw']:.3f}/{row['mae_mean_psis']:.3f}")

    pd.DataFrame(scen_rows).to_csv(os.path.join(a.out, "scenario_level.csv"), index=False)
    pd.DataFrame(chan_rows).to_csv(os.path.join(a.out, "channel_level.csv"), index=False)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
