"""Phase 4: PSIS-vs-refit error versus independent-refit (A vs B) variability.

Usage: python -I evidence/regional_roi/analyze_noise.py --ref <reference run> --refits_a runs/phase3_refits \
           --refits_b runs/phase3_refits_B --out evidence/regional_roi/results

For each scenario, channel and statistic (mean, q05, q50, q95; equal-tailed 90% interval endpoints):
  e_psis_A = psis - A      e_psis_B = psis - B      e_AB = A - B
Monte Carlo standard errors come from a moving-block bootstrap within chains (block length BLOCK),
applied to each estimator (reference-PSIS, refit A, refit B). For PSIS the bootstrap resamples
(draw, weight) pairs and renormalises the already-smoothed weights; it does NOT refit the Pareto tail
per replicate, so it understates the smoothing uncertainty. z_AB = e_AB / sqrt(se_A^2 + se_B^2) is the
calibration check of the bootstrap SEs: if they are right, z_AB across many cells has sd about 1.
"""
import argparse
import glob
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "regional_roi"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import numpy as np
import pandas as pd
import xarray as xr

import meridian.backend  # noqa: F401  (JAX x64, required by reweight)
import tensorflow_probability.substrates.jax as tfp_jax
from prior_reweight import PriorFactor, reweight, relative_efficiency
from scenarios import BASE_MU, BASE_SIGMA, SCENARIOS

tfd = tfp_jax.distributions
STATS = ("mean", "q05", "q50", "q95")
QS = {"q05": 0.05, "q50": 0.5, "q95": 0.95}
BLOCK, N_BOOT = 50, 300


def wsummary(x, w):
    """x (S, J), w (S,) -> dict stat -> (J,). Midpoint-CDF weighted quantiles, as in Phase 3."""
    w = w / w.sum()
    out = {"mean": (w[:, None] * x).sum(0)}
    order = np.argsort(x, axis=0)
    xs = np.take_along_axis(x, order, 0)
    ws = w[order]
    cdf = np.cumsum(ws, 0) - 0.5 * ws
    for name, q in QS.items():
        out[name] = np.array([np.interp(q, cdf[:, j], xs[:, j]) for j in range(x.shape[1])])
    return out


def block_indices(rng, n_chain, n_draw):
    nb = n_draw // BLOCK
    idx = []
    for c in range(n_chain):
        starts = rng.integers(0, n_draw - BLOCK + 1, size=nb)
        idx.append(c * n_draw + (starts[:, None] + np.arange(BLOCK)).ravel())
    return np.concatenate(idx)


def boot_se(x, w, n_chain, n_draw, rng):
    reps = [wsummary(x[i], w[i]) for i in (block_indices(rng, n_chain, n_draw) for _ in range(N_BOOT))]
    return {s: np.std([r[s] for r in reps], axis=0, ddof=1) for s in STATS}


def load(d):
    ds = xr.open_dataset(os.path.join(d, "posterior.nc"))
    meta = json.load(open(os.path.join(d, "run_metadata.json")))
    return np.asarray(ds["roi_m"]), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--refits_a", required=True)
    ap.add_argument("--refits_b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(20261029)

    ref, _ = load(a.ref)
    n_ch, n_dr, J = ref.shape
    flat = ref.reshape(-1, J)
    S = flat.shape[0]
    reff = relative_efficiency([ref])
    channels = [str(c) for c in xr.open_dataset(os.path.join(a.ref, "posterior.nc"))["roi_m"].coords["media_channel"].values]
    ref_prior = tfd.LogNormal(np.float64(BASE_MU), np.float64(BASE_SIGMA))

    rows = []
    for name, mu, sigma, family in SCENARIOS:
        da = [d for d in glob.glob(os.path.join(a.refits_a, f"{name}_seed*")) if "SMOKE" not in d]
        db = [d for d in glob.glob(os.path.join(a.refits_b, f"{name}_seed*")) if "SMOKE" not in d]
        if len(da) != 1 or len(db) != 1:
            print(f"{name}: A/B refit missing or ambiguous; skipped")
            continue
        A, ma = load(da[0])
        B, mb = load(db[0])
        res = reweight({"roi_m": flat}, [PriorFactor("roi_m", ref_prior, tfd.LogNormal(np.float64(mu), np.float64(sigma)))],
                       reff=reff)
        w_psis = res.psis_weights if res.psis_weights is not None else res.raw_weights
        uni = np.full(S, 1.0 / S)
        sP, sA, sB = wsummary(flat, w_psis), wsummary(A.reshape(-1, J), uni), wsummary(B.reshape(-1, J), uni)
        seP = boot_se(flat, w_psis, n_ch, n_dr, rng)
        seA = boot_se(A.reshape(-1, J), uni, *A.shape[:2], rng)
        seB = boot_se(B.reshape(-1, J), uni, *B.shape[:2], rng)
        for st in STATS:
            for j, ch in enumerate(channels):
                rows.append(dict(
                    scenario=name, family=family, status=res.status, pareto_k=res.pareto_k, channel=ch, stat=st,
                    psis=sP[st][j], refit_A=sA[st][j], refit_B=sB[st][j],
                    e_psis_A=sP[st][j] - sA[st][j], e_psis_B=sP[st][j] - sB[st][j], e_AB=sA[st][j] - sB[st][j],
                    se_psis=seP[st][j], se_A=seA[st][j], se_B=seB[st][j],
                    z_AB=(sA[st][j] - sB[st][j]) / np.hypot(seA[st][j], seB[st][j]),
                    z_psis_A=(sP[st][j] - sA[st][j]) / np.hypot(seP[st][j], seA[st][j]),
                    z_psis_B=(sP[st][j] - sB[st][j]) / np.hypot(seP[st][j], seB[st][j]),
                    A_rhat=ma["roi_m"]["rhat_max"], B_rhat=mb["roi_m"]["rhat_max"],
                    A_div=int(sum(ma["divergences_per_chain"])), B_div=int(sum(mb["divergences_per_chain"]))))
        print(f"{name}: done (k={res.pareto_k:.2f}, {res.status})", flush=True)

    ch = pd.DataFrame(rows)
    ch.to_csv(os.path.join(a.out, "channel_level_noise.csv"), index=False)
    g = ch.groupby(["scenario", "family", "status", "pareto_k", "stat"], sort=False)
    sc = g.agg(mae_psis_A=("e_psis_A", lambda x: np.abs(x).mean()), mae_psis_B=("e_psis_B", lambda x: np.abs(x).mean()),
               mae_AB=("e_AB", lambda x: np.abs(x).mean()), max_abs_z_AB=("z_AB", lambda x: np.abs(x).max()),
               max_abs_z_psis_A=("z_psis_A", lambda x: np.abs(x).max()),
               A_div=("A_div", "first"), B_div=("B_div", "first"), A_rhat=("A_rhat", "first"),
               B_rhat=("B_rhat", "first")).reset_index()
    sc["ratio_psisA_over_AB"] = sc.mae_psis_A / sc.mae_AB
    sc.to_csv(os.path.join(a.out, "scenario_level_noise.csv"), index=False)

    # calibration of the bootstrap SEs, from the A-vs-B differences (sd about 1 if SEs are right)
    cal = ch.groupby("stat").z_AB.agg(["count", "mean", "std"]).round(3)
    cal["frac_abs_gt2"] = ch.groupby("stat").z_AB.apply(lambda z: (np.abs(z) > 2).mean()).round(3)
    cal.to_csv(os.path.join(a.out, "se_calibration_z_AB.csv"))
    print(cal.to_string())
    print("wrote", a.out)


if __name__ == "__main__":
    main()
