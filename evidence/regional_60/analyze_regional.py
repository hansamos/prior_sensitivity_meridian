"""Regional test, step 3: PSIS from the long regional reference vs each target refit (same metrics as evidence/national_300/analyze_targets.py).

Perturbed parameters: roi_m (LogNormal) and eta_m (HalfNormal), both with 5 channel values. Baselines: split-chain (refit chains {0,1} vs {2,3})
and A vs B refits where a second refit exists. Usage:
  python -I evidence/regional_60/analyze_regional.py --ref runs/regional/<long ref> --refits_a runs/regional_targets_A --refits_b runs/regional_targets_B --out evidence/regional_60/results
"""
import argparse, glob, json, os, sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("MERIDIAN_BACKEND", "jax")
HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("common", "regional_other_params"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
from scipy.ndimage import gaussian_filter1d

import meridian.backend  # noqa: F401  (JAX x64)
from prior_reweight import PriorFactor, reweight, relative_efficiency
from scenarios_other import REFERENCE_PRIORS, make

REF_PRIORS = {"roi_m": ("LogNormal", (0.2, 0.9)), "eta_m": ("HalfNormal", (1.0,))}


def target_spec(r):
    if r.param == "roi_m":
        return ("LogNormal", (r.p1, r.p2))
    return ("HalfNormal", (r.p1,))


def wquant(x, w, q):
    o = np.argsort(x); xs, ws = x[o], w[o]
    cdf = (np.cumsum(ws) - 0.5 * ws) / ws.sum()
    return np.interp(q, cdf, xs)


def metrics(x, wx, y, wy):
    """x with weights wx (normalised) vs y with weights wy (normalised); returns dict."""
    mx, my = float(np.sum(wx * x)), float(np.sum(wy * y))
    sdy = float(np.sqrt(np.sum(wy * (y - my) ** 2)))
    qx = [wquant(x, wx, q) for q in (0.05, 0.5, 0.95)]; qy = [wquant(y, wy, q) for q in (0.05, 0.5, 0.95)]
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    edges = np.linspace(lo, hi, 201)
    px = np.histogram(x, edges, weights=wx)[0]; py = np.histogram(y, edges, weights=wy)[0]
    px = gaussian_filter1d(px, 2); py = gaussian_filter1d(py, 2)
    px = np.maximum(px / px.sum(), 1e-8); py = np.maximum(py / py.sum(), 1e-8)
    kl = float(np.sum(py * np.log(py / px)))
    grid = np.sort(np.concatenate([x, y]))
    def ecdf(v, w):
        o = np.argsort(v); cw = np.cumsum(w[o]) / w.sum()
        return np.interp(grid, v[o], cw, left=0.0, right=1.0)
    ks = float(np.max(np.abs(ecdf(x, wx) - ecdf(y, wy))))
    inter = max(0.0, min(qx[2], qy[2]) - max(qx[0], qy[0])); union = max(qx[2], qy[2]) - min(qx[0], qy[0])
    return dict(kl_refit_psis=kl, w1=float(stats.wasserstein_distance(x, y, wx, wy)), ks=ks,
                err_mean_sd=abs(mx - my) / sdy, err_med_sd=abs(qx[1] - qy[1]) / sdy,
                cov_contains=float(qx[0] <= my <= qx[2]), cov_jaccard=inter / union if union > 0 else 0.0,
                width_ratio=(qx[2] - qx[0]) / (qy[2] - qy[0]))


def ranking(rx, wx, ry, wy):
    mx = (wx[:, None] * rx).sum(0); my = (wy[:, None] * ry).sum(0)
    return dict(rank_spearman=float(stats.spearmanr(mx, my)[0]), rank_same_top=float(np.argmax(mx) == np.argmax(my)),
                rank_same_order=float(np.array_equal(np.argsort(mx), np.argsort(my))))


def load(d):
    ds = xr.open_dataset(os.path.join(d, "posterior_media.nc"))
    return {v: np.asarray(ds[v]) for v in ("roi_m", "eta_m", "alpha_m", "ec_m")}


def pair_rows(tag, tid, param, tier, ra, rb, wa, wb, extra):
    """metrics for sample a (weights wa) vs sample b (weights wb), per channel + ranking"""
    out = []
    rk = ranking(ra["roi_m"], wa, rb["roi_m"], wb)
    for j in range(5):
        m = metrics(ra[param][:, j], wa, rb[param][:, j], wb)
        out.append(dict(comparison=tag, target_id=tid, param=param, tier=tier, channel=j, **m, **rk, **extra))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True); ap.add_argument("--refits_a", required=True)
    ap.add_argument("--refits_b", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--targets", default=os.path.join(HERE, "calibration", "regional_targets.csv"))
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    T = pd.read_csv(a.targets)
    refd = {v: np.asarray(xr.open_dataset(os.path.join(a.ref, "posterior.nc"))[v]) for v in ("roi_m", "eta_m", "alpha_m", "ec_m")}
    rows = []
    for r in T.itertuples(index=False):
        da = glob.glob(os.path.join(a.refits_a, f"{r.target_id}_seed*"))
        if len(da) != 1 or not os.path.exists(os.path.join(da[0], "run_metadata.json")):
            print("missing refit A:", r.target_id); continue
        A = load(da[0]); meta = json.load(open(os.path.join(da[0], "run_metadata.json")))
        arr = refd[r.param]; flat = arr.reshape(-1, 5)
        res = reweight({r.param: flat}, [PriorFactor(r.param, make(REF_PRIORS[r.param]), make(target_spec(r)))],
                       reff=relative_efficiency([arr]))
        w = res.psis_weights if res.psis_weights is not None else res.raw_weights
        w = np.asarray(w) / np.sum(w)
        refflat = {v: x.reshape(-1, 5) for v, x in refd.items()}
        Aflat = {v: x.reshape(-1, 5) for v, x in A.items()}
        uA = np.full(len(Aflat["roi_m"]), 1.0 / len(Aflat["roi_m"]))
        ex = dict(pareto_k=res.pareto_k, status=res.status, ess_fraction=res.ess_fraction, raw_ess=res.raw_ess,
                  psis_ess=res.psis_ess, KL_prior=r.KL, TV_prior=r.TV, ptype=r.type, tier_name=r.tier_name,
                  refit_param_rhat=meta["key_diagnostics"][r.param]["rhat_max"],
                  refit_param_ess_bulk=meta["key_diagnostics"][r.param]["ess_bulk_min"], refit_div=meta["divergences"], refit_allvar_rhat=meta["all_variables"]["rhat_max"], refit_seconds=meta["fit_seconds"])
        rows += pair_rows("psis_vs_refit", r.target_id, r.param, r.tier, refflat, Aflat, w, uA, ex)
        # split-chain baseline: chains {0,1} vs {2,3} of refit A
        h1 = {v: x[:2].reshape(-1, 5) for v, x in A.items()}; h2 = {v: x[2:].reshape(-1, 5) for v, x in A.items()}
        u1 = np.full(len(h1["roi_m"]), 1.0 / len(h1["roi_m"]))
        rows += pair_rows("split_chain_baseline", r.target_id, r.param, r.tier, h1, h2, u1, u1, ex)
        if a.refits_b:
            db = glob.glob(os.path.join(a.refits_b, f"{r.target_id}_seed*"))
            if len(db) == 1 and os.path.exists(os.path.join(db[0], "run_metadata.json")):
                Bf = {v: x.reshape(-1, 5) for v, x in load(db[0]).items()}
                rows += pair_rows("refit_A_vs_B", r.target_id, r.param, r.tier, Aflat, Bf, uA, uA, ex)
                rows += pair_rows("psis_vs_refit_B", r.target_id, r.param, r.tier, refflat, Bf, w, uA, ex)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "metrics_channel_level.csv"), index=False)
    num = [c for c in df.columns if df[c].dtype.kind == "f" and c != "channel"]
    t = df.groupby(["comparison", "target_id", "param", "tier"], as_index=False)[num].mean()
    t.to_csv(os.path.join(a.out, "metrics_target_level.csv"), index=False)
    print("rows:", len(df), "targets:", t.target_id.nunique())
    print(t.groupby(["comparison", "param", "tier"])[["kl_refit_psis", "w1", "ks", "err_mean_sd"]].median().round(3).head(60).to_string())


if __name__ == "__main__":
    main()
