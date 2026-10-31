"""Stage 1: calibrate prior-distance tiers (quintiles of KL(target || reference prior)) and pick R targets per tier.

Reference priors (Meridian defaults / our explicit ROI prior): roi_m LogNormal(0.2, 0.9); alpha_m Uniform(0,1);
ec_m TruncatedNormal(0.8, 0.8; 0.1, 10). Perturbation families keep the target support inside the reference support:
  roi_m : LogNormal(mu', sigma'),  mu' = 0.2 + d*0.9 (d in [-2,2]),  sigma' = 0.9*f (f log-uniform in [0.25,4])
  ec_m  : TruncatedNormal(loc', scale'; 0.1, 10), loc' = 0.8 + d*0.8 (d in [-2,2]), scale' = 0.8*f
  alpha_m: Beta(a,b), mean m' in [0.05,0.95] (shift), concentration k' = 2*f (f log-uniform in [0.25,4]); Uniform = Beta(1,1), k=2, m=0.5
Type: 'shift' (d only), 'scale' (f only), 'both'. No model is fitted here.
Usage: python -I evidence/national_300/calibrate_perturbations.py [--n_candidates 3000] [--R 20]
"""
import argparse, math, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "calibration")
rng_seed = 20261031


def kl_tv_roi(mu, s):
    m0, s0 = 0.2, 0.9
    kl = math.log(s0 / s) + (s ** 2 + (mu - m0) ** 2) / (2 * s0 ** 2) - 0.5          # closed form, LogNormal||LogNormal
    z = np.linspace(-14, 14, 400001)                                                    # TV on the log scale
    p = stats.norm.pdf(z, m0, s0); q = stats.norm.pdf(z, mu, s)
    return kl, 0.5 * np.sum(np.abs(p - q)) * (z[1] - z[0])


def kl_tv_alpha(a, b):
    kl = -float(stats.beta(a, b).entropy())                                              # KL(Beta || Uniform(0,1)) = -H(Beta)
    n = 400000; x = (np.arange(n) + 0.5) / n
    return kl, 0.5 * np.sum(np.abs(stats.beta(a, b).pdf(x) - 1.0)) / n


def _tn(loc, sc):
    return stats.truncnorm((0.1 - loc) / sc, (10 - loc) / sc, loc=loc, scale=sc)


def kl_tv_ec(loc, sc):
    n = 400000; x = 0.1 + (np.arange(n) + 0.5) * (9.9 / n); dx = 9.9 / n
    p = _tn(0.8, 0.8).pdf(x); q = _tn(loc, sc).pdf(x)
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / np.maximum(p[m], 1e-300))) * dx), 0.5 * np.sum(np.abs(p - q)) * dx


def candidates(param, n, rng):
    rows = []
    types = rng.choice(["shift", "scale", "both"], size=n)
    for t in types:
        d = rng.uniform(-2, 2) if t in ("shift", "both") else 0.0
        f = math.exp(rng.uniform(math.log(0.25), math.log(4))) if t in ("scale", "both") else 1.0
        if param == "roi_m":
            h = dict(mu=0.2 + d * 0.9, sigma=0.9 * f); kl, tv = kl_tv_roi(h["mu"], h["sigma"])
        elif param == "ec_m":
            h = dict(loc=0.8 + d * 0.8, scale=0.8 * f); kl, tv = kl_tv_ec(h["loc"], h["scale"])
        else:
            mean = 0.5 + d * 0.225                                                       # d in [-2,2] -> mean in [0.05, 0.95]
            k = 2.0 * f; h = dict(a=mean * k, b=(1 - mean) * k, mean=mean, concentration=k)
            kl, tv = kl_tv_alpha(h["a"], h["b"])
        rows.append(dict(param=param, type=t, d_shift=d, f_scale=f, KL=kl, TV=tv, **h))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_candidates", type=int, default=3000)
    ap.add_argument("--R", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(rng_seed)
    allc, chosen, edges = [], [], []
    for param in ("roi_m", "alpha_m", "ec_m"):
        c = candidates(param, a.n_candidates, rng)
        q = np.quantile(c.KL, [0.2, 0.4, 0.6, 0.8])
        c["tier"] = np.digitize(c.KL, q)                                                 # 0..4
        names = ["very close", "close", "mid", "far", "very far"]
        c["tier_name"] = [names[i] for i in c.tier]
        edges.append(dict(param=param, KL_min=c.KL.min(), q20=q[0], q40=q[1], q60=q[2], q80=q[3], KL_max=c.KL.max(),
                          TV_median_by_tier=" / ".join(f"{c[c.tier == i].TV.median():.2f}" for i in range(5))))
        for i in range(5):
            pool = c[c.tier == i]
            # balance perturbation types within the tier as far as the pool allows, then fill randomly
            take = []
            per_type = a.R // 3
            for t in ("shift", "scale", "both"):
                sub = pool[pool.type == t]
                take += list(sub.sample(min(per_type, len(sub)), random_state=int(rng.integers(1e9))).index)
            rest = pool.drop(index=take)
            take += list(rest.sample(a.R - len(take), random_state=int(rng.integers(1e9))).index)
            sel = c.loc[take].copy(); sel["replicate"] = range(1, len(sel) + 1)
            chosen.append(sel)
        allc.append(c)
    allc = pd.concat(allc); chosen = pd.concat(chosen)
    chosen["target_id"] = chosen.param + "_T" + chosen.tier.astype(str) + "_r" + chosen.replicate.astype(str).str.zfill(2)
    allc.to_csv(os.path.join(OUT, "all_candidates.csv"), index=False)
    chosen.to_csv(os.path.join(OUT, "chosen_targets.csv"), index=False)
    e = pd.DataFrame(edges); e.to_csv(os.path.join(OUT, "tier_edges.csv"), index=False)
    pd.set_option("display.width", 200)
    print(e.round(3).to_string(index=False))
    print("\nchosen targets:", len(chosen))
    print(chosen.groupby(["param", "tier_name", "type"]).size().unstack(fill_value=0).to_string())
    print("\nKL range of chosen targets by tier:")
    print(chosen.groupby(["param", "tier"]).KL.agg(["min", "median", "max"]).round(3).to_string())


if __name__ == "__main__":
    main()
