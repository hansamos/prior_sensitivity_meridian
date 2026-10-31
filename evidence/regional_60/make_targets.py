"""Regional test, step 1: pick the 60 target priors (no model is fitted here). Rules are in evidence/regional_60/REGIONAL_PLAN.md.

roi_m : the 100 national roi_m targets (evidence/national_300/calibration/chosen_targets.csv), sorted by KL(target || reference prior) and cut
        into 3 rank terciles (close, mid, far); 10 per tercile at evenly spaced ranks -> the same priors as in the national test.
eta_m : HalfNormal(s) vs reference HalfNormal(1). KL(HN(s) || HN(1)) = -ln s + (s^2 - 1)/2. Pool: 400 log-spaced s in [0.2, 4]
        with KL >= 0.01; terciles of KL over the pool; 10 per tercile, 5 from each side of s = 1 (all 10 from one side if the
        other side has fewer than 5 pool values in that tercile), at evenly spaced KL ranks within each side.
Second refits (for the refit-to-refit noise floor): the 5th target by KL in each tier x parameter (6 in total).
Usage: python -I evidence/regional_60/make_targets.py
"""
import math, os
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
NAT = os.path.join(HERE, "..", "national_300", "calibration", "chosen_targets.csv")
OUT = os.path.join(HERE, "calibration", "regional_targets.csv")
TIERS = ["close", "mid", "far"]


def spaced(n_avail, k):
    return np.unique(np.round(np.linspace(0, n_avail - 1, k)).astype(int))


def roi_targets():
    t = pd.read_csv(NAT); t = t[t.param == "roi_m"].sort_values("KL").reset_index(drop=True)
    assert len(t) == 100
    t["tier"] = np.repeat([0, 1, 2], [33, 34, 33])
    rows = []
    for k in range(3):
        g = t[t.tier == k].reset_index(drop=True)
        pick = g.iloc[spaced(len(g), 10)]
        assert len(pick) == 10
        for r, (_, x) in enumerate(pick.iterrows(), 1):
            rows.append(dict(param="roi_m", tier=k, replicate=r, kind="LogNormal", p1=x.mu, p2=x.sigma, KL=x.KL, TV=x.TV,
                             source_target_id=x.target_id, type=x.type))
    return rows


def kl_hn(s):
    return -np.log(s) + (s ** 2 - 1) / 2


def tv_hn(s):
    x = np.linspace(0, 40, 400001)
    return 0.5 * np.sum(np.abs(stats.halfnorm.pdf(x, scale=s) - stats.halfnorm.pdf(x, scale=1.0))) * (x[1] - x[0])


def eta_targets():
    s = np.exp(np.linspace(math.log(0.2), math.log(4.0), 400))
    kl = kl_hn(s); m = kl >= 0.01
    s, kl = s[m], kl[m]
    edges = np.quantile(kl, [1 / 3, 2 / 3])
    tier = np.digitize(kl, edges)
    rows = []
    for k in range(3):
        sel = tier == k
        lo = np.where(sel & (s < 1))[0]; hi = np.where(sel & (s > 1))[0]
        lo = lo[np.argsort(kl[lo])]; hi = hi[np.argsort(kl[hi])]
        if len(lo) >= 5 and len(hi) >= 5:
            idx = list(lo[spaced(len(lo), 5)]) + list(hi[spaced(len(hi), 5)])
        else:
            side = lo if len(lo) > len(hi) else hi
            idx = list(side[spaced(len(side), 10)])
        assert len(idx) == 10, (k, len(lo), len(hi))
        idx = sorted(idx, key=lambda i: kl[i])
        for r, i in enumerate(idx, 1):
            rows.append(dict(param="eta_m", tier=k, replicate=r, kind="HalfNormal", p1=float(s[i]), p2=np.nan, KL=float(kl[i]),
                             TV=float(tv_hn(s[i])), source_target_id="", type="scale"))
    return rows


def main():
    d = pd.DataFrame(roi_targets() + eta_targets())
    d["tier_name"] = d.tier.map(dict(enumerate(TIERS)))
    d["target_id"] = d.param + "_R" + d.tier.astype(str) + "_r" + d.replicate.astype(str).str.zfill(2)
    d["second_refit"] = d.replicate == 5
    d = d.sort_values(["param", "tier", "replicate"]).reset_index(drop=True)
    d.insert(0, "index", np.arange(1, len(d) + 1))
    d["seed_A"] = 20000 + d["index"]; d["seed_B"] = 30000 + d["index"]
    os.makedirs(os.path.dirname(OUT), exist_ok=True); d.to_csv(OUT, index=False)
    print(len(d), "targets;", int(d.second_refit.sum()), "second refits")
    print(d.groupby(["param", "tier_name"]).KL.agg(["count", "min", "median", "max"]).round(3).to_string())
    print(d[d.param == "eta_m"][["target_id", "p1", "KL", "TV"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
