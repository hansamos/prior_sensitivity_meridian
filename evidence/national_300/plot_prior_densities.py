"""Prior densities per parameter (rows) and prior-change tier (columns): the reference prior and 4 perturbed priors of that tier,
picked at the lowest, ~1/3, ~2/3 and highest KL(new prior || reference prior) among the tier's 20 targets.
Reads evidence/national_300/calibration/chosen_targets.csv. Usage: python -I evidence/national_300/plot_prior_densities.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figures"); RES = os.path.join(HERE, "results")
SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#52524f"
LINES = ["#9ec5f4", "#3987e5", "#184f95", "#0d366b"]          # blue ramp: darker = larger KL within the tier
PARAMS = ["roi_m", "alpha_m", "ec_m"]; TIERS = ["very close", "close", "mid", "far", "very far"]
XR = {"roi_m": (0.01, 8.0), "alpha_m": (0.0005, 0.9995), "ec_m": (0.1, 6.0)}
XL = {"roi_m": "roi_m (ROI of each channel)", "alpha_m": "alpha_m (adstock decay)", "ec_m": "ec_m (half-saturation point)"}
TN = lambda loc, sc: stats.truncnorm((0.1 - loc) / sc, (10 - loc) / sc, loc=loc, scale=sc)


def dist(param, r=None):
    if param == "roi_m":
        return stats.lognorm(s=0.9, scale=np.exp(0.2)) if r is None else stats.lognorm(s=r.sigma, scale=np.exp(r.mu))
    if param == "alpha_m":
        return stats.beta(1, 1) if r is None else stats.beta(r.a, r.b)
    return TN(0.8, 0.8) if r is None else TN(r["loc"], r["scale"])


d = pd.read_csv(os.path.join(HERE, "calibration", "chosen_targets.csv"))
fig, axes = plt.subplots(3, 5, figsize=(19, 11.2), facecolor=SURFACE)
used = []
for i, pn in enumerate(PARAMS):
    x = np.linspace(*XR[pn], 800); ref = dist(pn).pdf(x); tops = []
    for j, tn in enumerate(TIERS):
        ax = axes[i, j]; ax.set_facecolor(SURFACE)
        g = d[(d.param == pn) & (d.tier == j)].sort_values("KL").reset_index(drop=True)
        ax.fill_between(x, ref, color="#d9d8d4", alpha=0.8, linewidth=0, zorder=1)
        ax.plot(x, ref, color=INK, lw=2.0, zorder=3, label="reference prior")
        top = ref.max()
        for k, idx in enumerate(np.linspace(0, len(g) - 1, 4).round().astype(int)):
            r = g.iloc[idx]; y = dist(pn, r).pdf(x); top = max(top, y.max())
            ax.plot(x, y, color=LINES[k], lw=1.7, zorder=2, label=f"KL {r.KL:.2f}  ({r.type})")
            used.append(dict(param=pn, tier=tn, target_id=r.target_id, KL=r.KL, TV=r.TV, type=r.type))
        cap = 4 * ref.max()
        ax.set_xlim(*XR[pn]); clipped = top > cap; tops.append(min(top * 1.08, cap))
        ax.set_title(f"{tn}   (KL {g.KL.min():.2f} to {g.KL.max():.2f})" if i == 0 else "", fontsize=9.5, color=INK, loc="left", fontweight="semibold", pad=8)
        if clipped:
            ax.text(0.99, 0.99, "peak clipped", transform=ax.transAxes, ha="right", va="top", fontsize=7, color=INK2)
        ax.legend(fontsize=7.2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, labelcolor=INK2, handlelength=1.6,
                  columnspacing=1.2)
        ax.tick_params(labelsize=7.5, colors=INK2, length=2); ax.tick_params(axis="y", labelleft=(j == 0))
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb7")
        ax.set_xlabel(XL[pn] if i == 2 or True else "", fontsize=7.8, color=INK2)
        if j == 0: ax.set_ylabel("prior density", fontsize=8, color=INK2)
    for ax in axes[i]:
        ax.set_ylim(0, max(tops))                      # same y-scale across the tiers of a row
H = 11.2
fig.text(0.012, 1 - 0.1 / H, "National model: reference prior and perturbed priors, by prior-change tier", ha="left", va="top", fontsize=12.5, color=INK,
         fontweight="semibold")
fig.text(0.012, 1 - 0.5 / H, "Black: reference prior. Blue: four of the 20 perturbed priors per tier (lowest to highest KL; darker = larger KL).", ha="left", va="top", fontsize=8.6, color=INK2)
fig.subplots_adjust(left=0.045, right=0.995, top=1 - 1.25 / H, bottom=0.10, wspace=0.12, hspace=0.62)
fig.savefig(os.path.join(FIG, "national_prior_densities.png"), dpi=140, facecolor=SURFACE)
pd.DataFrame(used).to_csv(os.path.join(RES, "prior_density_examples.csv"), index=False)
print("done", len(used))
