"""Three blog figures -> evidence/national_300/figures/blog/: (1) prior tiers, (2) error by tier (KL, mean error), (3) k-hat vs KL scatter.
Reads evidence/national_300/calibration/chosen_targets.csv, evidence/national_300/results/{metrics_target_level,heatmap_data}.csv. Usage: python -I evidence/national_300/make_blog_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); OUT = os.path.join(HERE, "figures", "blog")
os.makedirs(OUT, exist_ok=True)
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#52524f"
PARAMS = ["roi_m", "alpha_m", "ec_m"]; MT = ["very close", "close", "mid", "far", "very far"]
MERGE = dict(enumerate(MT))
TN = lambda loc, sc: stats.truncnorm((0.1 - loc) / sc, (10 - loc) / sc, loc=loc, scale=sc)


def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(labelsize=8, colors=INK2, length=2)


# ---------- 1. prior tiers ----------
XR = {"roi_m": (0.01, 8.0), "alpha_m": (0.0005, 0.9995), "ec_m": (0.1, 6.0)}
XL = {"roi_m": "roi_m", "alpha_m": "alpha_m", "ec_m": "ec_m"}
LINE = ["#9ec5f4", "#3987e5", "#0d366b"]


def dist(pn, r=None):
    if pn == "roi_m": return stats.lognorm(s=0.9, scale=np.exp(0.2)) if r is None else stats.lognorm(s=r.sigma, scale=np.exp(r.mu))
    if pn == "alpha_m": return stats.beta(1, 1) if r is None else stats.beta(r.a, r.b)
    return TN(0.8, 0.8) if r is None else TN(r["loc"], r["scale"])


d = pd.read_csv(os.path.join(HERE, "calibration", "chosen_targets.csv")); d["mt"] = d.tier.map(MERGE)
t = pd.read_csv(os.path.join(RES, "metrics_target_level.csv")); p = t[t.comparison == "psis_vs_refit"].copy(); p["mt"] = p.tier.map(MERGE)
fig, axes = plt.subplots(3, 5, figsize=(17.5, 9.4), facecolor=SURFACE)
for i, pn in enumerate(PARAMS):
    x = np.linspace(*XR[pn], 800); ref = dist(pn).pdf(x); tops = []
    for j, ti in enumerate(MT):
        ax = axes[i, j]; ax.set_facecolor(SURFACE)
        g = d[(d.param == pn) & (d.mt == ti)].sort_values("KL").reset_index(drop=True)
        ax.fill_between(x, ref, color="#d9d8d4", alpha=.8, lw=0); ax.plot(x, ref, color=INK, lw=2.2, label="reference prior")
        top = ref.max()
        for k, idx in enumerate([0, len(g) // 2, len(g) - 1]):
            r = g.iloc[idx] if pn != "ec_m" else g.iloc[idx]
            y = dist(pn, r).pdf(x); top = max(top, y.max())
            ax.plot(x, y, color=LINE[k], lw=1.8, label=f"new prior, KL = {r.KL:.2f}")
        tops.append(min(top * 1.08, 4 * ref.max()))
        ax.set_xlim(*XR[pn]); clean(ax)
        if i == 0: ax.set_title(f"{ti}", fontsize=11, color=INK, loc="left", fontweight="semibold", pad=8)
        ax.set_xlabel(XL[pn], fontsize=8.5, color=INK2)
        ax.legend(fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, labelcolor=INK2, handlelength=1.5)
        if j: ax.tick_params(axis="y", labelleft=False)
        else: ax.set_ylabel("prior density", fontsize=8.5, color=INK2)
    for ax in axes[i]: ax.set_ylim(0, max(tops))
fig.text(0.012, 0.985, "What close and far priors look like", ha="left", va="top", fontsize=13, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.07, right=0.99, top=0.93, bottom=0.10, wspace=0.12, hspace=0.62)
fig.savefig(os.path.join(OUT, "fig1_prior_tiers.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

# ---------- 2. error by tier ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), facecolor=SURFACE)
for ax, (m, ttl, sub, fmt) in zip(axes, [("kl_refit_psis", "KL( full refit ‖ reweighted posterior )", "lower = closer", "{:.3f}"),
                                         ("err_mean_sd", "Error of the posterior mean, in SDs", "|reweighted mean − refit mean| ÷ refit SD", "{:.3f}")]):
    ax.set_facecolor(SURFACE)
    M = p.groupby(["mt", "param"])[m].median().unstack("param").reindex(index=MT, columns=PARAMS).values
    norm = LogNorm(M.min(), M.max() * 1.001, clip=True)
    for i in range(5):
        for j in range(3):
            c = CMAP(norm(M[i, j])); ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=c, edgecolor=SURFACE, linewidth=2))
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=10, color="#fff" if c[0] < .45 else INK)
    ax.set_xlim(-.5, 2.5); ax.set_ylim(4.5, -.5); ax.set_xticks(range(3)); ax.set_xticklabels(PARAMS, fontsize=9, color=INK2)
    ax.set_yticks(range(5)); ax.set_yticklabels(MT if ax is axes[0] else [], fontsize=9, color=INK2)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(ttl, fontsize=10.5, color=INK, loc="left", fontweight="semibold", pad=34)
    ax.annotate(sub, (0, 1), xytext=(0, 20), textcoords="offset points", xycoords="axes fraction", fontsize=8.3, color=INK2, va="bottom")
fig.text(0.012, 0.985, "Error of the reweighted posterior, by size of prior change", ha="left", va="top", fontsize=13, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.1, right=0.99, top=0.66, bottom=0.03, wspace=0.12)
fig.savefig(os.path.join(OUT, "fig2_error_by_tier.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

# ---------- 3. k-hat vs error ----------
b = t[t.comparison == "split_chain_baseline"]; floor = (b.groupby("param").kl_refit_psis.median() * 0.5)
fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURFACE); ax.set_facecolor(SURFACE)
COL = {"roi_m": "#0072B2", "alpha_m": "#E69F00", "ec_m": "#009E73"}; MK = {"roi_m": "o", "alpha_m": "s", "ec_m": "^"}
ax.axvspan(0.7, 1.35, color="#f1f0ec", zorder=0)
ax.axhspan(floor.min(), floor.max(), color="#d9d8d4", alpha=.8, zorder=1, lw=0)
for pn in PARAMS:
    g = p[p.param == pn]
    ax.scatter(g.pareto_k.clip(upper=1.33), g.kl_refit_psis.clip(lower=1e-4), s=26, marker=MK[pn], color=COL[pn], alpha=.75, edgecolor=SURFACE, linewidth=.5, label=pn, zorder=3)
for v in (0.5, 0.7): ax.axvline(v, color=INK2, ls=(0, (4, 3)), lw=1, zorder=2)
ax.set_yscale("log"); ax.set_xlim(-0.5, 1.35); ax.set_ylim(1e-3, 3)
nb = int((p.pareto_k > 1.33).sum())
ax.text(1.33, 2.3, "k̂ ≥ 0.7: refit", ha="right", va="top", fontsize=9, color=INK2)
ax.text(-0.47, floor.max() * 1.15, "refit-to-refit noise", ha="left", va="bottom", fontsize=8.5, color=INK2)
ax.set_xlabel(f"Pareto k̂ of the importance weights  ({nb} priors with k̂ > 1.33 drawn at the right edge)", fontsize=10, color=INK2); ax.set_ylabel("KL(full refit ‖ reweighted posterior)", fontsize=10, color=INK2)
ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2); clean(ax)
fig.text(0.02, 0.985, "k̂ flags when the reweighted posterior can no longer be trusted", ha="left", va="top", fontsize=13, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.12)
fig.savefig(os.path.join(OUT, "fig3_khat_vs_error.png"), dpi=150, facecolor=SURFACE); plt.close(fig)
# ---------- 4. ESS vs error ----------
fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURFACE); ax.set_facecolor(SURFACE)
ax.axvspan(1e-2, 10, color="#f1f0ec", zorder=0)
ax.axhspan(floor.min(), floor.max(), color="#d9d8d4", alpha=.8, zorder=1, lw=0)
for pn in PARAMS:
    g = p[p.param == pn]
    ax.scatter((100 * g.ess_fraction).clip(lower=1e-2), g.kl_refit_psis.clip(lower=1e-4), s=26, marker=MK[pn], color=COL[pn], alpha=.75, edgecolor=SURFACE, linewidth=.5, label=pn, zorder=3)
ax.axvline(10, color=INK2, ls=(0, (4, 3)), lw=1, zorder=2)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-2, 100); ax.set_ylim(1e-3, 3)
ax.text(8, 2.3, "ESS < 10% of draws: check or refit", ha="right", va="top", fontsize=9, color=INK2)
ax.text(1.2e-2, floor.max() * 1.15, "refit-to-refit noise", ha="left", va="bottom", fontsize=8.5, color=INK2)
ax.set_xlabel("Effective sample size after reweighting (% of draws, log scale)", fontsize=10, color=INK2); ax.set_ylabel("KL(full refit ‖ reweighted posterior)", fontsize=10, color=INK2)
ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0.0, 0.1), labelcolor=INK2); clean(ax)
fig.text(0.02, 0.985, "A low effective sample size flags an unreliable reweighted posterior", ha="left", va="top", fontsize=13, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.12)
fig.savefig(os.path.join(OUT, "fig4_ess_vs_error.png"), dpi=150, facecolor=SURFACE); plt.close(fig)
print("done")
