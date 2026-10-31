"""Four regional summary figures -> evidence/regional_60/figures/: (1) prior tiers, (2) error by tier, (3) k-hat vs KL, (4) ESS vs KL.
Same style as evidence/national_300/make_blog_figures.py. Hollow markers = refit flagged by the pre-registered rule (R-hat > 1.01 or bulk ESS < 400 on the perturbed parameter).
Reads evidence/regional_60/calibration/regional_targets.csv and evidence/regional_60/results/metrics_target_level.csv. Usage: python -I evidence/regional_60/make_regional_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#52524f"
PARAMS = ["roi_m", "eta_m"]; TIERS = ["close", "mid", "far"]
COL = {"roi_m": "#0072B2", "eta_m": "#D55E00"}; MK = {"roi_m": "o", "eta_m": "s"}


def clean(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#bdbcb7")
    ax.tick_params(labelsize=8, colors=INK2, length=2)


d = pd.read_csv(os.path.join(HERE, "calibration", "regional_targets.csv"))
t = pd.read_csv(os.path.join(RES, "metrics_target_level.csv"))
p = t[t.comparison == "psis_vs_refit"].merge(d[["target_id", "tier_name"]], on="target_id")
p["flag"] = (p.refit_param_rhat > 1.01) | (p.refit_param_ess_bulk < 400)
b = t[t.comparison == "split_chain_baseline"]; floor = b.groupby("param").kl_refit_psis.median() * 0.5
nflag = int(p.flag.sum())

# ---------- 1. prior tiers ----------
XR = {"roi_m": (0.01, 8.0), "eta_m": (0.0, 4.0)}
LINE = ["#9ec5f4", "#3987e5", "#0d366b"]


def dist(pn, r=None):
    if pn == "roi_m": return stats.lognorm(s=0.9, scale=np.exp(0.2)) if r is None else stats.lognorm(s=r.p2, scale=np.exp(r.p1))
    return stats.halfnorm(scale=1.0) if r is None else stats.halfnorm(scale=r.p1)


fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.6), facecolor=SURFACE)
for i, pn in enumerate(PARAMS):
    x = np.linspace(*XR[pn], 800); ref = dist(pn).pdf(x); tops = []
    for j, ti in enumerate(TIERS):
        ax = axes[i, j]; ax.set_facecolor(SURFACE)
        g = d[(d.param == pn) & (d.tier_name == ti)].sort_values("KL").reset_index(drop=True)
        ax.fill_between(x, ref, color="#d9d8d4", alpha=.8, lw=0); ax.plot(x, ref, color=INK, lw=2.2, label="reference prior")
        top = ref.max()
        for k, idx in enumerate([0, len(g) // 2, len(g) - 1]):
            r = g.iloc[idx]; y = dist(pn, r).pdf(x); top = max(top, y.max())
            ax.plot(x, y, color=LINE[k], lw=1.8, label=f"new prior, KL = {r.KL:.2f}")
        tops.append(min(top * 1.08, 4 * ref.max()))
        ax.set_xlim(*XR[pn]); clean(ax)
        if i == 0: ax.set_title(ti, fontsize=11, color=INK, loc="left", fontweight="semibold", pad=8)
        ax.set_xlabel(pn, fontsize=8.5, color=INK2)
        ax.legend(fontsize=7.6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=2, labelcolor=INK2, handlelength=1.5)
        if j: ax.tick_params(axis="y", labelleft=False)
        else: ax.set_ylabel("prior density", fontsize=8.5, color=INK2)
    for ax in axes[i]: ax.set_ylim(0, max(tops))
fig.text(0.012, 0.985, "Regional model: what close, mid and far priors look like", ha="left", va="top", fontsize=13, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.16, wspace=0.12, hspace=0.75)
fig.savefig(os.path.join(OUT, "reg_fig1_prior_tiers.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

# ---------- 2. error by tier ----------
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), facecolor=SURFACE)
for ax, (m, ttl, sub, fmt) in zip(axes, [("kl_refit_psis", "KL( full refit ‖ reweighted posterior )", "lower = closer", "{:.3f}"),
                                         ("err_mean_sd", "Error of the posterior mean, in SDs", "|reweighted mean − refit mean| ÷ refit SD", "{:.3f}")]):
    ax.set_facecolor(SURFACE)
    M = p.groupby(["tier_name", "param"])[m].median().unstack("param").reindex(index=TIERS, columns=PARAMS).values
    norm = LogNorm(M.min(), M.max() * 1.001, clip=True)
    for i in range(3):
        for j in range(2):
            c = CMAP(norm(M[i, j])); ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=c, edgecolor=SURFACE, linewidth=2))
            ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=10, color="#fff" if c[0] < .45 else INK)
    ax.set_xlim(-.5, 1.5); ax.set_ylim(2.5, -.5); ax.set_xticks(range(2)); ax.set_xticklabels(PARAMS, fontsize=9, color=INK2)
    ax.set_yticks(range(3)); ax.set_yticklabels(TIERS if ax is axes[0] else [], fontsize=9, color=INK2)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(ttl, fontsize=10.5, color=INK, loc="left", fontweight="semibold", pad=34)
    ax.annotate(sub, (0, 1), xytext=(0, 20), textcoords="offset points", xycoords="axes fraction", fontsize=8.3, color=INK2, va="bottom")
fig.text(0.012, 0.985, "Regional model: error of the reweighted posterior, by size of prior change", ha="left", va="top", fontsize=12.5, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.13, right=0.99, top=0.64, bottom=0.03, wspace=0.12)
fig.savefig(os.path.join(OUT, "reg_fig2_error_by_tier.png"), dpi=150, facecolor=SURFACE); plt.close(fig)


# ---------- 3 & 4. diagnostics vs error ----------
def scatter(ax, xcol, xtf, xlim, xlabel, legend_kw=dict(loc="upper left")):
    ax.axhspan(floor.min(), floor.max(), color="#d9d8d4", alpha=.8, zorder=1, lw=0)
    for pn in PARAMS:
        for fl, lab in ((False, pn), (True, None)):
            g = p[(p.param == pn) & (p.flag == fl)]
            if g.empty: continue
            kw = dict(facecolor="none", edgecolor=COL[pn], linewidth=1.3) if fl else dict(color=COL[pn], edgecolor=SURFACE, linewidth=.5)
            ax.scatter(xtf(g[xcol]), g.kl_refit_psis.clip(lower=1e-4), s=34, marker=MK[pn], alpha=.8, label=lab, zorder=3, **kw)
    ax.set_yscale("log"); ax.set_ylim(1e-3, 3)
    ax.text(xlim[0] + 0.01 * (xlim[1] - xlim[0]) if ax.get_xscale() == "linear" else xlim[0] * 1.2, floor.max() * 1.15, "refit-to-refit noise", ha="left", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlabel(xlabel, fontsize=10, color=INK2); ax.set_ylabel("KL(full refit ‖ reweighted posterior)", fontsize=10, color=INK2); clean(ax)
    h, l = ax.get_legend_handles_labels()
    h.append(plt.Line2D([], [], marker="o", ls="", markerfacecolor="none", markeredgecolor=INK2, markersize=6)); l.append(f"refit flagged ({nflag})")
    ax.legend(h, l, frameon=False, fontsize=9, labelcolor=INK2, **legend_kw)


fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURFACE); ax.set_facecolor(SURFACE)
ax.axvspan(0.7, 1.35, color="#f1f0ec", zorder=0)
scatter(ax, "pareto_k", lambda s: s.clip(upper=1.33), (-0.5, 1.35), "")
for v in (0.5, 0.7): ax.axvline(v, color=INK2, ls=(0, (4, 3)), lw=1, zorder=2)
ax.set_xlim(-0.5, 1.35); nb = int((p.pareto_k > 1.33).sum())
ax.text(1.33, 2.3, "k̂ ≥ 0.7: refit", ha="right", va="top", fontsize=9, color=INK2)
ax.set_xlabel(f"Pareto k̂ of the importance weights  ({nb} priors with k̂ > 1.33 drawn at the right edge)", fontsize=10, color=INK2)
fig.text(0.02, 0.985, "Regional model: k̂ flags when the reweighted posterior can no longer be trusted", ha="left", va="top", fontsize=12.5, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.12)
fig.savefig(os.path.join(OUT, "reg_fig3_khat_vs_error.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURFACE); ax.set_facecolor(SURFACE)
ax.set_xscale("log"); ax.axvspan(1e-2, 10, color="#f1f0ec", zorder=0)
scatter(ax, "ess_fraction", lambda s: (100 * s).clip(lower=1e-2), (1e-2, 100), "Effective sample size after reweighting (% of draws, log scale)", dict(loc="lower left", bbox_to_anchor=(0.0, 0.1)))
ax.axvline(10, color=INK2, ls=(0, (4, 3)), lw=1, zorder=2); ax.set_xlim(1e-2, 100)
ax.text(8, 2.3, "ESS < 10% of draws: check or refit", ha="right", va="top", fontsize=9, color=INK2)
fig.text(0.02, 0.985, "Regional model: a low effective sample size flags an unreliable reweighted posterior", ha="left", va="top", fontsize=12.5, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.12)
fig.savefig(os.path.join(OUT, "reg_fig4_ess_vs_error.png"), dpi=150, facecolor=SURFACE); plt.close(fig)
print("done", nflag, "flagged; noise band", floor.round(4).to_dict())
