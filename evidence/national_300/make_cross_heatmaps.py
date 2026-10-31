"""k-hat bins x effective-draw bins (rows x columns), pooled over all 300 priors and per parameter (100 priors each, all tiers pooled).
One figure per error metric in evidence/national_300/figures/khat_ess_cross/; cell = median error (mean for share/rank metrics), text also shows n.
Reads evidence/national_300/results/metrics_target_level.csv (comparison == psis_vs_refit). Usage: python -I evidence/national_300/make_cross_heatmaps.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures", "khat_ess_cross")
os.makedirs(FIG, exist_ok=True)
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#52524f"
PARAMS = ["roi_m", "alpha_m", "ec_m"]
K_LAB = ["k̂ < 0.3", "0.3 ≤ k̂ < 0.5", "0.5 ≤ k̂ < 0.7", "k̂ ≥ 0.7"]; E_LAB = ["< 30%", "30–50%", "50–70%", "70–100%"]
METRICS = [("kl_refit_psis", "KL( full refit ‖ reweighted posterior )", "median", "{:.3f}"),
           ("w1", "Wasserstein-1( full refit, reweighted posterior )", "median", "{:.3f}"),
           ("ks", "KS distance( full refit, reweighted posterior )", "median", "{:.3f}"),
           ("err_mean_sd", "Error of the posterior mean, in full-refit SDs", "median", "{:.3f}"),
           ("err_med_sd", "Error of the posterior median, in full-refit SDs", "median", "{:.3f}"),
           ("miss", "Share of cases: full-refit mean outside the reweighted 90% interval", "mean", "{:.2f}"),
           ("gap", "90% interval mismatch, full refit vs reweighted", "median", "{:.3f}"),
           ("rankdis", "Channel-ranking disagreement, full refit vs reweighted", "mean", "{:.2f}")]

t = pd.read_csv(os.path.join(RES, "metrics_target_level.csv"))
t["miss"] = 1 - t.cov_contains; t["gap"] = 1 - t.cov_jaccard; t["rankdis"] = 1 - t.rank_spearman
p = t[t.comparison == "psis_vs_refit"].copy()
p["kb"] = pd.cut(p.pareto_k, [-np.inf, .3, .5, .7, np.inf], labels=False, right=False)
p["eb"] = pd.cut(p.ess_fraction, [-np.inf, .3, .5, .7, np.inf], labels=False, right=False)
panels = [("All parameters pooled", p)] + [(pn, p[p.param == pn]) for pn in PARAMS]
rows = []
for col, ttl, how, fmt in METRICS:
    grids = []
    for name, g in panels:
        M = np.full((4, 4), np.nan); N = np.zeros((4, 4), int)
        for i in range(4):
            for j in range(4):
                c = g[(g.kb == i) & (g.eb == j)]; N[i, j] = len(c)
                if len(c): M[i, j] = c[col].agg(how)
                rows.append(dict(metric=col, panel=name, k_bin=K_LAB[i], ess_bin=E_LAB[j], n=len(c), value=M[i, j]))
        grids.append((name, M, N))
    allv = np.concatenate([m[~np.isnan(m)] for _, m, _ in grids]); pos = allv[allv > 0]
    vmax = allv.max() * 1.001; vmin = max(pos.min(), 1e-3) if len(pos) else 1e-3
    if vmax <= vmin * 1.5: vmin = vmax / 30
    norm = LogNorm(vmin, vmax, clip=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.2), facecolor=SURFACE)
    for ax, (name, M, N) in zip(axes.ravel(), grids):
        ax.set_facecolor(SURFACE)
        for i in range(4):
            for j in range(4):
                if np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f1f0ec", edgecolor=SURFACE, linewidth=2))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="#a3a29d"); continue
                c = CMAP(norm(max(M[i, j], vmin))); tc = "#fff" if c[0] < .45 else INK
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=c, edgecolor=SURFACE, linewidth=2))
                ax.text(j, i - 0.1, fmt.format(M[i, j]), ha="center", va="center", fontsize=11, fontweight="semibold", color=tc)
                ax.text(j, i + 0.26, f"n = {N[i, j]}", ha="center", va="center", fontsize=8, color=tc)
        ax.set_xlim(-.5, 3.5); ax.set_ylim(3.5, -.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(E_LAB, fontsize=8.5, color=INK2)
        ax.set_yticks(range(4)); ax.set_yticklabels(K_LAB, fontsize=8.5, color=INK2)
        ax.xaxis.tick_top(); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        ax.set_title(f"{name}  ({N.sum()} priors)", fontsize=10, color=INK, loc="left", fontweight="semibold", pad=40)
        ax.annotate("share of effective draws", (0.5, 1), xytext=(0, 17), textcoords="offset points", xycoords="axes fraction",
                    ha="center", va="bottom", fontsize=8, color=INK2)
    fig.text(0.02, 0.985, ttl, ha="left", va="top", fontsize=12, color=INK, fontweight="semibold")
    fig.text(0.02, 0.953, "Rows: Pareto k̂ bin. Columns: share of effective draws. Cell: " + ("mean" if how == "mean" else "median") + " over the priors in the cell; darker = larger.",
             ha="left", va="top", fontsize=8.8, color=INK2)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.84, bottom=0.03, wspace=0.42, hspace=0.38)
    fig.savefig(os.path.join(FIG, f"{col}.png"), dpi=140, facecolor=SURFACE); plt.close(fig)
pd.DataFrame(rows).to_csv(os.path.join(RES, "khat_ess_cross_data.csv"), index=False)
print("done")

# ---------- KL by prior-change tier: one k-hat x effective-draw grid per parameter (rows) and tier (columns) ----------
TIERS = ["very close", "close", "mid", "far", "very far"]
if np.issubdtype(p.tier.dtype, np.number):
    p["tier"] = p.tier.map(dict(enumerate(TIERS)))
col, ttl, how, fmt = METRICS[0]
tabs, rows = {}, []
for pn in PARAMS:
    for tn in TIERS:
        g = p[(p.param == pn) & (p.tier == tn)]
        M = np.full((4, 4), np.nan); N = np.zeros((4, 4), int)
        for i in range(4):
            for j in range(4):
                c = g[(g.kb == i) & (g.eb == j)]; N[i, j] = len(c)
                if len(c): M[i, j] = c[col].agg(how)
                rows.append(dict(param=pn, tier=tn, k_bin=K_LAB[i], ess_bin=E_LAB[j], n=len(c), kl=M[i, j]))
        tabs[(pn, tn)] = (M, N, g.KL_prior.min(), g.KL_prior.max())
pd.DataFrame(rows).to_csv(os.path.join(RES, "khat_ess_cross_tier_data.csv"), index=False)
allv = np.concatenate([m[~np.isnan(m)] for m, _, _, _ in tabs.values()])
norm = LogNorm(max(allv[allv > 0].min(), 1e-3), allv.max() * 1.001, clip=True)
H = 13.5
fig, axes = plt.subplots(3, 5, figsize=(24, H), facecolor=SURFACE)
for r, pn in enumerate(PARAMS):
    for c, tn in enumerate(TIERS):
        ax = axes[r, c]; ax.set_facecolor(SURFACE); M, N, klo, khi = tabs[(pn, tn)]
        for i in range(4):
            for j in range(4):
                if np.isnan(M[i, j]):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f1f0ec", edgecolor=SURFACE, linewidth=2))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="#a3a29d"); continue
                cc = CMAP(norm(max(M[i, j], norm.vmin))); tc = "#fff" if cc[0] < .45 else INK
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=cc, edgecolor=SURFACE, linewidth=2))
                ax.text(j, i - 0.1, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=9, fontweight="semibold", color=tc)
                ax.text(j, i + 0.27, f"n = {N[i, j]}", ha="center", va="center", fontsize=7, color=tc)
        ax.set_xlim(-.5, 3.5); ax.set_ylim(3.5, -.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(E_LAB, fontsize=7.5, color=INK2)
        ax.set_yticks(range(4)); ax.set_yticklabels(K_LAB if c == 0 else [], fontsize=8, color=INK2)
        ax.xaxis.tick_top(); ax.tick_params(length=0)
        for s in ax.spines.values(): s.set_visible(False)
        if r == 0:
            ax.set_title(f"{tn}\n(prior KL {klo:.2f} to {khi:.2f})", fontsize=9.5, color=INK, loc="left", fontweight="semibold", pad=24)
        else:
            ax.set_title(f"(prior KL {klo:.2f} to {khi:.2f})", fontsize=8.5, color=INK2, loc="left", pad=22)
        if c == 0:
            ax.set_ylabel(pn, fontsize=11, color=INK, fontweight="semibold", labelpad=10)
fig.text(0.012, 1 - 0.1 / H, "KL( full refit ‖ reweighted posterior ), by Pareto k̂ bin and share of effective draws, within each prior-change tier", ha="left", va="top",
         fontsize=12, color=INK, fontweight="semibold")
fig.text(0.012, 1 - 0.5 / H, "Rows of panels: parameter. Columns: prior-change tier. Inside: k̂ bin (rows) × share of effective draws (columns); cell = median KL, n = priors; – = none.",
         ha="left", va="top", fontsize=8.8, color=INK2)
fig.subplots_adjust(left=0.07, right=0.995, top=1 - 1.45 / H, bottom=0.02, wspace=0.14, hspace=0.42)
fig.savefig(os.path.join(FIG, "kl_by_tier.png"), dpi=130, facecolor=SURFACE); plt.close(fig)
print("done tier grids")
