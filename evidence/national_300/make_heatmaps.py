"""Stage 5: national-model heatmaps. Reads evidence/national_300/results/metrics_target_level.csv (comparison == psis_vs_refit).
Fig A (national_error_heatmaps.png): rows = prior-change tier (quintiles of KL new prior || reference prior), columns = parameter,
  one panel per error metric of the reweighted posterior vs the full refit. Cell = median over 20 targets (mean for share/rank metrics).
Fig B (national_diagnostics_heatmap.png): per parameter, rows = quintiles of a diagnostic (k-hat; share of effective draws, best at top),
  columns = error metrics. Cell = median (mean for share/rank metrics) over the 20 targets in that quintile.
Fig C (national_diagnostics_by_tier.png): k-hat / effective share / prior TV by tier and parameter.
Darker = worse. Usage: python -I evidence/national_300/make_heatmaps.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results"); FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])
SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#52524f"
PARAMS = ["roi_m", "alpha_m", "ec_m"]; TIERS = ["very close", "close", "mid", "far", "very far"]

t = pd.read_csv(os.path.join(RES, "metrics_target_level.csv"))
t["miss"] = 1 - t.cov_contains; t["gap"] = 1 - t.cov_jaccard; t["rankdis"] = 1 - t.rank_spearman
p = t[t.comparison == "psis_vs_refit"].copy()
if np.issubdtype(p.tier.dtype, np.number):
    p["tier"] = p.tier.map(dict(enumerate(TIERS)))
# (column, bold title, grey subtitle, short column label, aggregation, number format)
METRICS = [
    ("kl_refit_psis", "KL( full refit ‖ reweighted posterior )", "lower = closer", "KL", "median", "{:.3f}"),
    ("w1", "Wasserstein-1( full refit, reweighted posterior )", "parameter units; lower = closer", "Wasserstein-1", "median", "{:.3f}"),
    ("ks", "KS distance( full refit, reweighted posterior )", "largest gap between the two CDFs; lower = closer", "KS", "median", "{:.3f}"),
    ("err_mean_sd", "Error of the posterior mean, in full-refit SDs", "|reweighted mean − full-refit mean| ÷ full-refit SD", "mean error\n(in SDs)", "median", "{:.3f}"),
    ("err_med_sd", "Error of the posterior median, in full-refit SDs", "|reweighted median − full-refit median| ÷ full-refit SD", "median\nerror", "median", "{:.3f}"),
    ("miss", "Full-refit mean outside the reweighted 90% interval", "share of cases", "mean out\n90% int.", "mean", "{:.2f}"),
    ("gap", "90% interval mismatch, full refit vs reweighted", "1 − interval overlap; 0 = identical", "90% int.\nmismatch", "median", "{:.3f}"),
    ("rankdis", "Channel-ranking disagreement, full refit vs reweighted", "1 − Spearman correlation; 0 = same ranking", "rank\ndisagr.", "mean", "{:.2f}"),
]


def style(ax, xt, yt, xl_fs=8, yl_fs=8, left=True):
    ax.set_xlim(-.5, len(xt) - .5); ax.set_ylim(len(yt) - .5, -.5)
    ax.set_xticks(range(len(xt))); ax.set_xticklabels(xt, fontsize=xl_fs, color=INK2)
    ax.set_yticks(range(len(yt))); ax.set_yticklabels(yt if left else [], fontsize=yl_fs, color=INK2)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)


def cells(ax, M, fmt, norm, fs=8.5):
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]; c = CMAP(norm(max(v, norm.vmin)))
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=c, edgecolor=SURFACE, linewidth=2))
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color="#fff" if c[0] < .45 else INK)


def titled(ax, title, sub, pad=30, off=17):
    ax.set_title(title, fontsize=8.8, color=INK, loc="left", pad=pad, fontweight="semibold")
    if sub:
        ax.annotate(sub, (0, 1), xytext=(0, off), textcoords="offset points", fontsize=7.8, color=INK2, va="bottom", ha="left",
                    xycoords="axes fraction")


def lognorm(M):
    pos = M[M > 0]; vmax = np.nanmax(M) * 1.001
    vmin = max(pos.min(), 1e-3) if len(pos) else 1e-3
    if vmax <= vmin * 1.5:                       # (nearly) constant panel: anchor so that zero is the lightest colour
        vmin = vmax / 30
    return LogNorm(vmin=vmin, vmax=vmax, clip=True)


# ---------- Fig A: error by tier x parameter ----------
fig, axes = plt.subplots(4, 2, figsize=(10.5, 3.2 * 4), facecolor=SURFACE)
H = 3.2 * 4; out = []
for k, ax in enumerate(axes.ravel()):
    ax.set_facecolor(SURFACE)
    if k >= len(METRICS):
        ax.axis("off"); continue
    col, ttl, sub, _, how, fmt = METRICS[k]
    M = p.groupby(["tier", "param"])[col].agg(how).unstack("param").reindex(index=TIERS, columns=PARAMS).values
    cells(ax, M, fmt, lognorm(M)); style(ax, PARAMS, TIERS, left=(k % 2 == 0))
    titled(ax, ttl, sub)
    for ti, tn in enumerate(TIERS):
        for pj, pn in enumerate(PARAMS):
            out.append(dict(metric=col, tier=tn, param=pn, value=M[ti, pj]))
fig.text(0.015, 1 - 0.1 / H, "National model: how far the PSIS-reweighted posterior is from a full refit", ha="left", va="top", fontsize=12,
         color=INK, fontweight="semibold")
fig.text(0.015, 1 - 0.5 / H, "Rows: prior-change tier. Columns: parameter. Cell: median of 20 priors (mean for the share panel).", ha="left", va="top", fontsize=8.8, color=INK2)
fig.subplots_adjust(left=0.1, right=0.97, top=1 - 1.5 / H, bottom=0.01, wspace=0.2, hspace=0.75)
fig.savefig(os.path.join(FIG, "national_error_heatmaps.png"), dpi=150, facecolor=SURFACE); plt.close(fig)
pd.DataFrame(out).to_csv(os.path.join(RES, "heatmap_data.csv"), index=False)

# ---------- Fig B: diagnostic quintiles x error metrics, per parameter ----------
DIAG = [("pareto_k", "Pareto k̂ of the importance weights", "k̂", True), ("ess_fraction", "Share of effective draws", "effective draws", False)]
rows = []
tabs = {}
KB = [-np.inf, 0.3, 0.5, 0.7, np.inf]; KLAB = ["k̂ < 0.3", "0.3 ≤ k̂ < 0.5", "0.5 ≤ k̂ < 0.7", "k̂ ≥ 0.7"]
for dcol, dname, short, asc in DIAG:
    for pn in PARAMS:
        g = p[p.param == pn].copy()
        if dcol == "pareto_k":
            g["q"] = pd.cut(g[dcol], KB, labels=False, right=False); nb = 4
        else:
            g["q"] = pd.qcut(g[dcol].rank(method="first", ascending=asc), 5, labels=False); nb = 5   # 0 = highest ESS
        M = np.array([[g[g.q == q][m[0]].agg(m[4]) if (g.q == q).any() else np.nan for m in METRICS] for q in range(nb)])
        lab = []
        for q in range(nb):
            n = int((g.q == q).sum())
            if dcol == "pareto_k":
                lab.append(f"{KLAB[q]}   (n = {n})")
            else:
                lo, hi = g[g.q == q][dcol].min(), g[g.q == q][dcol].max()
                lab.append(f"Q{q + 1}:  {100 * hi:.1f}% to {100 * lo:.1f}%   (n = {n})")
        tabs[(dcol, pn)] = (M, lab)
        for q in range(nb):
            for j, m in enumerate(METRICS):
                rows.append(dict(diagnostic=dcol, param=pn, bin=q + 1, range=lab[q], metric=m[0], value=M[q, j]))
pd.DataFrame(rows).to_csv(os.path.join(RES, "diagnostic_quintile_data.csv"), index=False)
allM = np.concatenate([tabs[k][0] for k in tabs])                                      # per-metric colour range, shared by all panels
norms = [lognorm(allM[:, j]) for j in range(len(METRICS))]
fig, axes = plt.subplots(2, 3, figsize=(21, 8.0), facecolor=SURFACE)
for r, (dcol, dname, short, asc) in enumerate(DIAG):
    for c, pn in enumerate(PARAMS):
        ax = axes[r, c]; ax.set_facecolor(SURFACE); M, lab = tabs[(dcol, pn)]
        for i in range(M.shape[0]):
            for j, m in enumerate(METRICS):
                v = M[i, j]
                if np.isnan(v):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f1f0ec", edgecolor=SURFACE, linewidth=2))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=8, color="#a3a29d"); continue
                col = CMAP(norms[j](max(v, norms[j].vmin)))
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=col, edgecolor=SURFACE, linewidth=2))
                ax.text(j, i, m[5].format(v), ha="center", va="center", fontsize=8, color="#fff" if col[0] < .45 else INK)
        style(ax, [m[3] for m in METRICS], lab, xl_fs=6.6, yl_fs=8)
        titled(ax, f"{pn}:  " + ("Pareto k̂ bins (0.3 / 0.5 / 0.7)" if dcol == "pareto_k" else "quintiles of effective-draw share"), ("n = priors in the bin" if dcol == "pareto_k" else "n = priors in the bin"), pad=50, off=34)
fig.text(0.012, 0.985, "National model: how well do k̂ and the effective-draw share predict the error of the reweighted posterior?", ha="left", va="top",
         fontsize=12, color=INK, fontweight="semibold")
fig.text(0.012, 0.953, "Rows: diagnostic bins. Cell: median error of the reweighted posterior vs the full refit (mean for the share panel). Darker = larger.", ha="left", va="top", fontsize=8.8, color=INK2)
fig.subplots_adjust(left=0.1, right=0.995, top=0.76, bottom=0.03, wspace=0.55, hspace=0.50)
fig.savefig(os.path.join(FIG, "national_diagnostics_heatmap.png"), dpi=150, facecolor=SURFACE); plt.close(fig)

# ---------- Fig C: diagnostics by tier ----------
fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), facecolor=SURFACE)
D = [("pareto_k", "Pareto k̂ of the importance weights", "median; lower = more reliable", "{:.2f}", LogNorm(0.05, 1.2, clip=True)),
     ("ess_fraction", "Effective share of reference draws after reweighting", "median; darker = fewer", "{:.3f}", None),
     ("TV_prior", "Prior change: total variation distance", "between new and reference prior (median)", "{:.2f}", LogNorm(0.05, 0.6, clip=True))]
for ax, (col, ttl, sub, fmt, norm) in zip(axes, D):
    ax.set_facecolor(SURFACE)
    M = p.groupby(["tier", "param"])[col].median().unstack("param").reindex(index=TIERS, columns=PARAMS).values
    if norm is None:
        M2 = 1 - M; nm = LogNorm(0.2, 1.0, clip=True)
        for i in range(5):
            for j in range(3):
                c = CMAP(nm(max(M2[i, j], 0.2))); ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=c, edgecolor=SURFACE, linewidth=2))
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=8.5, color="#fff" if c[0] < .45 else INK)
    else:
        cells(ax, M, fmt, norm)
    style(ax, PARAMS, TIERS, left=(ax is axes[0])); titled(ax, ttl, sub)
fig.text(0.015, 0.985, "National model: reliability diagnostics by size of prior change", ha="left", va="top", fontsize=12, color=INK, fontweight="semibold")
fig.text(0.015, 0.925, "Rows: prior-change tier. Columns: parameter. Cell: median of 20 priors.", ha="left", va="top", fontsize=8.8, color=INK2)
fig.subplots_adjust(left=0.08, right=0.99, top=0.66, bottom=0.04, wspace=0.25)
fig.savefig(os.path.join(FIG, "national_diagnostics_by_tier.png"), dpi=170, facecolor=SURFACE); plt.close(fig)
# ---------- Fig D: per parameter, per prior-change tier: diagnostic bins x error metrics ----------
EB = [-np.inf, 0.3, 0.5, 0.7, np.inf]
DIAG2 = [("pareto_k", KB, KLAB, False), ("ess_fraction", EB, ["effective draws 70–100%", "50–70%", "30–50%", "< 30%"], True)]
SHORT = ["KL", "Wasser-\nstein-1", "KS", "mean\nerror", "median\nerror", "mean out\n90% int.", "90% int.\nmismatch", "rank\ndisagr."]
tab, rows = {}, []
for dcol, edges, labs, flip in DIAG2:
    for pn in PARAMS:
        for tn in TIERS:
            g = p[(p.param == pn) & (p.tier == tn)].copy()
            b = pd.cut(g[dcol], edges, labels=False, right=False)
            b = 3 - b if flip else b
            n = np.array([(b == q).sum() for q in range(4)])
            M = np.array([[g[b == q][m[0]].agg(m[4]) if n[q] else np.nan for m in METRICS] for q in range(4)])
            tab[(dcol, pn, tn)] = (n, M, g.KL_prior.min(), g.KL_prior.max())
            for q in range(4):
                for j, m in enumerate(METRICS):
                    rows.append(dict(diagnostic=dcol, param=pn, tier=tn, bin=labs[q], n=n[q], metric=m[0], value=M[q, j]))
pd.DataFrame(rows).to_csv(os.path.join(RES, "diagnostic_tier_bin_data.csv"), index=False)
allv = np.stack([np.concatenate([tab[k][1][:, j] for k in tab]) for j in range(len(METRICS))])
norms2 = [lognorm(allv[j][~np.isnan(allv[j])]) for j in range(len(METRICS))]
for pn in PARAMS:
    H = 7.6
    fig, axes = plt.subplots(2, 5, figsize=(24, H), facecolor=SURFACE)
    for r, (dcol, edges, labs, flip) in enumerate(DIAG2):
        for c, tn in enumerate(TIERS):
            ax = axes[r, c]; ax.set_facecolor(SURFACE); n, M, klo, khi = tab[(dcol, pn, tn)]
            for i in range(4):
                ax.text(0, i, str(n[i]), ha="center", va="center", fontsize=8.5, color=INK2)
                for j, m in enumerate(METRICS):
                    v = M[i, j]
                    if np.isnan(v):
                        ax.add_patch(plt.Rectangle((j + .5, i - .5), 1, 1, facecolor="#f1f0ec", edgecolor=SURFACE, linewidth=2))
                        ax.text(j + 1, i, "–", ha="center", va="center", fontsize=8, color="#a3a29d"); continue
                    col = CMAP(norms2[j](max(v, norms2[j].vmin)))
                    ax.add_patch(plt.Rectangle((j + .5, i - .5), 1, 1, facecolor=col, edgecolor=SURFACE, linewidth=2))
                    ax.text(j + 1, i, m[5].format(v), ha="center", va="center", fontsize=7.6, color="#fff" if col[0] < .45 else INK)
            style(ax, ["n"] + SHORT, labs if dcol == "ess_fraction" else labs, xl_fs=6.4, yl_fs=8, left=(c == 0))
            ax.set_xlim(-.5, len(METRICS) + .5)
            if r == 0:
                titled(ax, f"{tn}  (prior KL {klo:.2f} to {khi:.2f})", "", pad=44)
    fig.text(0.012, 1 - 0.1 / H, f"National model, {pn}: error of the reweighted posterior by diagnostic bin, within each prior-change tier", ha="left", va="top",
             fontsize=12, color=INK, fontweight="semibold")
    fig.text(0.012, 1 - 0.5 / H, "Columns: prior-change tier. Rows: diagnostic bins; n = priors in the bin; – = none. Cell: median error vs the full refit (mean for the share column).", ha="left", va="top", fontsize=8.6, color=INK2)
    fig.subplots_adjust(left=0.085, right=0.995, top=1 - 1.75 / H, bottom=0.04, wspace=0.12, hspace=0.35)
    fig.savefig(os.path.join(FIG, f"national_diagnostics_tables_{pn}.png"), dpi=140, facecolor=SURFACE); plt.close(fig)

print("done")
