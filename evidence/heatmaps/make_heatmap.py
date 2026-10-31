"""Heatmap of k-hat, PSIS effective draws (%), and PSIS-vs-refit error per statistic (mean, q05, q50, q95).

Rows: one per scenario x parameter (joint changes give one row per changed field), sorted by k-hat.
Error cell = mean over channels of |PSIS - refit A| / combined Monte Carlo SE (bootstrap), i.e. mean |z|;
pure Monte Carlo noise gives about 0.8. Darker = worse in every panel. Cell text is the actual value.
Reads saved CSVs only. Usage: python -I evidence/heatmaps/make_heatmap.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# reference blue ramp (dataviz palette.md sequential, steps 100 -> 700) and light-mode ink / surface
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#1a1a19", "#52524f", "#d9d8d4"

roi = pd.read_csv(os.path.join(ROOT, "regional_roi/results/channel_level_noise.csv"))
ess = pd.read_csv(os.path.join(ROOT, "regional_roi/results/scenario_level.csv"))[["scenario", "ess_fraction"]]
roi = roi.merge(ess, on="scenario")
roi["field"] = "roi_m"
cols = ["scenario", "field", "pareto_k", "ess_fraction", "stat", "z_psis_A"]
d = pd.concat([roi[cols],
               pd.read_csv(os.path.join(ROOT, "regional_other_params/results/channel_level.csv"))[cols],
               pd.read_csv(os.path.join(ROOT, "regional_other_params/results_multi/channel_level.csv"))[cols]], ignore_index=True)


def label(sc, field):
    if sc[0] == "J":
        return f"{sc[:2]} joint ({field})"
    if sc[0] == "C" and sc[1].isdigit():
        return f"{sc[:2]} channel-specific"
    return sc if field == "roi_m" else f"{sc}"


rows = []
for (sc, field), g in d.groupby(["scenario", "field"], sort=False):
    r = dict(row=label(sc, field), field=field, k=g.pareto_k.iloc[0], ess_pct=100 * g.ess_fraction.iloc[0])
    for st in ("mean", "q05", "q50", "q95"):
        r[st] = g[g.stat == st].z_psis_A.abs().mean()
    rows.append(r)
t = pd.DataFrame(rows).sort_values(["k", "row"]).reset_index(drop=True)
t.to_csv(os.path.join(OUT, "khat_ess_heatmap_data.csv"), index=False)

n = len(t)
fig = plt.figure(figsize=(11.5, 0.27 * n + 3.2), facecolor=SURFACE)
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 4], wspace=0.06,
                      left=0.255, right=0.925, top=1 - 1.15 / (0.27 * n + 3.2), bottom=1.1 / (0.27 * n + 3.2))
panels = [
    ("k̂", t[["k"]].values, Normalize(-0.5, 1.5), "{:.2f}", [""], "clipped at 1.5"),
    ("Effective draws", t[["ess_pct"]].values, Normalize(0, 100), "{:.0f}%", [""],
     "darker = fewer"),
    ("Error vs refit A in Monte Carlo SEs (mean |z| over channels)", t[["mean", "q05", "q50", "q95"]].values,
     Normalize(0.5, 4.0), "{:.1f}", ["mean", "5%", "50%", "95%"], "mean |z|, colour clipped at 4"),
]


def text_color(rgba):
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return "#ffffff" if lum < 0.45 else INK


for p, (title, vals, norm, fmt, xl, cap) in enumerate(panels):
    ax = fig.add_subplot(gs[0, p], facecolor=SURFACE)
    shown = 100 - vals if p == 1 else vals          # ESS: fewer draws = darker
    nn = Normalize(0, 100) if p == 1 else norm
    ax.imshow(shown, cmap=CMAP, norm=nn, aspect="auto", interpolation="nearest")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            c = CMAP(nn(shown[i, j]))
            ax.text(j, i, fmt.format(vals[i, j]), ha="center", va="center", fontsize=7.5, color=text_color(c))
    ax.set_xticks(range(len(xl)))
    ax.set_xticklabels(xl, fontsize=8.5, color=INK2)
    ax.xaxis.tick_top()
    ax.set_title(title, fontsize=9.5, color=INK, pad=22 if p == 2 else 22, loc="center", fontweight="semibold")
    ax.set_yticks(range(n) if p == 0 else [])
    if p == 0:
        ax.set_yticklabels([r.row if (r.field == "roi_m" or r.row[0] == "J") else f"{r.row}  [{r.field}]" for r in t.itertuples()],
                           fontsize=7.5, color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    # 2px surface gaps between cells
    ax.set_xticks(np.arange(-0.5, vals.shape[1]), minor=True)
    ax.set_yticks(np.arange(-0.5, n), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    # k-hat thresholds as horizontal rules across all panels
    for thr in (0.3, 0.5, 0.7):
        pos = np.searchsorted(t.k.values, thr)       # first row with k >= thr
        if 0 < pos < n:
            ax.axhline(pos - 0.5, color=INK, linewidth=0.9, linestyle=(0, (4, 3)))
            if p == 2:
                ax.text(vals.shape[1] - 0.45, pos - 0.5, f"k̂ ≥ {thr}", ha="left", va="center", fontsize=7.5,
                        color=INK2, clip_on=False)

# legend bars (one per panel)
for p, (title, vals, norm, fmt, xl, cap) in enumerate(panels):
    pos = fig.axes[p].get_position()
    cax = fig.add_axes([pos.x0, 0.045 / (0.27 * n / 3.2 + 1) * 0 + 0.5 / (0.27 * n + 3.2), pos.width, 0.012])
    grad = np.linspace(0, 1, 256)[None, :]
    cax.imshow(grad, cmap=CMAP, aspect="auto")
    cax.set_yticks([])
    cax.set_xticks([0, 255])
    lo, hi = {0: ("−0.5", "1.5"), 1: ("100%", "0%"), 2: ("0.5", "≥ 4")}[p]
    cax.set_xticklabels([lo, hi], fontsize=7.5, color=INK2)
    cax.get_xticklabels()[0].set_ha("left")
    cax.get_xticklabels()[1].set_ha("right")
    cax.tick_params(length=0, pad=2)
    for s in cax.spines.values():
        s.set_visible(False)
    cax.set_xlabel(cap, fontsize=7.5, color=INK2, labelpad=2)

fig.suptitle("PSIS reliability by scenario: k̂, effective draws, and error per statistic",
             x=0.02, y=0.992, ha="left", fontsize=12.5, color=INK, fontweight="semibold")
fig.text(0.02, 0.972 - 0.0 * n, "44 scenario × parameter rows sorted by k̂; dashed lines mark k̂ = 0.3, 0.5, 0.7. "
         "Error = |PSIS − refit A| over combined Monte Carlo SE (≈0.8 is pure noise).",
         fontsize=8.3, color=INK2, ha="left", va="top")
fig.savefig(os.path.join(OUT, "khat_ess_heatmap.png"), dpi=170, facecolor=SURFACE)
print("wrote", os.path.join(OUT, "khat_ess_heatmap.png"), "rows:", n)
