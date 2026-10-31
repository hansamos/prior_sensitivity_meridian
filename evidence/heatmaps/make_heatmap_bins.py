"""k-hat bin x ESS bin heatmaps of PSIS error, pooled and per parameter type.

Cell = median over (scenario, parameter) blocks in that bin of
    mean |PSIS - refit A| / mean |refit A - refit B|        (means over channels and the four statistics)
i.e. PSIS error as a multiple of the refit-to-refit difference; about 1 means "as close to a refit as two refits
are to each other". Text shows that multiple and n = number of distinct scenarios in the cell.
Bins: k-hat <0.3, [0.3,0.5), [0.5,0.7), >=0.7; effective draws <30%, [30,50), [50,70), [70,100]%.
Reads saved CSVs only. Usage: python -I evidence/heatmaps/make_heatmap_bins.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#52524f"

roi = pd.read_csv(os.path.join(ROOT, "regional_roi/results/channel_level_noise.csv"))
ess = pd.read_csv(os.path.join(ROOT, "regional_roi/results/scenario_level.csv"))[["scenario", "ess_fraction"]]
roi = roi.merge(ess, on="scenario")
roi["field"] = "roi_m"
cols = ["scenario", "field", "pareto_k", "ess_fraction", "stat", "e_psis_A", "e_AB"]
d = pd.concat([roi[cols],
               pd.read_csv(os.path.join(ROOT, "regional_other_params/results/channel_level.csv"))[cols],
               pd.read_csv(os.path.join(ROOT, "regional_other_params/results_multi/channel_level.csv"))[cols]], ignore_index=True)

# parameter type: joint changes get their own panel; channel-specific alpha_m stays in alpha_m
d["ptype"] = np.where(d.scenario.str.match(r"^J\d"), "joint (alpha_m + ec_m)", d.field)

blocks = []
for (sc, field, pt), g in d.groupby(["scenario", "field", "ptype"], sort=False):
    blocks.append(dict(scenario=sc, field=field, ptype=pt, k=g.pareto_k.iloc[0], ess=100 * g.ess_fraction.iloc[0],
                       psis_err=g.e_psis_A.abs().mean(), ab_err=g.e_AB.abs().mean()))
b = pd.DataFrame(blocks)
b["ratio"] = b.psis_err / b.ab_err

K_EDGES, K_LAB = [-np.inf, 0.3, 0.5, 0.7, np.inf], ["k̂ < 0.3", "0.3 – 0.5", "0.5 – 0.7", "k̂ ≥ 0.7"]
E_EDGES, E_LAB = [-np.inf, 30, 50, 70, np.inf], ["< 30%", "30 – 50%", "50 – 70%", "70 – 100%"]
b["kb"] = pd.cut(b.k, K_EDGES, labels=False, right=False)
b["eb"] = pd.cut(b.ess, E_EDGES, labels=False, right=False)

panels = [("All parameters", b)] + [(pt, b[b.ptype == pt]) for pt in
                                    ["roi_m", "alpha_m", "ec_m", "sigma", "eta_m", "joint (alpha_m + ec_m)"]]
norm = LogNorm(vmin=0.7, vmax=8.0)


def text_color(v):
    return "#ffffff" if CMAP(norm(v))[0] < 0.45 else INK


rows_out = []
fig, axes = plt.subplots(2, 4, figsize=(15.5, 8.6), facecolor=SURFACE)
for ax, (title, g) in zip(axes.ravel(), panels):
    ax.set_facecolor(SURFACE)
    for i in range(4):
        for j in range(4):
            cell = g[(g.kb == i) & (g.eb == j)]
            if len(cell) == 0:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0ec", edgecolor=SURFACE,
                                           linewidth=2, hatch="////", alpha=0.0))
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0ec", edgecolor=SURFACE,
                                           linewidth=2))
                ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="#a3a29d")
                continue
            v = float(np.median(cell.ratio))
            n = cell.scenario.nunique()
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CMAP(norm(v)), edgecolor=SURFACE,
                                       linewidth=2))
            ax.text(j, i - 0.12, f"{v:.1f}×", ha="center", va="center", fontsize=12, fontweight="semibold",
                    color=text_color(v))
            ax.text(j, i + 0.22, f"n = {n}", ha="center", va="center", fontsize=8.5, color=text_color(v))
            rows_out.append(dict(panel=title, k_bin=K_LAB[i], ess_bin=E_LAB[j], n_scenarios=n, n_blocks=len(cell),
                                 median_ratio=v, min_ratio=cell.ratio.min(), max_ratio=cell.ratio.max()))
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(3.5, -0.5)
    ax.set_xticks(range(4))
    ax.set_xticklabels(E_LAB, fontsize=8.5, color=INK2)
    ax.set_yticks(range(4))
    ax.set_yticklabels(K_LAB, fontsize=8.5, color=INK2)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"{title}  ({g.scenario.nunique()} scenarios)", fontsize=10.5, color=INK, fontweight="semibold",
                 pad=30, loc="left")

# last slot: legend and reading guide
ax = axes.ravel()[-1]
ax.axis("off")
ax.text(0.0, 1.0, "How to read", fontsize=10.5, color=INK, fontweight="semibold", transform=ax.transAxes, va="top")
ax.text(0.0, 0.90,
        "Cell = median PSIS error as a multiple\nof the refit-to-refit difference.\n"
        "1x = PSIS is as close to a refit as two\nrefits are to each other; darker = worse.\n"
        "n = distinct scenarios in the cell.\n"
        "\u2013 = no scenario in that cell.",
        fontsize=8.8, color=INK2, transform=ax.transAxes, va="top", linespacing=1.45)
ax.text(0.0, 0.54,
        "Rows: k\u0302 bin (low at top).\nColumns: share of reference draws\nused effectively by PSIS.",
        fontsize=8.8, color=INK2, transform=ax.transAxes, va="top", linespacing=1.45)
cax = ax.inset_axes([0.0, 0.17, 0.92, 0.06])
cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect="auto")
cax.set_yticks([])
ticks = [0.7, 1, 2, 4, 8]
cax.set_xticks([255 * (np.log(t) - np.log(0.7)) / (np.log(8.0) - np.log(0.7)) for t in ticks])
cax.set_xticklabels(["\u22640.7", "1", "2", "4", "\u22658"], fontsize=8.5, color=INK2)
cax.set_xlabel("PSIS error \u00f7 refit-to-refit difference (log scale)", fontsize=8.3, color=INK2, labelpad=3)
cax.tick_params(length=0)
for sp in cax.spines.values():
    sp.set_visible(False)

fig.suptitle("PSIS error by k̂ bin and effective-draw bin, per parameter type", x=0.015, y=0.995, ha="left",
             fontsize=13.5, color=INK, fontweight="semibold")
fig.subplots_adjust(left=0.07, right=0.985, top=0.83, bottom=0.04, wspace=0.42, hspace=0.38)
fig.savefig(os.path.join(OUT, "khat_ess_bins_heatmap.png"), dpi=170, facecolor=SURFACE)
pd.DataFrame(rows_out).to_csv(os.path.join(OUT, "khat_ess_bins_heatmap_data.csv"), index=False)
print("wrote figure; non-empty cells:", len(rows_out))
