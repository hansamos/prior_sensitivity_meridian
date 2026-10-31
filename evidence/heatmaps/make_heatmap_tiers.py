"""Heatmaps by prior-distance tier (close / mid / far) per parameter type.

Two figures, same layout: rows = how much the prior was changed (tier from the total variation distance
between target and reference prior: close < 0.15, mid 0.15-0.4, far >= 0.4; for channel-specific and joint
changes the largest distance over the changed channels/fields), columns = k-hat bins (figure 1) or share of
effective PSIS draws (figure 2). Cell = median over (scenario, parameter) blocks of
    mean |PSIS - refit A| / mean |refit A - refit B|   (means over channels and the four statistics)
i.e. PSIS error as a multiple of the refit-to-refit difference; text also shows n = distinct scenarios.
Reads saved CSVs only. Usage: python -I evidence/heatmaps/make_heatmap_tiers.py
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

RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", RAMP)
SURFACE, INK, INK2 = "#fcfcfb", "#1a1a19", "#52524f"
norm = LogNorm(vmin=0.7, vmax=8.0)

roi = pd.read_csv(os.path.join(ROOT, "regional_roi/results/channel_level_noise.csv"))
ess = pd.read_csv(os.path.join(ROOT, "regional_roi/results/scenario_level.csv"))[["scenario", "ess_fraction"]]
roi = roi.merge(ess, on="scenario")
roi["field"] = "roi_m"
cols = ["scenario", "field", "pareto_k", "ess_fraction", "stat", "e_psis_A", "e_AB"]
d = pd.concat([roi[cols],
               pd.read_csv(os.path.join(ROOT, "regional_other_params/results/channel_level.csv"))[cols],
               pd.read_csv(os.path.join(ROOT, "regional_other_params/results_multi/channel_level.csv"))[cols]], ignore_index=True)
d["ptype"] = np.where(d.scenario.str.match(r"^J\d"), "joint (alpha_m + ec_m)", d.field)

dist = pd.read_csv(os.path.join(OUT, "prior_change_distance.csv"))
dist["key"] = dist.scenario.str.split("_").str[0]
tier_of = dict(zip(dist.key, dist.tier.astype(str)))

blocks = []
for (sc, field, pt), g in d.groupby(["scenario", "field", "ptype"], sort=False):
    blocks.append(dict(scenario=sc, field=field, ptype=pt, tier=tier_of[sc.split("_")[0]],
                       k=g.pareto_k.iloc[0], ess=100 * g.ess_fraction.iloc[0],
                       ratio=g.e_psis_A.abs().mean() / g.e_AB.abs().mean()))
b = pd.DataFrame(blocks)
b["kb"] = pd.cut(b.k, [-np.inf, 0.3, 0.5, 0.7, np.inf], labels=False, right=False)
b["eb"] = pd.cut(b.ess, [-np.inf, 30, 50, 70, np.inf], labels=False, right=False)

TIERS = ["close", "mid", "far"]
TIER_LAB = ["close  (TV < 0.15)", "mid  (0.15 – 0.4)", "far  (TV ≥ 0.4)"]
K_LAB = ["k̂<0.3", "0.3–0.5", "0.5–0.7", "k̂≥0.7"]
E_LAB = ["<30%", "30–50%", "50–70%", "70–100%"]
panels = [("All parameters", b)] + [(pt, b[b.ptype == pt]) for pt in
                                    ["roi_m", "alpha_m", "ec_m", "sigma", "eta_m", "joint (alpha_m + ec_m)"]]


def text_color(v):
    return "#ffffff" if CMAP(norm(v))[0] < 0.45 else INK


def draw(binkey, labels, fname, title, note):
    out_rows = []
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4), facecolor=SURFACE)
    for ax, (ptitle, g) in zip(axes.ravel(), panels):
        ax.set_facecolor(SURFACE)
        for i, tier in enumerate(TIERS):
            for j in range(4):
                cell = g[(g.tier == tier) & (g[binkey] == j)]
                if len(cell) == 0:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#f1f0ec", edgecolor=SURFACE,
                                               linewidth=2))
                    ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="#a3a29d")
                    continue
                v, n = float(np.median(cell.ratio)), cell.scenario.nunique()
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=CMAP(norm(v)), edgecolor=SURFACE,
                                           linewidth=2))
                ax.text(j, i - 0.12, f"{v:.1f}×", ha="center", va="center", fontsize=12, fontweight="semibold",
                        color=text_color(v))
                ax.text(j, i + 0.24, f"n = {n}", ha="center", va="center", fontsize=8.5, color=text_color(v))
                out_rows.append(dict(panel=ptitle, tier=tier, bin=labels[j], n_scenarios=n, n_blocks=len(cell),
                                     median_ratio=v, min_ratio=cell.ratio.min(), max_ratio=cell.ratio.max()))
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(2.5, -0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels, fontsize=8, color=INK2)
        ax.set_yticks(range(3))
        ntier = [g[g.tier == t].scenario.nunique() for t in TIERS]
        ax.set_yticklabels([f"{TIER_LAB[i]}  [{ntier[i]}]" for i in range(3)], fontsize=8.3, color=INK2)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{ptitle}  ({g.scenario.nunique()} scenarios)", fontsize=10.5, color=INK,
                     fontweight="semibold", pad=26, loc="left")
    ax = axes.ravel()[-1]
    ax.axis("off")
    ax.text(0.0, 1.0, "How to read", fontsize=10.5, color=INK, fontweight="semibold", transform=ax.transAxes,
            va="top")
    ax.text(0.0, 0.90, note, fontsize=8.8, color=INK2, transform=ax.transAxes, va="top", linespacing=1.45)
    cax = ax.inset_axes([0.0, 0.14, 0.92, 0.06])
    cax.imshow(np.linspace(0, 1, 256)[None, :], cmap=CMAP, aspect="auto")
    cax.set_yticks([])
    ticks = [0.7, 1, 2, 4, 8]
    cax.set_xticks([255 * (np.log(t) - np.log(0.7)) / (np.log(8.0) - np.log(0.7)) for t in ticks])
    cax.set_xticklabels(["≤0.7", "1", "2", "4", "≥8"], fontsize=8.5, color=INK2)
    cax.set_xlabel("PSIS error ÷ refit-to-refit difference (log scale)", fontsize=8.3, color=INK2, labelpad=3)
    cax.tick_params(length=0)
    for sp in cax.spines.values():
        sp.set_visible(False)
    fig.suptitle(title, x=0.015, y=0.995, ha="left", fontsize=13.5, color=INK, fontweight="semibold")
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.04, wspace=0.62, hspace=0.42)
    fig.savefig(os.path.join(OUT, fname), dpi=170, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(out_rows)


common = ("Rows: how much the prior was changed\n(total variation distance, target vs\nreference prior); [n] = scenarios in tier.\n"
          "Cell = median PSIS error as a multiple of\nthe refit-to-refit difference; 1× = as close\nas two refits; darker = worse.\n"
          "n = scenarios in the cell; – = none.")
r1 = draw("kb", K_LAB, "khat_by_distance_tier_heatmap.png",
          "PSIS error by prior-change size (rows) and k̂ bin (columns), per parameter type", common)
r2 = draw("eb", E_LAB, "ess_by_distance_tier_heatmap.png",
          "PSIS error by prior-change size (rows) and effective-draw share (columns), per parameter type", common)
r1.assign(binning="khat").to_csv(os.path.join(OUT, "khat_by_distance_tier_data.csv"), index=False)
r2.assign(binning="ess").to_csv(os.path.join(OUT, "ess_by_distance_tier_data.csv"), index=False)
print("cells:", len(r1), len(r2))
