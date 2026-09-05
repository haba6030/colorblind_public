#!/usr/bin/env python3
"""
figS — appendix companion to fig8. Forward-tuning LOCO-rho (FE-6 forward-encoder,
voxel-pattern prediction) across V1-hV4 for the two CVD subjects.

This is the secondary/encoding metric split out of the old fig8 2x2 so the main
fig8 stays on the three HC-likeness metrics (adjacc / SRM disparity / RDM).

Layout: 2 rows (subjects) x 1 column
  Row 1 = Deutan (sub-08),  Row 2 = Protan (sub-09)
  Metric = LOCO forward-tuning rho (higher = HC-like)

Conditions (display renaming; JSON keys unchanged):
  nofilter -> "No-filter"     (gray circle)
  window   -> "Deployed"      (blue square)
  optimal  -> "Personalized"  (orange diamond)

Usage: python generate_figS3_forward_tuning.py [--variant native|matched]
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : docs/PAPER/Figures/scripts/generate_figS3_forward_tuning.py  (development repository, commit 53c81c2)
# Role      : Figure S3 (forward-tuning rho under each filter).
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[3]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats

BASE = Path("/Users/jinilkim/LocalProj/colorBlind_analysis")
RESDIR = BASE / "analysis/phase6_behavioral_analysis/exp2_neural/results"
OUTDIR = BASE / "docs/PAPER/Figures"

ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABELS = ["V1", "V2", "V3", "hV4"]
SUBJECTS = [("08", "Deutan"), ("09", "Protan")]

# Wong colorblind-safe palette (matched to fig8)
HC_BAR = "#CCCCCC"
C_NOFILT = "#666666"
C_WINDOW = "#0072B2"
C_OPTIM = "#D55E00"
COND_STYLE = {
    "nofilter": (C_NOFILT, "o", "No-filter"),
    "window":   (C_WINDOW, "s", "Deployed"),
    "optimal":  (C_OPTIM,  "D", "Individualized"),
}
CONDS = ["nofilter", "window", "optimal"]


def crawford_howell(x, mean_hc, sd_hc, n, tail):
    if sd_hc <= 0 or not np.isfinite(sd_hc):
        return 0.0, 1.0
    t = (x - mean_hc) / (sd_hc * np.sqrt((n + 1) / n))
    if tail == "lower":
        p = stats.t.cdf(t, df=n - 1)
    else:
        p = stats.t.sf(t, df=n - 1)
    return float(t), float(p)


def sig_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_data(hl):
    d = {"hc_mean": [], "hc_sd": [], "hc_dots": [], "n": [],
         "nofilter": [], "window": [], "optimal": [], "tail": "lower"}
    for roi in ROIS:
        h = hl[roi]
        d["hc_mean"].append(h["hc_loco_rho_n4_mean"])
        d["hc_sd"].append(h["hc_loco_rho_n4_sd"])
        d["hc_dots"].append(h["hc_loco_rho_n4_values"])
        d["n"].append(h["hc_n"])
        d["nofilter"].append(h["nofilter_baseline_exp1"]["loco_rho_n4_matched"])
        d["window"].append(h["conditions"]["window"]["loco_rho_mean"])
        d["optimal"].append(h["conditions"]["optimal"]["loco_rho_mean"])
    return d


def draw_panel(ax, d, letter, ylim, show_title):
    x = np.arange(len(ROIS)); bw = 0.6
    hc_mean = np.array(d["hc_mean"], float); hc_sd = np.array(d["hc_sd"], float)

    ax.bar(x, hc_mean, width=bw, color=HC_BAR, zorder=2, linewidth=0)
    if np.isfinite(hc_sd).any():
        ax.errorbar(x, hc_mean, yerr=np.nan_to_num(hc_sd), fmt="none",
                    color="#333", capsize=2.5, linewidth=0.9, zorder=3)

    offs = {"nofilter": -0.22, "window": 0.0, "optimal": 0.22}
    for cond in CONDS:
        col, mk, _ = COND_STYLE[cond]
        yv = np.array(d[cond], float)
        ax.plot(x + offs[cond], yv, marker=mk, color=col, markersize=6, linewidth=0,
                markeredgecolor="white", markeredgewidth=0.6, zorder=5)

    ax.set_ylim(ylim)
    ylo, yhi = ax.get_ylim(); yr = yhi - ylo
    for cond in CONDS:
        col = COND_STYLE[cond][0]
        yv = np.array(d[cond], float)
        for i in range(len(ROIS)):
            if not np.isfinite(hc_sd[i]):
                continue
            _, p = crawford_howell(yv[i], hc_mean[i], hc_sd[i], d["n"][i], d["tail"])
            s = sig_star(p)
            if s:
                ax.text(x[i] + offs[cond], yv[i] + yr * 0.03, s, fontsize=6.5, color=col,
                        ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(ROI_LABELS, fontsize=7)
    ax.set_ylabel("Forward-tuning ρ", fontsize=7, labelpad=2)
    ax.tick_params(axis="both", labelsize=6.5, length=3)
    ax.axhline(0, color="#bbb", linewidth=0.6, zorder=0)
    strip(ax)

    ax.text(-0.09, 1.04, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")
    if show_title:
        ax.text(0.5, 1.14, "LOCO forward-tuning ρ", transform=ax.transAxes, fontsize=7.5,
                va="bottom", ha="center", color="#222", fontweight="bold")
        ax.text(0.5, 1.04, "Encoding — voxel-pattern prediction  (↑ control-like)",
                transform=ax.transAxes, fontsize=6.3, va="bottom", ha="center", color="#666")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="matched", choices=["native", "matched"])
    args = ap.parse_args()

    data = {}
    for sub, _ in SUBJECTS:
        hl = json.load(open(RESDIR / f"exp2_hc_likeness_sub-{sub}_{args.variant}.json"))
        data[sub] = panel_data(hl)

    # Shared y-limits across both rows
    vals = []
    for sub, _ in SUBJECTS:
        d = data[sub]
        vals += list(d["hc_mean"])
        for cond in CONDS:
            vals += list(d[cond])
        for i in range(len(ROIS)):
            if d["hc_dots"][i] is not None:
                vals += list(d["hc_dots"][i])
        hcm = np.array(d["hc_mean"], float); hcs = np.array(d["hc_sd"], float)
        vals += list(hcm + np.nan_to_num(hcs))
        vals += list(hcm - np.nan_to_num(hcs))
    vlo, vhi = min(vals), max(vals)
    pad = (vhi - vlo) * 0.20 + 1e-6
    ylim = (vlo - pad, vhi + pad)

    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 82 * mm))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.985, top=0.78, bottom=0.20, wspace=0.24)

    letters = ["A", "B"]
    for c, (sub, col_label) in enumerate(SUBJECTS):
        ax = fig.add_subplot(gs[0, c])
        draw_panel(ax, data[sub], letters[c], ylim, show_title=True)
        ax.text(0.5, 1.30, col_label, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="bottom", ha="center", color="#111")

    handles = [
        mpatches.Patch(facecolor=HC_BAR, label="Control reference (mean ± SD)", edgecolor="none"),
    ] + [Line2D([0], [0], marker=COND_STYLE[c][1], color="w", markerfacecolor=COND_STYLE[c][0],
                markersize=6, label=COND_STYLE[c][2], linewidth=0) for c in CONDS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, 0.02), handletextpad=0.4, columnspacing=1.6)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / "figS3_forward_tuning.png"
    pdf = OUTDIR / "figS3_forward_tuning.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    print("\n=== figS anchors (forward-tuning ρ) ===")
    for sub in ["08", "09"]:
        d = data[sub]; i = ROIS.index("V4")
        print(f"rho hV4 sub-{sub}: NF {d['nofilter'][i]:.2f} Dep {d['window'][i]:.2f} "
              f"Pers {d['optimal'][i]:.2f} Ctrl {d['hc_mean'][i]:.2f}")


if __name__ == "__main__":
    main()
