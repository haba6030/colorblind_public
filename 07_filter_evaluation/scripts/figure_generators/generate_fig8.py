#!/usr/bin/env python3
"""
fig8 — exp2 neural filter evaluation (N=2: deutan sub-08, protan sub-09).

Paper version. Output → docs/PAPER/Figures/fig8_filter_eval.{png,pdf}
(no suptitle — the LaTeX caption carries the title, matching fig2/fig3 style).

Layout: 2 rows (subjects) x 3 columns (metrics)
  Row 1 = Deutan (sub-08),  Row 2 = Protan (sub-09)
  Col A = LOCO adjacent accuracy (interpolation; chance 91/360 = 0.25; higher = HC-like)
  Col B = SRM disparity (shared-space alignment; lower = HC-like)
  Col C = RDM similarity to HC (Spearman; higher = HC-like)

The forward-tuning LOCO-rho panel now lives in the appendix figure
(generate_figS3_forward_tuning.py).

Conditions (display renaming; JSON keys unchanged):
  nofilter -> "No-filter"     (gray circle)
  window   -> "Deployed"      (blue square)
  optimal  -> "Personalized"  (orange diamond)

Usage: python generate_fig8.py [--variant native|matched]   (paper uses matched)
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : docs/PAPER/Figures/scripts/generate_fig8.py  (development repository, commit 53c81c2)
# Role      : Figure 7 (interpolation and geometry under each filter).
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

# Adjacent-accuracy chance. The forward-encoding readout takes an argmax over all
# 360 integer hues (utils_forward_model.decode_hue), and adjacent accuracy counts a
# prediction correct when its circular error is <= 45 deg (loco_canonical.py). A
# prediction drawn uniformly from that 360-hue output space lands inside the
# tolerance on 91 of 360 draws. Verified by simulation (0.253 over 20,000 draws).
# NOT 3/8 -- that holds only for decoders that output one of the eight stimulus hues.
CHANCE_ADJ = 91 / 360
from scipy import stats

BASE = Path("/Users/jinilkim/LocalProj/colorBlind_analysis")
RESDIR = BASE / "analysis/phase6_behavioral_analysis/exp2_neural/results"
OUTDIR = BASE / "docs/PAPER/Figures"

ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABELS = ["V1", "V2", "V3", "hV4"]

SUBJECTS = [("08", "Deutan"), ("09", "Protan")]

# Wong colorblind-safe palette
HC_BAR = "#CCCCCC"
HC_DOT = "#555555"
C_NOFILT = "#666666"   # No-filter
C_WINDOW = "#0072B2"   # Deployed (blue)
C_OPTIM = "#D55E00"    # Personalized (orange = focus)
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
    if tail == "lower":   # deficit = x below HC
        p = stats.t.cdf(t, df=n - 1)
    else:                 # "upper": deficit = x above HC (disparity)
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


# metric -> (tail, higher_better, short title, y-label)
METRICS = {
    "adjacc": ("lower", True,  "LOCO adjacent accuracy", "Adjacent accuracy"),
    "srm":    ("upper", False, "SRM disparity",          "SRM disparity"),
    "rdm":    ("lower", True,  "RDM similarity to controls",    "Spearman ρ to controls"),
}
METRIC_ORDER = ["adjacc", "srm", "rdm"]


def panel_data(hl, cv, metric, rm=None):
    tail = METRICS[metric][0]
    d = {"hc_mean": [], "hc_sd": [], "hc_dots": [], "n": [],
         "nofilter": [], "window": [], "optimal": [], "tail": tail}
    for roi in ROIS:
        h = hl[roi]
        # Geometry panels use the run-matched estimates (HC reference and the
        # unfiltered baseline rebuilt from 4 runs over all C(6,4)=15 subsets) so
        # that every panel compares equal amounts of data. The interpolation
        # panel was already run-matched via the *_n4 fields. exp2_convergent's
        # 6-run geometry values are retained in that file for reference only.
        c = (rm["rois"][roi] if rm is not None else cv[roi])
        if metric == "adjacc":
            d["hc_mean"].append(h["hc_loco_adjacc_n4_mean"])
            d["hc_sd"].append(h["hc_loco_adjacc_n4_sd"])
            d["hc_dots"].append(h["hc_loco_adjacc_n4_values"])
            d["n"].append(h["hc_n"])
            d["nofilter"].append(h["nofilter_baseline_exp1"]["loco_adjacc_n4_matched"])
            d["window"].append(h["conditions"]["window"]["loco_adjacc_mean"])
            d["optimal"].append(h["conditions"]["optimal"]["loco_adjacc_mean"])
        elif metric == "srm":
            s = c["srm"]
            d["hc_mean"].append(s["hc_disp_mean"]); d["hc_sd"].append(s["hc_disp_sd"])
            d["hc_dots"].append(None); d["n"].append(h["hc_n"])
            d["nofilter"].append(s["conditions"]["nofilter"]["disparity"])
            d["window"].append(s["conditions"]["window"]["disparity"])
            d["optimal"].append(s["conditions"]["optimal"]["disparity"])
        elif metric == "rdm":
            rp = c["srm_rdm_paper"]
            d["hc_mean"].append(rp["_hc"]["spearman_self_loo_mean"])
            d["hc_sd"].append(np.nan)
            d["hc_dots"].append(None); d["n"].append(h["hc_n"])
            d["nofilter"].append(rp["nofilter"]["spearman_to_hc"])
            d["window"].append(rp["window"]["spearman_to_hc"])
            d["optimal"].append(rp["optimal"]["spearman_to_hc"])
    return d


def draw_panel(ax, d, metric, letter, ylim=None, show_title=True):
    tail, higher_better, title, ylabel = METRICS[metric]
    x = np.arange(len(ROIS)); bw = 0.6
    hc_mean = np.array(d["hc_mean"], float); hc_sd = np.array(d["hc_sd"], float)

    # In the interpolation panels V1-V3 carry no interpretation (the controls do
    # not exceed the color-label permutation null there): transparent bars with a
    # dashed outline, muted markers, and no significance stars.
    muted = {0, 1, 2} if metric == "adjacc" else set()

    bars = ax.bar(x, hc_mean, width=bw, color=HC_BAR, zorder=2, linewidth=0)
    for i in muted:
        bars[i].set_facecolor("none")
        bars[i].set_edgecolor("#aaaaaa")
        bars[i].set_linestyle("--")
        bars[i].set_linewidth(0.9)
    if np.isfinite(hc_sd).any():
        for i in range(len(ROIS)):
            if not np.isfinite(hc_sd[i]):
                continue
            ax.errorbar([x[i]], [hc_mean[i]], yerr=[hc_sd[i]], fmt="none",
                        color="#333", capsize=2.5, linewidth=0.9, zorder=3,
                        alpha=0.35 if i in muted else 1.0)

    offs = {"nofilter": -0.22, "window": 0.0, "optimal": 0.22}
    for cond in CONDS:
        col, mk, _ = COND_STYLE[cond]
        yv = np.array(d[cond], float)
        for i in range(len(ROIS)):
            ax.plot([x[i] + offs[cond]], [yv[i]], marker=mk, color=col, markersize=6,
                    linewidth=0, markeredgecolor="white", markeredgewidth=0.6, zorder=5,
                    alpha=0.35 if i in muted else 1.0)

    if ylim is not None:
        ax.set_ylim(ylim)
    ylo, yhi = ax.get_ylim(); yr = yhi - ylo
    for cond in CONDS:
        col = COND_STYLE[cond][0]
        yv = np.array(d[cond], float)
        for i in range(len(ROIS)):
            if i in muted or not np.isfinite(hc_sd[i]):
                continue
            _, p = crawford_howell(yv[i], hc_mean[i], hc_sd[i], d["n"][i], d["tail"])
            s = sig_star(p)
            if s:
                ax.text(x[i] + offs[cond], yv[i] + yr * 0.03, s, fontsize=6.5, color=col,
                        ha="center", va="bottom", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(ROI_LABELS, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.tick_params(axis="both", labelsize=6.5, length=3)
    ax.axhline(0, color="#bbb", linewidth=0.6, zorder=0)
    if metric == "adjacc":
        ax.axhline(CHANCE_ADJ, color="#999", linestyle="--", linewidth=0.7, zorder=1)
    strip(ax)

    ax.text(-0.05, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")
    if show_title:
        arrow = "  (↑ control-like)" if higher_better else "  (↓ control-like)"
        ax.text(0.5, 1.14, title, transform=ax.transAxes, fontsize=7.3, va="bottom",
                ha="center", color="#222", fontweight="bold")
        ax.text(0.5, 1.05, arrow.strip(), transform=ax.transAxes, fontsize=6.2,
                va="bottom", ha="center", color="#666")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="matched", choices=["native", "matched"])
    args = ap.parse_args()

    # Load both subjects
    data = {}
    for sub, _ in SUBJECTS:
        hl = json.load(open(RESDIR / f"exp2_hc_likeness_sub-{sub}_{args.variant}.json"))
        cv = json.load(open(RESDIR / f"exp2_convergent_sub-{sub}_{args.variant}.json"))
        rm_path = RESDIR / f"exp2_runmatched_geometry_sub-{sub}_{args.variant}.json"
        rm = json.load(open(rm_path)) if rm_path.exists() else None
        if rm is None:
            raise SystemExit(f"missing run-matched geometry: {rm_path}")
        data[sub] = {m: panel_data(hl, cv, m, rm) for m in METRIC_ORDER}

    # Shared y-limits per metric column (across both subject rows)
    ylims = {}
    for m in METRIC_ORDER:
        vals = []
        for sub, _ in SUBJECTS:
            d = data[sub][m]
            vals += list(d["hc_mean"])
            for cond in CONDS:
                vals += list(d[cond])
            for i in range(len(ROIS)):
                if d["hc_dots"][i] is not None:
                    vals += list(d["hc_dots"][i])
            hcs = np.array(d["hc_sd"], float)
            hcm = np.array(d["hc_mean"], float)
            if np.isfinite(hcs).any():
                vals += list(hcm + np.nan_to_num(hcs))
        vlo, vhi = min(vals), max(vals)
        if m == "adjacc":
            vhi = max(vhi, CHANCE_ADJ)
        pad = (vhi - vlo) * 0.20 + 1e-6
        lo = min(vlo - pad * 0.5, 0.0)
        ylims[m] = (lo, vhi + pad)

    mm = 1 / 25.4
    fig = plt.figure(figsize=(180 * mm, 118 * mm))
    gs = fig.add_gridspec(2, 3, left=0.115, right=0.985, top=0.86, bottom=0.175,
                          wspace=0.36, hspace=0.42)

    panel_letters = [["A", "B", "C"], ["D", "E", "F"]]
    for r, (sub, row_label) in enumerate(SUBJECTS):
        for cc, m in enumerate(METRIC_ORDER):
            ax = fig.add_subplot(gs[r, cc])
            draw_panel(ax, data[sub][m], m, panel_letters[r][cc],
                       ylim=ylims[m], show_title=(r == 0))
            if cc == 0:
                ax.text(-0.42, 0.5, row_label, transform=ax.transAxes, fontsize=8.5,
                        fontweight="bold", va="center", ha="center", rotation=90,
                        color="#111")

    handles = [
        mpatches.Patch(facecolor=HC_BAR, label="Control reference (mean ± SD)", edgecolor="none"),
        Line2D([0], [0], color="#999", linestyle="--", linewidth=0.9, label="chance (91/360)"),
    ] + [Line2D([0], [0], marker=COND_STYLE[c][1], color="w", markerfacecolor=COND_STYLE[c][0],
                markersize=6, label=COND_STYLE[c][2], linewidth=0) for c in CONDS]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.8, frameon=False,
               bbox_to_anchor=(0.5, 0.03), handletextpad=0.4, columnspacing=1.4)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    png = OUTDIR / "fig8_filter_eval.png"
    pdf = OUTDIR / "fig8_filter_eval.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")

    # Verification anchors
    print("\n=== VERIFICATION ANCHORS (rendered values) ===")
    def row(sub, m, roi):
        d = data[sub][m]; i = ROIS.index(roi)
        return d["nofilter"][i], d["window"][i], d["optimal"][i], d["hc_mean"][i]
    for sub in ["08", "09"]:
        nf, wn, op, hc = row(sub, "adjacc", "V4")
        print(f"adjacc hV4 sub-{sub}: NF {nf:.2f} Dep {wn:.2f} Pers {op:.2f} Ctrl {hc:.2f}")
    for sub, roi in [("08", "V2"), ("09", "V1")]:
        nf, wn, op, hc = row(sub, "srm", roi)
        print(f"srm {roi} sub-{sub}: NF {nf:.2f} Dep {wn:.2f} Pers {op:.2f} Ctrl {hc:.2f}")
    for sub, roi in [("08", "V2"), ("09", "V1")]:
        nf, wn, op, hc = row(sub, "rdm", roi)
        print(f"rdm {roi} sub-{sub}: NF {nf:.2f} Dep {wn:.2f} Pers {op:.2f} Ctrl-self {hc:.2f}")


if __name__ == "__main__":
    main()
