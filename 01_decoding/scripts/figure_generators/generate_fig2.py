#!/usr/bin/env python3
"""
Figure 2 — LORO (discrimination) vs LOCO (interpolation) dissociation in CVD
=============================================================================
Panel A: LORO ForwardEncoding accuracy — discrimination preserved in CVD
Panel B: LOCO adjacent_acc (ForwardEncoding) — interpolation impaired in hV4
Panel C: Per-hue adjacent_acc at hV4

Alignment space (2026-08-07). Both panels use the PROCRUSTES-aligned amplitudes,
which are the canonical Phase-1 input (utils_forward_model.load_amplitudes reads
amplitudes_procrustes.npy and has no SRM path, so loco_canonical.py and the
adjacent-accuracy permutation test both run in this space). Before this revision
panel B mixed spaces -- HC came from results/loco_srm while CVD came from
results/loco_decoding_comparison, which is procrustes -- and panel A was SRM while
the reported statistics were procrustes. SRM is retained only where a common space
is required (cross-subject transfer, representational geometry) and the SRM
LORO/LOCO tables are reported in Supplementary S20 as an alignment robustness check.

Sample: n = 7 HC at every ROI. sub-07 is retained at hV4 despite its low voxel
count (16 vs 67-70 in the other six); excluding it leaves every conclusion
unchanged and slightly weakens the CVD contrasts.

Chance: Panel A = 1/8 = 0.125 (exact). Panels B/C = 91/360 = 0.25 (adjacent);
see the CHANCE_ADJ comment below for why this is not 3/8.
Statistics: Crawford & Howell t-test vs the HC distribution. Panel A is two-tailed
(the hypothesis is preservation); panels B/C are one-tailed lower (directional deficit).
Panel B annotates significance only at regions passing the permutation gate
(PERM_PASS_ROIS) -- accuracies for all four regions are plotted regardless.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : docs/PAPER/Figures/scripts/generate_fig2.py  (development repository, commit 53c81c2)
# Role      : Figure 4 (LORO / LOCO across V1-hV4).
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
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import sys
from pathlib import Path
from scipy import stats

# Repo root, resolved from this file: docs/PAPER/Figures/scripts/ -> parents[4].
BASE = str(Path(__file__).resolve().parents[4])
LORO_DIR    = f"{BASE}/analysis/phase3_decoder_comparing/results/loro/procrustes"
C010_DIR    = (f"{BASE}/analysis/phase1_procrustes_decoding/results/visualization/"
               f"full_dataset_C010_with_residuals")
OUT_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# LOCO is computed here by the canonical Phase-1 routine rather than read from
# phase3_decoder_comparing/results/loco/procrustes. Those stored values come from a
# second implementation (test_decoding_methods.py -> loro_baseline.loco_cv) and agree
# with the canonical routine at 35 of the 36 subject-by-ROI cells; the exception is
# sub-07 hV4 (16 voxels), which differs by 0.0042. Computing from loco_canonical
# keeps the figure, the reported statistics and the permutation null on one
# implementation. Runtime is about a minute.
sys.path.insert(0, f"{BASE}/analysis/phase4_forward_model/scripts")
from loco_canonical import loco_forward_readouts
from utils_forward_model import create_basis_matrix, HUE_ANGLES

SUBS_HC  = [f"{i:02d}" for i in range(1, 8)]
SUBS_CVD = ["08", "09"]
ROIS     = ["V1", "V2", "V3", "V4"]
ROI_LABELS  = ["V1", "V2", "V3", "hV4"]
HUE_NAMES   = ["Red", "Org", "Yel", "Grn", "Cyn", "Blu", "Pur", "Mag"]

CHANCE_EXACT = 0.125
# Adjacent-accuracy chance. The forward-encoding readout takes an argmax over all
# 360 integer hues (utils_forward_model.decode_hue), and adjacent accuracy counts
# a prediction correct when its circular error is <= 45 deg (loco_canonical.py).
# A prediction drawn uniformly from that 360-hue output space therefore lands
# inside the tolerance on 91 of 360 draws. Verified by simulation: 20,000 random
# channel responses give 0.253 (per-true-hue range 0.249-0.257) and an argmax
# that is uniform over the eight 45-deg bins.
# NOT 3/8 -- that value holds only for decoders whose output is one of the eight
# stimulus hues (LDA/SVM/MLP via labels_to_hue in loco_baseline.py).
CHANCE_ADJ   = 91 / 360

# Regions credited with hue interpolation, i.e. those whose healthy-control mean
# exceeds the per-subject color-label permutation null (Methods, Cross-color
# interpolation). Single-case CVD comparisons are pre-specified to be reported
# only in these regions, so panel B annotates significance only here. Accuracies
# for all four regions are plotted regardless.
PERM_PASS_ROIS = {"V4"}

HC_COLOR  = "#AAAAAA"
HC_DOT    = "#555555"
S08_COLOR = "#D55E00"
S09_COLOR = "#009E73"
CHANCE_COL = "#888888"

# STIM_LAB palette (8 hue stimuli, same Lab values as Fig 1 / Fig 3)
COLOR_LAB = [
    [59.90,  62.69,   3.78],   # Red
    [64.20,  49.20,  45.58],   # Orange
    [57.27,  13.06,  41.69],   # Yellow
    [69.08, -55.02,  47.38],   # Green
    [74.61, -41.33,  -4.89],   # Cyan
    [69.14, -11.45, -40.91],   # Blue
    [60.68,  19.18, -54.13],   # Purple
    [60.17,  46.82, -40.31],   # Magenta
]

def _lab2rgb(L, a, b):
    y = (L + 16)/116; x = a/500 + y; z = y - b/200
    xyz = np.array([x, y, z])
    xyz = np.where(xyz > 0.206893, xyz**3, (xyz - 16/116)/7.787)
    xyz *= [0.95047, 1.0, 1.08883]
    M = np.array([[3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [0.0557, -0.2040,  1.0570]])
    rgb = M @ xyz
    rgb = np.where(rgb <= 0.0031308, 12.92*rgb, 1.055*rgb**(1/2.4) - 0.055)
    return tuple(np.clip(rgb, 0, 1))

HUE_RGB = [_lab2rgb(*lab) for lab in COLOR_LAB]

# ── Load data ─────────────────────────────────────────────────────────────────
# LOCO: canonical FE-6 OLS readout, per-hue adjacent accuracy (loco_canonical).
_HUES  = np.asarray(HUE_ANGLES, dtype=float)
_C8    = create_basis_matrix(_HUES, 6, "fe")
_BASIS = create_basis_matrix(np.arange(360), 6, "fe")

def _loco_per_hue(sub, roi):
    amp = np.load(f"{C010_DIR}/sub-{sub}/{roi}/amplitudes_procrustes.npy")
    return loco_forward_readouts(amp, _C8, basis_full=_BASIS,
                                 decoder="ols", tasks=("adj",))["adj"]

loco_per_hue = {sub: {roi: _loco_per_hue(sub, roi) for roi in ROIS}
                for sub in SUBS_HC + SUBS_CVD}

loco_acc = {sub: {roi: float(loco_per_hue[sub][roi].mean()) for roi in ROIS}
            for sub in SUBS_HC + SUBS_CVD}

hue_acc_08 = {i: float(loco_per_hue["08"]["V4"][i]) for i in range(8)}
hue_acc_09 = {i: float(loco_per_hue["09"]["V4"][i]) for i in range(8)}
hue_acc_hc = {i: [float(loco_per_hue[s]["V4"][i]) for s in SUBS_HC] for i in range(8)}

hue_acc_hc_mean = np.array([np.mean(hue_acc_hc[i]) for i in range(8)])
hue_acc_hc_sem  = np.array([np.std(hue_acc_hc[i], ddof=1) / np.sqrt(len(SUBS_HC)) for i in range(8)])
hue_acc_08_arr  = np.array([hue_acc_08[i] for i in range(8)])
hue_acc_09_arr  = np.array([hue_acc_09[i] for i in range(8)])

loro_acc = {}
for sub in SUBS_HC + SUBS_CVD:
    with open(f"{LORO_DIR}/sub-{sub}_performance_raw.json") as f:
        d = json.load(f)
    space = d["results"]["procrustes"]
    loro_acc[sub] = {}
    for roi in ROIS:
        folds = space[roi]["ForwardEncoding"]
        loro_acc[sub][roi] = np.mean([fold["acc_exact"] for fold in folds])

loco_hc_mean = np.array([np.mean([loco_acc[s][r] for s in SUBS_HC]) for r in ROIS])
loco_hc_sem  = np.array([np.std([loco_acc[s][r] for s in SUBS_HC], ddof=1)/np.sqrt(len(SUBS_HC)) for r in ROIS])
loco_hc_ind  = {r: [loco_acc[s][r] for s in SUBS_HC] for r in ROIS}

loro_hc_mean = np.array([np.mean([loro_acc[s][r] for s in SUBS_HC]) for r in ROIS])
loro_hc_sem  = np.array([np.std([loro_acc[s][r] for s in SUBS_HC], ddof=1)/np.sqrt(len(SUBS_HC)) for r in ROIS])
loro_hc_ind  = {r: [loro_acc[s][r] for s in SUBS_HC] for r in ROIS}

def crawford_howell(x_i, hc_vals, tail="lower"):
    """Crawford & Howell (1998) t-test for single case vs control sample.

    tail: "lower" (CVD < HC, deficit), "upper" (CVD > HC), or "two" (preserved).
    """
    n = len(hc_vals)
    mean_hc = np.mean(hc_vals)
    sd_hc = np.std(hc_vals, ddof=1)
    t = (x_i - mean_hc) / (sd_hc * np.sqrt((n + 1) / n))
    if tail == "lower":
        p = stats.t.cdf(t, df=n - 1)
    elif tail == "upper":
        p = stats.t.sf(t, df=n - 1)
    else:
        p = 2 * stats.t.sf(abs(t), df=n - 1)
    return t, p

# Panel A (LORO accuracy) — two-tailed; the hypothesis here is preservation
ch_loro_per_roi = {}
for r in ROIS:
    ch_loro_per_roi[r] = {
        "08": crawford_howell(loro_acc["08"][r], loro_hc_ind[r], tail="two"),
        "09": crawford_howell(loro_acc["09"][r], loro_hc_ind[r], tail="two"),
    }
ch_loro_08 = ch_loro_per_roi["V4"]["08"]
ch_loro_09 = ch_loro_per_roi["V4"]["09"]

# Panel B (LOCO adjacent acc) — one-sided lower per ROI (deficit = lower acc)
ch_loco_per_roi = {}
for r in ROIS:
    ch_loco_per_roi[r] = {
        "08": crawford_howell(loco_acc["08"][r], loco_hc_ind[r], tail="lower"),
        "09": crawford_howell(loco_acc["09"][r], loco_hc_ind[r], tail="lower"),
    }
ch_08 = ch_loco_per_roi["V4"]["08"]
ch_09 = ch_loco_per_roi["V4"]["09"]

# Panel C — per-hue Crawford & Howell at hV4 (each hue tested independently)
ch_per_hue_08 = {i: crawford_howell(hue_acc_08[i], hue_acc_hc[i], tail="lower")
                 for i in range(8)}
ch_per_hue_09 = {i: crawford_howell(hue_acc_09[i], hue_acc_hc[i], tail="lower")
                 for i in range(8)}

# ── Figure layout ─────────────────────────────────────────────────────────────
mm = 1/25.4
fig = plt.figure(figsize=(180*mm, 105*mm))

gs = fig.add_gridspec(1, 3,
                      left=0.09, right=0.97,
                      top=0.76, bottom=0.18,
                      wspace=0.42,
                      width_ratios=[1, 1, 1.1])

ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

x  = np.arange(len(ROIS))
bw = 0.55

def strip_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _sig_label(p):
    """Return (text, fontsize, weight). Empty text for n.s. — keeps the plot clean."""
    if p < 0.001: return ("***", 8.5, "bold")
    if p < 0.01:  return ("**",  8.5, "bold")
    if p < 0.05:  return ("*",   8.5, "bold")
    return ("", 0, "normal")

def plot_bars(ax, hc_mean, hc_sem, hc_ind, v08, v09,
              ylabel, chance, y_lo, y_hi,
              per_roi_p08=None, per_roi_p09=None, sig_rois=None):
    ax.bar(x, hc_mean, width=bw, color=HC_COLOR, zorder=2, linewidth=0)
    ax.errorbar(x, hc_mean, yerr=hc_sem, fmt="none",
                color="#333333", capsize=2.5, linewidth=0.9, zorder=3)
    np.random.seed(42)
    for i, roi in enumerate(ROIS):
        jit = np.random.uniform(-0.12, 0.12, len(hc_ind[roi]))
        ax.scatter(i + jit, hc_ind[roi], s=8, color=HC_DOT,
                   alpha=0.65, zorder=4, linewidths=0)
    for i in range(len(ROIS)):
        ax.plot(x[i], v08[i], marker="s", color=S08_COLOR,
                markersize=6.5, zorder=5, linewidth=0,
                markeredgecolor="white", markeredgewidth=0.4)
        ax.plot(x[i], v09[i], marker="^", color=S09_COLOR,
                markersize=6.5, zorder=5, linewidth=0,
                markeredgecolor="white", markeredgewidth=0.4)
    # Chance line only (no text label)
    ax.axhline(chance, color=CHANCE_COL, linestyle="--",
               linewidth=0.8, zorder=1, alpha=0.75)
    # Per-ROI per-subject Crawford & Howell sig labels — attached directly above
    # each CVD marker (sig only; n.s. omitted)
    if per_roi_p08 is not None and per_roi_p09 is not None:
        y_pad = (y_hi - y_lo) * 0.028
        for i, roi in enumerate(ROIS):
            if sig_rois is not None and roi not in sig_rois:
                continue
            lbl08, fs08, fw08 = _sig_label(per_roi_p08[roi])
            lbl09, fs09, fw09 = _sig_label(per_roi_p09[roi])
            if lbl08:
                ax.text(x[i], v08[i] + y_pad, lbl08,
                        fontsize=fs08, color=S08_COLOR, fontweight=fw08,
                        ha="center", va="bottom")
            if lbl09:
                ax.text(x[i], v09[i] + y_pad, lbl09,
                        fontsize=fs09, color=S09_COLOR, fontweight=fw09,
                        ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(ROI_LABELS, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7, labelpad=3)
    ax.tick_params(axis="both", labelsize=6.5, length=3)
    ax.set_ylim(y_lo, y_hi)
    strip_spines(ax)

# ── Panel A: LORO (leave-one-run-out, discrimination) ────────────────────────
loro_08 = np.array([loro_acc["08"][r] for r in ROIS])
loro_09 = np.array([loro_acc["09"][r] for r in ROIS])

plot_bars(ax_a,
          hc_mean=loro_hc_mean, hc_sem=loro_hc_sem, hc_ind=loro_hc_ind,
          v08=loro_08, v09=loro_09,
          ylabel="Proportion correct",
          chance=CHANCE_EXACT,
          y_lo=0.0, y_hi=1.12,
          per_roi_p08={r: ch_loro_per_roi[r]["08"][1] for r in ROIS},
          per_roi_p09={r: ch_loro_per_roi[r]["09"][1] for r in ROIS})

ax_a.text(0.0, 1.20, "A", transform=ax_a.transAxes,
          fontsize=10, fontweight="bold", va="bottom", ha="left")
ax_a.text(0.105, 1.21, "Discrimination", transform=ax_a.transAxes,
          fontsize=7.5, va="bottom", ha="left", color="#222", fontweight="bold")
ax_a.text(0.105, 1.10, "LORO (leave-one-run-out)", transform=ax_a.transAxes,
          fontsize=6.5, va="bottom", ha="left", color="#555")

# ── Panel B: LOCO (leave-one-color-out, interpolation) ───────────────────────
loco_08 = np.array([loco_acc["08"][r] for r in ROIS])
loco_09 = np.array([loco_acc["09"][r] for r in ROIS])

plot_bars(ax_b,
          hc_mean=loco_hc_mean, hc_sem=loco_hc_sem, hc_ind=loco_hc_ind,
          v08=loco_08, v09=loco_09,
          ylabel="Adjacent accuracy",
          chance=CHANCE_ADJ,
          y_lo=0.0, y_hi=0.85,
          per_roi_p08={r: ch_loco_per_roi[r]["08"][1] for r in ROIS},
          per_roi_p09={r: ch_loco_per_roi[r]["09"][1] for r in ROIS},
          sig_rois=PERM_PASS_ROIS)

ax_b.text(0.0, 1.20, "B", transform=ax_b.transAxes,
          fontsize=10, fontweight="bold", va="bottom", ha="left")
ax_b.text(0.105, 1.21, "Interpolation", transform=ax_b.transAxes,
          fontsize=7.5, va="bottom", ha="left", color="#222", fontweight="bold")
ax_b.text(0.105, 1.10, "LOCO (leave-one-color-out)", transform=ax_b.transAxes,
          fontsize=6.5, va="bottom", ha="left", color="#555")

# ── Panel C: Per-hue at hV4 (LOCO, same encoding as A/B) ─────────────────────
hue_x = np.arange(8)

# HC = bar (per-hue STIM_LAB color), CVD = markers — matches Panels A/B encoding
ax_c.bar(hue_x, hue_acc_hc_mean, width=0.55,
         color=HUE_RGB, linewidth=0.4, edgecolor="#444", zorder=2)
ax_c.errorbar(hue_x, hue_acc_hc_mean, yerr=hue_acc_hc_sem,
              fmt="none", color="#333333", capsize=1.8, linewidth=0.8, zorder=3)
ax_c.plot(hue_x, hue_acc_08_arr,
          marker="s", color=S08_COLOR, markersize=5.5,
          linestyle="-", linewidth=0.7,
          markeredgecolor="white", markeredgewidth=0.4, zorder=5)
ax_c.plot(hue_x, hue_acc_09_arr,
          marker="^", color=S09_COLOR, markersize=5.5,
          linestyle="-", linewidth=0.7,
          markeredgecolor="white", markeredgewidth=0.4, zorder=5)

ax_c.axhline(CHANCE_ADJ, color=CHANCE_COL, linestyle="--",
             linewidth=0.8, zorder=1, alpha=0.75)

# Per-hue per-subject sig labels attached directly above each CVD marker
y_pad_c = 0.04
for i in range(8):
    lbl08, fs08, fw08 = _sig_label(ch_per_hue_08[i][1])
    lbl09, fs09, fw09 = _sig_label(ch_per_hue_09[i][1])
    if lbl08:
        ax_c.text(i, hue_acc_08_arr[i] + y_pad_c, lbl08,
                  fontsize=fs08, color=S08_COLOR, fontweight=fw08,
                  ha="center", va="bottom")
    if lbl09:
        ax_c.text(i, hue_acc_09_arr[i] + y_pad_c, lbl09,
                  fontsize=fs09, color=S09_COLOR, fontweight=fw09,
                  ha="center", va="bottom")

ax_c.set_xticks(hue_x)
ax_c.set_xticklabels(HUE_NAMES, rotation=35, ha="right", fontsize=6.5)
ax_c.set_ylabel("Adjacent accuracy", fontsize=7, labelpad=3)
ax_c.set_ylim(0, 1.0)
ax_c.set_xlim(-0.6, 7.6)
ax_c.tick_params(axis="both", labelsize=6.5, length=3)
strip_spines(ax_c)
ax_c.text(0.0, 1.20, "C", transform=ax_c.transAxes,
          fontsize=10, fontweight="bold", va="bottom", ha="left")
ax_c.text(0.105, 1.21, "Per-hue interpolation at hV4",
          transform=ax_c.transAxes,
          fontsize=7.5, va="bottom", ha="left", color="#222", fontweight="bold")
ax_c.text(0.105, 1.10, "LOCO (leave-one-color-out)",
          transform=ax_c.transAxes,
          fontsize=6.5, va="bottom", ha="left", color="#555")

# ── Shared legend ─────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=HC_COLOR, label="Control mean ± SEM", edgecolor="none"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=HC_DOT,
           markersize=5, label="Individual controls", linewidth=0),
    Line2D([0], [0], marker="s", color="w", markerfacecolor=S08_COLOR,
           markersize=6, label="Deutan", linewidth=0),
    Line2D([0], [0], marker="^", color="w", markerfacecolor=S09_COLOR,
           markersize=6, label="Protan", linewidth=0),
]
fig.legend(handles=legend_handles, loc="lower center",
           ncol=4, fontsize=6.5, frameon=False,
           bbox_to_anchor=(0.5, 0.01))

# ── Save ──────────────────────────────────────────────────────────────────────
out_png = os.path.join(OUT_DIR, "fig2_loro_loco.png")
out_pdf = os.path.join(OUT_DIR, "fig2_loro_loco.pdf")
fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_png}")

print("\nCrawford & Howell — hV4 (parametric t, df=6); test: CVD performance < controls")
print("  LORO accuracy (one-sided lower, deficit = lower accuracy):")
print(f"    sub-08: t={ch_loro_08[0]:.3f}, p={ch_loro_08[1]:.4f}")
print(f"    sub-09: t={ch_loro_09[0]:.3f}, p={ch_loro_09[1]:.4f}")
print("  LOCO adjacent acc (one-sided lower, deficit = lower acc):")
print(f"    sub-08: t={ch_08[0]:.3f}, p={ch_08[1]:.4f}")
print(f"    sub-09: t={ch_09[0]:.3f}, p={ch_09[1]:.4f}")

print("\nPer-hue C&H at hV4 (LOCO, one-sided lower):")
print(f"  {'Hue':<7} {'sub-08 t':<10} {'p':<8} {'sub-09 t':<10} {'p':<8}")
for i, name in enumerate(HUE_NAMES):
    t08, p08 = ch_per_hue_08[i]
    t09, p09 = ch_per_hue_09[i]
    sig08 = _sig_label(p08)[0] or "-"
    sig09 = _sig_label(p09)[0] or "-"
    print(f"  {name:<7} {t08:>+8.2f}  {p08:.4f} {sig08:<3} "
          f"{t09:>+8.2f}  {p09:.4f} {sig09}")
