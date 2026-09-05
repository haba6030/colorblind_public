#!/usr/bin/env python3
"""
Supplementary S1 — Overall cortical color signal is PRESERVED in CVD.

Backs the abstract claim: the cortical color representation in CVD is
"not reduced in overall signal but distorted in structure."

This is a CONSOLIDATION + VISUALIZATION of the pre-existing pre-SRM activation
analysis (analysis/phase2_SRM_across_between/activation_prior_analysis.py,
2026-03-27). No recomputation: it reads the frozen result JSON and renders a
paper-ready figure + a compact summary table. The numbers are byte-identical
to the source analysis; this folder only makes them findable and citable.

Univariate metrics compared HC (n=7) vs CVD (n=3) per ROI (V1, V2, V3, hV4):
  mean_abs_act   - signal magnitude (mean |beta|)   [higher = stronger signal]
  snr_median     - median voxel SNR across runs      [higher = cleaner signal]
  modulation_depth - color tuning range (max-min)    [higher = stronger tuning]
  run_reliability  - run-to-run profile correlation  [higher = more reliable]

Group test = Welch's t; per-CVD-subject = Crawford & Howell (1998) single-case.

Run (local): conda activate srm; python plot_overall_signal.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase_supplementary/scripts/plot_overall_signal.py  (development repository, commit 53c81c2)
# Role      : Consolidates the activation metrics with single-case tests -> overall_signal_results.json (S10).
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
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIGURES = HERE.parent / "figures"
# Self-contained local copy; fall back to the canonical source if absent.
JSON_LOCAL = RESULTS / "overall_signal_results.json"
JSON_SRC = (HERE.parent.parent / "phase2_SRM_across_between" / "results"
            / "activation_prior" / "activation_prior_results.json")

ROIS = ["V1", "V2", "V3", "hV4"]
HC_SUBJS = [f"sub-0{i}" for i in range(1, 8)]          # sub-01..07
CVD = {"sub-08": ("deutan", "#d95f02", "s"),
       "sub-09": ("protan", "#1b9e77", "^"),
       "sub-10": ("deutan", "#7570b3", "D")}

METRICS = [
    ("mean_abs_act",    "Signal magnitude  (mean |β|)"),
    ("snr_median",      "SNR  (median)"),
    ("modulation_depth", "Color modulation depth"),
    ("run_reliability", "Run-to-run reliability"),
]


def load():
    path = JSON_LOCAL if JSON_LOCAL.exists() else JSON_SRC
    with open(path) as f:
        return json.load(f), path


def fmt_p(p):
    if p < 0.001:
        return "p<.001"
    return f"p={p:.2f}".replace("0.", ".")


def main():
    data, path = load()
    print(f"Loaded: {path}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    x = np.arange(len(ROIS))

    for ax, (mkey, mlabel) in zip(axes.ravel(), METRICS):
        hc_means, hc_sds = [], []
        for roi in ROIS:
            gt = data[roi]["group_tests"][mkey]
            hc_means.append(gt["hc_mean"])
            hc_sds.append(gt["hc_sd"])
        hc_means = np.array(hc_means)
        hc_sds = np.array(hc_sds)

        # HC group band: mean +/- SD bar
        ax.bar(x, hc_means, yerr=hc_sds, width=0.62, color="0.82",
               edgecolor="0.4", capsize=4, zorder=1, label="HC (n=7)")

        # individual HC dots
        for roi_i, roi in enumerate(ROIS):
            ps = data[roi]["per_subject"]
            hc_vals = [ps[s][mkey] for s in HC_SUBJS if s in ps]
            jit = (np.random.RandomState(roi_i).rand(len(hc_vals)) - 0.5) * 0.22
            ax.scatter(np.full(len(hc_vals), roi_i) + jit, hc_vals,
                       s=16, color="0.45", zorder=2, alpha=0.8)

        # CVD individual markers
        for sub, (ctype, col, mk) in CVD.items():
            vals = [data[roi]["per_subject"][sub][mkey] for roi in ROIS]
            ax.scatter(x, vals, s=70, color=col, marker=mk, zorder=4,
                       edgecolor="black", linewidth=0.6,
                       label=f"{sub} ({ctype})")

        # Welch p per ROI, annotated at top
        ymax = max(hc_means + hc_sds)
        for roi_i, roi in enumerate(ROIS):
            p = data[roi]["group_tests"][mkey]["p"]
            ax.text(roi_i, ymax * 1.18, fmt_p(p), ha="center", va="bottom",
                    fontsize=8, color="0.25")

        ax.set_xticks(x)
        ax.set_xticklabels(ROIS)
        ax.set_title(mlabel, fontsize=11)
        ax.set_ylim(top=ymax * 1.32)
        if mkey == "run_reliability":
            ax.axhline(0, color="0.6", lw=0.8, ls="--", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    # shared legend
    handles = [plt.Rectangle((0, 0), 1, 1, color="0.82", ec="0.4")]
    labels = ["HC (n=7, mean±SD)"]
    for sub, (ctype, col, mk) in CVD.items():
        handles.append(Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                              markeredgecolor="black", markersize=9))
        labels.append(f"{sub} ({ctype})")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Overall cortical color signal is preserved in CVD "
                 "(Welch HC vs CVD, all p>.09)", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    for ext in ("png", "pdf"):
        out = FIGURES / f"overall_signal_preserved.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"Saved: {out}")

    # ---- compact summary table to stdout + markdown ----
    lines = ["| Metric | ROI | HC mean±SD | CVD mean±SD | Welch p | d |",
             "|---|---|---|---|---|---|"]
    for mkey, mlabel in METRICS:
        for roi in ROIS:
            gt = data[roi]["group_tests"][mkey]
            lines.append(
                f"| {mlabel} | {roi} | {gt['hc_mean']:.4g}±{gt['hc_sd']:.2g} "
                f"| {gt['cvd_mean']:.4g}±{gt['cvd_sd']:.2g} "
                f"| {gt['p']:.3f} | {gt['d']:+.2f} |")
    table = "\n".join(lines)
    print("\n" + table)
    (RESULTS / "overall_signal_summary_table.md").write_text(table + "\n")
    print(f"\nSaved: {RESULTS / 'overall_signal_summary_table.md'}")


if __name__ == "__main__":
    main()
