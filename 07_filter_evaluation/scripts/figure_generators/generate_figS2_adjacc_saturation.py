#!/usr/bin/env python3
"""
figS2_adjacc_saturation — LOCO adjacent-accuracy run-count saturation (paper primary metric).

Reads run_count_validation/adjacc_retention_summary.json (produced by
analysis/phase6_behavioral_analysis/scripts/run_count_adjacc.py) and
plots adjacent accuracy vs run count per ROI: HC mean ± SEM band, 91/360 chance
line, and the two CVD single cases (sub-08 deutan, sub-09 protan).
Demonstrates the hV4 landmark retains at n=4.

Output → docs/PAPER/Figures/figS2_adjacc_saturation.{png,pdf}
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : docs/PAPER/Figures/scripts/generate_figS2_adjacc_saturation.py  (development repository, commit 53c81c2)
# Role      : Figure S2 (adjacent accuracy vs run count).
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

BASE = Path(__file__).resolve().parents[4]  # repo root (docs/PAPER/Figures/scripts -> repo)
SUMM = BASE / "analysis/phase6_behavioral_analysis/run_count_validation/adjacc_retention_summary.json"
OUTDIR = BASE / "docs/PAPER/Figures"

ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABELS = ["V1", "V2", "V3", "hV4"]
NVALS = [2, 3, 4, 5, 6]
# Adjacent-accuracy chance. The forward-encoding readout takes an argmax over all
# 360 integer hues (utils_forward_model.decode_hue), and adjacent accuracy counts a
# prediction correct when its circular error is <= 45 deg (loco_canonical.py). A
# prediction drawn uniformly from that 360-hue output space lands inside the
# tolerance on 91 of 360 draws. Verified by simulation (0.253 over 20,000 draws).
# NOT 3/8 -- that holds only for decoders that output one of the eight stimulus hues.
CHANCE = 91 / 360

HC_LINE = "#444444"
HC_BAND = "#CCCCCC"
C_DEUT = "#D55E00"   # sub-08 deutan (orange)
C_PROT = "#029E73"   # sub-09 protan (teal)
C_S10 = "#BBBBBB"    # sub-10 deutan control (faint)
CVD = {"08": (C_DEUT, "s", "Deutan"),
       "09": (C_PROT, "^", "Protan")}


def strip(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main():
    s = json.load(open(SUMM))["per_roi"]
    mm = 1 / 25.4
    fig, axes = plt.subplots(1, 4, figsize=(180 * mm, 52 * mm), sharey=True)
    for j, (roi, lab) in enumerate(zip(ROIS, ROI_LABELS)):
        ax = axes[j]
        d = s[roi]
        hc_m = np.array([d[str(n)]["hc_mean"] for n in NVALS])
        hc_e = np.array([d[str(n)]["hc_sem"] for n in NVALS])
        ax.fill_between(NVALS, hc_m - hc_e, hc_m + hc_e, color=HC_BAND, zorder=1)
        ax.plot(NVALS, hc_m, color=HC_LINE, lw=1.4, zorder=3, label="Control mean ± SEM")
        ax.axhline(CHANCE, color="#999", ls="--", lw=0.8, zorder=2)
        for subj, (col, mk, _) in CVD.items():
            yv = np.array([d[str(n)]["cvd"][subj]["adjacc"] for n in NVALS])
            faint = subj == "sub-10"
            ax.plot(NVALS, yv, color=col, lw=1.0 if not faint else 0.8,
                    ls="-" if not faint else ":", marker=mk, markersize=4,
                    markeredgecolor="white", markeredgewidth=0.4,
                    alpha=1.0 if not faint else 0.7, zorder=4)
        ax.axvline(4, color="#D55E00", lw=0.6, ls=":", alpha=0.4, zorder=0)
        ax.set_xticks(NVALS)
        ax.set_xlabel("runs ($n$)", fontsize=7)
        ax.set_title(lab, fontsize=8, fontweight="bold")
        ax.tick_params(labelsize=6.5, length=3)
        if j == 0:
            ax.set_ylabel("LOCO adjacent acc.", fontsize=7)
        ax.text(2.05, CHANCE + 0.008, "chance 0.25", fontsize=5.5, color="#999", va="bottom")
        strip(ax)
    axes[0].set_ylim(0.05, 0.62)

    handles = [Line2D([0], [0], color=HC_LINE, lw=1.4, label="Control mean ± SEM")] + \
        [Line2D([0], [0], color=c, marker=m, lw=1.0, markersize=4, label=l)
         for c, m, l in CVD.values()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6.3,
               frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.subplots_adjust(left=0.07, right=0.99, top=0.86, bottom=0.30, wspace=0.12)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"figS2_adjacc_saturation.{ext}", dpi=300, bbox_inches="tight")
    print("Saved:", OUTDIR / "figS2_adjacc_saturation.png")


if __name__ == "__main__":
    main()
