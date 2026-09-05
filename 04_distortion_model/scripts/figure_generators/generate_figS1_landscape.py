"""Supplementary Figure (S18) — Per-subject loss landscape over (beta_s, beta_c) with argmin cloud.

Publication version (2026-06-05). Two CVD subjects side-by-side, single clean
composite panel each (no per-atom sub-panels, no internal jargon). Reuses the
canonical composite reconstruction + 300-resample argmin cloud from
scripts/viz_closure_ground_plot.py so the surface is identical to closure.

Canonical fits (PIPELINE_2_CLOSURE.md 2026-06-01):
  sub-08 deutan  (beta_s=+6,  beta_c=-42)   gamma_OY  + RDM_V2
  sub-09 protan  (beta_s=+2,  beta_c=+24)   gamma_all + RDM_V1

Output: docs/PAPER/Figures/figS1_landscape.{pdf,png}
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : docs/PAPER/Figures/scripts/phase2/generate_figS1_landscape.py  (development repository, commit 53c81c2)
# Role      : Figure S1 (loss landscape with resample argmins).
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


import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_P2_SCRIPTS = (Path(__file__).resolve().parents[5]
               / "analysis" / "phase5_filter_optimization" / "scripts")
sys.path.insert(0, str(_P2_SCRIPTS))
from two_comp import BS_GRID, BC_GRID                                   # noqa: E402
from viz_closure_ground_plot import (                                   # noqa: E402
    build_composite_full_hc, load_resample_argmins,
)

OUT = Path(__file__).resolve().parents[2]

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],  # Arial first: IN requires Arial or Helvetica; kept uniform across all figures
    "font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
})

CANDIDATES = [
    dict(id="S08_bc_dom", subject="sub-08", family="deutan",
         combo_key="γOY|RDMV2|noLOCO", gamma_atoms=["OY"], rdm_rois=["V2"],
         fit_point=(6.0, -42.0), title="Deutan"),
    dict(id="S09_bc_rot", subject="sub-09", family="protan",
         combo_key="γALL|RDMV1|noLOCO", gamma_atoms=["ALL"], rdm_rois=["V1"],
         fit_point=(2.0, 24.0), title="Protan"),
]


def plot_panel(ax, comp, fit_pt, bs_s, bc_s, title):
    BC, BS = np.meshgrid(BC_GRID, BS_GRID)
    # vmin = true global min so the argmin (star) is the unique brightest cell;
    # clip only the high tail to keep outliers from washing out the gradient.
    vmin, vmax = np.nanmin(comp), np.nanpercentile(comp, 99)
    im = ax.pcolormesh(BC, BS, comp, cmap="viridis_r", shading="auto",
                       norm=Normalize(vmin=vmin, vmax=vmax), rasterized=True)
    ax.contour(BC, BS, comp, levels=np.nanpercentile(comp, [5, 15, 30, 50, 70]),
               colors="white", alpha=0.35, linewidths=0.6)

    # resample argmin cloud (bootstrap uncertainty)
    if len(bs_s) > 0:
        for (bv, cv), n in Counter(zip(bs_s, bc_s)).items():
            ax.scatter(cv, bv, s=8 + 2 * n, c="white", edgecolors="black",
                       linewidth=0.3, alpha=0.5, zorder=3)
    # selected fit (argmin)
    fb, fc = fit_pt
    ax.scatter(fc, fb, marker="*", s=320, c="#d62728", edgecolors="white",
               linewidth=1.1, zorder=5)
    ax.annotate(f"({fb:+.0f}°, {fc:+.0f}°)", (fc, fb),
                textcoords="offset points", xytext=(8, 8), fontsize=8.5,
                color="white", fontweight="bold", zorder=6)

    ax.axhline(0, color="0.7", lw=0.5, ls=":"); ax.axvline(0, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel(r"$\beta_c$  (confusion-axis rotation, °)")
    ax.set_ylabel(r"$\beta_s$  (S-cone-axis rotation, °)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(BC_GRID[0], BC_GRID[-1]); ax.set_ylim(BS_GRID[0], BS_GRID[-1])
    return im


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.5), constrained_layout=True)
    im = None
    for ax, cand in zip(axes, CANDIDATES):
        print(f"building {cand['id']} ...", flush=True)
        comp, *_ = build_composite_full_hc(cand)
        bs_s, bc_s = load_resample_argmins(cand)
        im = plot_panel(ax, comp, cand["fit_point"], bs_s, bc_s, cand["title"])

    cb = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.015, aspect=30,
                      location="right")
    cb.set_label("composite loss (z-scored; lower = better fit)")

    # shared legend for the two markers
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#d62728",
               markeredgecolor="white", markersize=15, label="selected fit (argmin)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="black", markersize=8,
               label="Control-resample argmins (N=300)"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8,
                   framealpha=0.9, handletextpad=0.4)

    for ext in ("pdf", "png"):
        p = OUT / f"figS1_landscape.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print("saved:", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
