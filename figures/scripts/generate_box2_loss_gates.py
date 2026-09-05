#!/usr/bin/env python3
"""
generate_box2_loss_gates.py

Replacement content for box 2 ("Candidate loss atoms") of the pipeline
schematic (`fig:pipeline`, file stem `fig3_workflow`).

WHY THIS REPLACES THE OLD BOX 2 (2026-09-03, author instruction)
----------------------------------------------------------------
The previous box 2 was a block of bullet text that also announced the OUTCOME
of the selection procedure: the per-participant winning combinations
(deutan gamma_OY + L_RDM(V2); protan gamma_all + L_RDM(V1)) and a struck-out
"not selected" tag on the LOCO atom.  Those are results -- they are reported in
Results (results_v4.tex, the two-component subsection) -- and a Methods
schematic should show the PROCEDURE only.  The project figure-caption rule in
CLAUDE.md ("captions state the measured quantity, method, symbols and test
direction, not the result") points the same way.

So this asset shows only:
  * the three candidate loss atoms, as icons in the visual language of
    Figure 1 panel C (same STIM_LAB swatch colors, same thin-stroke style), and
  * the three selection gates, given the visual weight of the panel.
No inner boxes, no winning combination, no "not selected".

The L_RDM and L_LOCO icons are the Figure 1 panel-C glyphs.  L_gamma has no
Figure 1 counterpart (Figure 1 stage B is neural only), so it is drawn in the
same language as a threshold-ratio pair.

WHY STANDALONE.  `fig3_workflow` is the one manuscript figure that is NOT a
single script: it is a PowerPoint composite (`fig3_assets/Presentation1*.pptx`)
into which pre-rendered box assets are placed.  This script produces such an
asset; `patch_fig3_box2.py` inserts it into the .pptx.

Geometry: the box-2 interior of the slide measures 2.47 x 4.22 in, so the
figure is created at that size and the point sizes below are slide points.

Output: docs/PAPER/Figures/fig3_assets/box2_loss_gates.{png,pdf}
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : figures
# Manuscript: shared module
# Original  : docs/PAPER/Figures/scripts/generate_box2_loss_gates.py  (development repository, commit 53c81c2)
# Role      : figure generator or asset
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[2]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_fig1_v3 import MEASURED_LAB, lab_to_srgb, HUES  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "fig3_assets"

W_IN, H_IN = 2.51, 4.26          # box-2 interior of the slide
GREEN = "#006033"                # box-2 header fill, sampled from the composite
INK = "0.15"
SUB = "0.35"

RGBS = [lab_to_srgb(L, a, b) for L, a, b in MEASURED_LAB]


def _square_inset(ax, x0, y0, w):
    """Square inset in axes fractions, compensating the panel aspect."""
    sub = ax.inset_axes([x0, y0, w, w * (W_IN / H_IN)], transform=ax.transAxes)
    sub.set_xlim(-1.6, 1.6)
    sub.set_ylim(-1.6, 1.6)
    sub.set_aspect("equal")
    sub.axis("off")
    return sub


def icon_gamma(sub):
    """Behavioral JND ratio: the CVD discrimination threshold against control.
    The dashed line marks the control level, so the excess reads as elevation."""
    base = -1.05
    for x, h, fc in ((-0.60, 0.95, "0.86"), (0.60, 1.75, RGBS[1])):
        sub.add_patch(plt.Rectangle((x - 0.34, base), 0.68, h,
                                    fc=fc, ec="0.35", lw=0.6, zorder=3))
    sub.plot([-1.05, 1.05], [base + 0.95, base + 0.95], color="0.35",
             lw=0.7, linestyle="--", zorder=4)


def icon_rdm(sub):
    """Representational distances: a similar pair sits close, a distant pair far.
    Same glyph as Figure 1 panel C, stage B."""
    def pair(y, i, j, x1, x2):
        sub.scatter([x1, x2], [y, y], s=26, c=[RGBS[i], RGBS[j]],
                    edgecolors="0.3", linewidths=0.4, zorder=5)
        sub.add_patch(FancyArrowPatch((x1 + 0.20, y), (x2 - 0.20, y),
                                      arrowstyle="<|-|>", mutation_scale=4,
                                      lw=0.8, color="0.3"))
    pair(0.60, 0, 1, -0.72, 0.72)
    pair(-0.62, 0, 4, -1.20, 1.20)


def icon_loco(sub):
    """One hue held out and decoded. Same glyph as Figure 1 panel C, stage B."""
    r = 1.08
    sub.add_patch(Circle((0, 0), r, fill=False, ec="0.85", lw=0.6))
    for h, rgb in zip(HUES, RGBS):
        a = np.radians(h)
        if h == 225:
            sub.scatter(r * np.cos(a), r * np.sin(a), s=26, facecolors="none",
                        edgecolors="0.35", linewidths=0.7, linestyle="--")
            continue
        sub.scatter(r * np.cos(a), r * np.sin(a), s=22, c=[rgb],
                    edgecolors="0.3", linewidths=0.35)
    aa = np.radians(235.0)
    sub.add_patch(FancyArrowPatch((0, 0), (0.80 * np.cos(aa), 0.80 * np.sin(aa)),
                                  arrowstyle="-|>", mutation_scale=5,
                                  lw=1.0, color="0.2"))


def main():
    fig = plt.figure(figsize=(W_IN, H_IN))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ── candidate loss atoms ────────────────────────────────────────────────
    atoms = [
        (icon_gamma, r"$L_\gamma$",     "JND ratio"),
        (icon_rdm,   r"$L_{\rm RDM}$",  r"$\Delta$RDM direction"),
        (icon_loco,  r"$L_{\rm LOCO}$", "hV4 LOCO"),
    ]
    # icon | symbol | description on one line, columns aligned across rows and
    # the whole block centred in the panel
    icon_w = 0.185                                  # axes fractions
    icon_h = icon_w * (W_IN / H_IN)                 # square on the page
    row_y = [0.940, 0.820, 0.700]                   # row centers, axes fractions
    gap_a, gap_b = 4.0, 5.0                         # data units between columns

    def _w(txt, size):
        t = ax.text(0, -50, txt, fontsize=size)     # measured off-canvas
        fig.canvas.draw()
        bb = t.get_window_extent().transformed(ax.transData.inverted())
        t.remove()
        return bb.width

    sym_col = max(_w(sym, 12.5) for _, sym, _ in atoms)
    desc_col = max(_w(desc, 9.0) for _, _, desc in atoms)
    block = icon_w * 100 + gap_a + sym_col + gap_b + desc_col
    x_icon = (100 - block) / 2
    x_sym = x_icon + icon_w * 100 + gap_a
    x_desc = x_sym + sym_col + gap_b

    for (draw, sym, desc), yc in zip(atoms, row_y):
        draw(_square_inset(ax, x_icon / 100, yc - icon_h / 2, icon_w))
        ycen = yc * 100
        ax.text(x_sym, ycen, sym, fontsize=12.5, color=INK,
                ha="left", va="center")
        ax.text(x_desc, ycen, desc, fontsize=9.0, color=SUB,
                ha="left", va="center")

    # ── the three atoms converge into the gate stack ────────────────────────
    ax.plot([12, 12, 88, 88], [62.5, 59.0, 59.0, 62.5], color="0.55", lw=0.8,
            solid_capstyle="round", solid_joinstyle="round")
    ax.add_patch(FancyArrowPatch((50, 59.0), (50, 54.5), arrowstyle="-|>",
                                 mutation_scale=8, lw=1.1, color="0.45"))

    # ── three gates, given the visual weight of the panel ───────────────────
    # gate names follow \S sec:methods:selection verbatim
    gates = [
        "1 \u00b7 Separation",
        "2 \u00b7 Boundary saturation",
        "3 \u00b7 Held-out test loss",
    ]
    widths = [94, 82, 70]
    y = 51.0
    for name, w in zip(gates, widths):
        h = 12.5
        ax.add_patch(FancyBboxPatch((50 - w / 2, y - h), w, h,
                                    boxstyle="round,pad=0,rounding_size=2.0",
                                    fc=GREEN, ec="none", zorder=3))
        ax.text(50, y - h / 2, name, fontsize=10.0, fontweight="bold",
                color="white", ha="center", va="center", zorder=4)
        y -= h
        if name is not gates[-1]:
            ax.add_patch(FancyArrowPatch((50, y - 0.5), (50, y - 4.0),
                                         arrowstyle="-|>", mutation_scale=8,
                                         lw=1.1, color="0.45"))
            y -= 4.5

    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT / f"box2_loss_gates.{ext}"
        fig.savefig(p, dpi=600, facecolor="white")
        print("saved:", p)
    plt.close(fig)


if __name__ == "__main__":
    main()
