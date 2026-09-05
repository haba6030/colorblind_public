#!/usr/bin/env python3
"""
generate_fig1_v3.py — Figure 1 (fig:paradigm) for the Imaging Neuroscience submission.

WHY THIS SCRIPT EXISTS
----------------------
The previous Figure 1 (`fig1_generated_v2.pdf`) was produced with an image-
generation AI and had no generator script, which caused two problems:

  1. Its text was raster/DejaVu, violating the journal requirement that figures
     use Arial or Helvetica.
  2. Its panel B was an AI-rendered brain. Synthetic anatomy in a neuroimaging
     paper invites a credibility question it cannot answer, and it forced the
     AI-tool disclosure to name a figure.

Decision (2026-08-18): drop the brain/ROI panel entirely -- the ROIs are
atlas-defined (Wang et al. 2015) and cited in Methods, so no anatomical panel is
required -- and rebuild the remaining three panels programmatically. Figure 1 is
therefore fully script-generated and contains no AI-produced content.

PANELS
------
  A  Stimulus set: 8 hues in the CIE L*a*b* a*b* plane.
  B  RSVP trial structure, composited from the actual screen captures.
  C  Analysis pipeline: Stage A alignment -> Stage B neural features
     -> Stage C modeling and filter inversion.

SWATCH COLORS (2026-09-02, author decision): all swatches are rendered from
the canonical screenshot-derived STIM_LAB values (`analysis/utils/
utils_color_decoding.py` COLOR_LAB == phase5 `stim_lab_render.STIM_LAB`,
duplicated below as MEASURED_LAB), i.e. the same color source the other
figures use (e.g. fig7_filter via stim_lab_render.render_at_hue). The earlier
version colored the swatches from the nominal L* = 75 / chroma = 40
specification and visibly disagreed with the panel-B screen captures and with
Figure 6. Panel-A GEOMETRY (dot positions, 45-deg labels) still shows the
nominal design specification; reconciling the Methods L*/chroma sentence with
the screenshot estimates remains the P0 manuscript-text item tracked in
SUBMISSION_CHECKLIST_IMAGING_NEURO.md (SHOW_MEASURED overlays those a*b*
positions when enabled).

Fonts: run with MATPLOTLIBRC pointed at Figures/scripts/inrc so Arial is used for
text and for mathtext. See Figures/scripts/FONT_POLICY.md.

    export MATPLOTLIBRC="<repo>/docs/PAPER/Figures/scripts/inrc"
    python3 generate_fig1_v3.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : figures
# Manuscript: shared module
# Original  : docs/PAPER/Figures/scripts/generate_fig1_v3.py  (development repository, commit 53c81c2)
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from PIL import Image

FIGDIR = Path(__file__).resolve().parent.parent
OUT_STEM = FIGDIR / "fig1_paradigm_v3"

# Panel-B square-capture aspect is computed at runtime from the laid-out axes
# (see panel_b), so gridspec/figsize changes cannot silently stretch the discs.

SHOW_MEASURED = False  # see the module docstring before enabling

# Nominal stimulus specification, as described in Methods.
L_NOMINAL = 75.0
C_NOMINAL = 40.0
HUES = np.arange(0, 360, 45)
HUE_NAMES = ["Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple", "Magenta"]

# Screenshot-derived estimates, for the optional second layer only.
MEASURED_LAB = np.array([
    [59.90, 62.69, 3.78], [64.20, 49.20, 45.58], [57.27, 13.06, 41.69],
    [69.08, -55.02, 47.38], [74.61, -41.33, -4.89], [69.14, -11.45, -40.91],
    [60.68, 19.18, -54.13], [60.17, 46.82, -40.31],
])


def lab_to_srgb(L, a, b):
    """CIELab (D65) -> sRGB in [0, 1], clipped. Self-contained on purpose so the
    released figure script has no cross-package dependency."""
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0

    def finv(t):
        return t ** 3 if t ** 3 > 0.008856 else (t - 16.0 / 116.0) / 7.787

    xn, yn, zn = 0.95047, 1.00000, 1.08883
    X, Y, Z = finv(fx) * xn, finv(fy) * yn, finv(fz) * zn
    r = 3.2406 * X - 1.5372 * Y - 0.4986 * Z
    g = -0.9689 * X + 1.8758 * Y + 0.0415 * Z
    bl = 0.0557 * X - 0.2040 * Y + 1.0570 * Z

    def gamma(c):
        c = max(0.0, min(1.0, c))
        return 1.055 * c ** (1 / 2.4) - 0.055 if c > 0.0031308 else 12.92 * c

    return (gamma(r), gamma(g), gamma(bl))


def stim_rgb():
    """Swatch colors: canonical STIM_LAB rendering (what was displayed),
    matching stim_lab_render-based figures. Geometry stays nominal."""
    return [lab_to_srgb(L, a, b) for L, a, b in MEASURED_LAB]


# ─── Panel A ──────────────────────────────────────────────────────────────────
def panel_a(ax):
    lim = 78
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    for r in (20, 40, 60):
        ax.add_patch(Circle((0, 0), r, fill=False, ec="0.85", lw=0.5, zorder=0))
    ax.axhline(0, color="0.85", lw=0.5, zorder=0)
    ax.axvline(0, color="0.85", lw=0.5, zorder=0)

    rgbs = stim_rgb()
    for h, name, rgb in zip(HUES, HUE_NAMES, rgbs):
        x = C_NOMINAL * np.cos(np.radians(h))
        y = C_NOMINAL * np.sin(np.radians(h))
        ax.add_patch(Circle((x, y), 7.2, fc=rgb, ec="0.35", lw=0.5, zorder=3))
        lx = 62 * np.cos(np.radians(h))
        ly = 62 * np.sin(np.radians(h))
        ax.text(lx, ly, f"{h}°", fontsize=6, ha="center", va="center",
                color="0.35", zorder=4,
                bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))

    if SHOW_MEASURED:
        ax.scatter(MEASURED_LAB[:, 1], MEASURED_LAB[:, 2], s=14, facecolors="none",
                   edgecolors="0.2", lw=0.7, zorder=5, label="screenshot estimate")
        ax.legend(fontsize=5.5, loc="lower left", frameon=False)

    ax.set_xlabel("$a^{*}$", fontsize=7.5, labelpad=1)
    ax.set_ylabel("$b^{*}$", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=6, length=2.5)
    ax.set_xticks([-60, -30, 0, 30, 60])
    ax.set_yticks([-60, -30, 0, 30, 60])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ─── Panel B ──────────────────────────────────────────────────────────────────
def panel_b(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    bb = ax.get_position()
    fig = ax.get_figure()
    panel_b_aspect = (bb.width * fig.get_figwidth()) / (bb.height * fig.get_figheight())

    frames = [
        ("sources/rsvp_fix.png", "Fixation\n3–6 s"),
        ("sources/rsvp_stim_red.png", "Colored disc\n1.5 s"),
        ("sources/rsvp_stim_kblack.png", "Letter stream\n1-back on K"),
    ]
    # Fractions of the panel axes; each capture gets its own square inset so the
    # disc cannot be stretched.
    side, gap = 0.285, 0.060
    # vertically center the capture strip (labels included) on the row midline,
    # matching panel A; hug the left edge of the panel instead of overflowing right
    strip_h = side * panel_b_aspect
    y0 = 0.5 - strip_h / 2 + 0.055
    x = 0.0
    for fname, label in frames:
        # Height compensates for the panel's non-square shape so the capture,
        # and therefore the disc, stays square on the page.
        sub = ax.inset_axes([x, y0, side, side * panel_b_aspect],
                            transform=ax.transAxes)
        img = np.asarray(Image.open(FIGDIR / fname).convert("RGB"))
        sub.imshow(img, aspect="equal")
        sub.set_xticks([])
        sub.set_yticks([])
        for sp in sub.spines.values():
            sp.set_edgecolor("0.4")
            sp.set_linewidth(0.6)
        ax.text(x + side / 2, y0 - 0.04, label, transform=ax.transAxes,
                fontsize=6.2, ha="center", va="top", color="0.2", linespacing=1.35)
        x += side + gap

    for k in range(2):
        xa = 0.0 + (k + 1) * side + k * gap
        ymid = y0 + side * panel_b_aspect / 2
        ax.annotate("", xy=(xa + gap - 0.006, ymid),
                    xytext=(xa + 0.006, ymid),
                    xycoords=ax.transAxes, textcoords=ax.transAxes,
                    arrowprops=dict(arrowstyle="-|>", lw=0.9, color="0.3",
                                    mutation_scale=8))
    # run-count / ISI header removed (2026-09-02, author request): the caption
    # and Methods carry those numbers.


# ─── Panel C ──────────────────────────────────────────────────────────────────
# 2026-09-02 (author request): the three stage boxes carried bullet-point TEXT
# only. Each stage is now depicted graphically; the wording lives in the
# fig:paradigm caption. Only short anchor labels (LORO / LOCO / RDM, etc.)
# remain in the panel.

def _square_inset(ax, x0, y0, w):
    """Square inset in panel-C axes fractions; compensates the panel aspect."""
    bb = ax.get_position()
    fig = ax.get_figure()
    asp = (bb.width * fig.get_figwidth()) / (bb.height * fig.get_figheight())
    sub = ax.inset_axes([x0, y0, w, w * asp], transform=ax.transAxes)
    sub.set_xlim(-1.6, 1.6)
    sub.set_ylim(-1.6, 1.6)
    sub.set_aspect("equal")
    sub.axis("off")
    return sub


def _mini_ring(sub, rgbs, rot=0.0, ghost_rot=None, skip=None, r=1.05, s=11):
    """8-hue ring; optional misaligned ghost run and a held-out (skipped) hue."""
    sub.add_patch(Circle((0, 0), r, fill=False, ec="0.85", lw=0.5))
    if ghost_rot is not None:
        for h, rgb in zip(HUES, rgbs):
            a = np.radians(h + ghost_rot)
            sub.scatter(r * np.cos(a), r * np.sin(a), s=s, facecolors="none",
                        edgecolors=[rgb], linewidths=0.7)
    for h, rgb in zip(HUES, rgbs):
        a = np.radians(h + rot)
        if skip is not None and h == skip:
            sub.scatter(r * np.cos(a), r * np.sin(a), s=s + 3, facecolors="none",
                        edgecolors="0.35", linewidths=0.7, linestyle="--")
            continue
        sub.scatter(r * np.cos(a), r * np.sin(a), s=s, c=[rgb],
                    edgecolors="0.3", linewidths=0.35)


def panel_c(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    stages = [
        ("Stage A · Alignment", "#1f4e79"),
        ("Stage B · Neural features", "#1e6b52"),
        ("Stage C · Model and filter", "#a33a2b"),
    ]

    bw, bgap, by, bh = 29.7, 4.3, 6.0, 88.0
    hdr = 13.0
    x = 0.7
    for title, col in stages:
        ax.add_patch(FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0.6,rounding_size=1.5",
                                    fc="white", ec=col, lw=1.0, zorder=2))
        ax.add_patch(FancyBboxPatch((x, by + bh - hdr), bw, hdr,
                                    boxstyle="round,pad=0.6,rounding_size=1.5",
                                    fc=col, ec=col, lw=1.0, zorder=3))
        ax.text(x + bw / 2, by + bh - hdr / 2, title, fontsize=7.0, fontweight="bold",
                ha="center", va="center", color="white", zorder=4)
        x += bw + bgap

    for k in range(2):
        xa = 0.7 + (k + 1) * bw + k * bgap
        ax.add_patch(FancyArrowPatch((xa + 0.6, by + bh / 2), (xa + bgap - 0.6, by + bh / 2),
                                    arrowstyle="-|>", mutation_scale=8,
                                    lw=1.0, color="0.35", zorder=5))

    rgbs = stim_rgb()
    glyph_y = 0.30          # inset bottom, panel fractions
    label_y = 0.195         # anchor-label baseline
    sw_a, sw_b, sw_c = 0.095, 0.078, 0.095

    # Stage A: misaligned per-run patterns -> one shared space
    sub = _square_inset(ax, 0.030, glyph_y, sw_a)
    _mini_ring(sub, rgbs, ghost_rot=24.0)
    ax.text(0.030 + sw_a / 2, label_y, "per-run\npatterns", transform=ax.transAxes,
            fontsize=5.6, ha="center", va="top", color="0.25", linespacing=1.3)
    ax.annotate("", xy=(0.176, 0.50), xytext=(0.138, 0.50), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.35",
                                mutation_scale=7))
    sub = _square_inset(ax, 0.180, glyph_y, sw_a)
    _mini_ring(sub, rgbs)
    ax.text(0.180 + sw_a / 2, label_y, "shared space\n(control-trained SRM)",
            transform=ax.transAxes, fontsize=5.6, ha="center", va="top",
            color="0.25", linespacing=1.3)

    # Stage B: LORO (diagonal confusion), LOCO (held-out hue decoded), RDM
    # six runs as color strips (each run contains all 8 hues); the top strip is
    # pulled out with a dashed outline = the held-out run
    sub = _square_inset(ax, 0.362, glyph_y, sw_b)
    n_runs, seg_w, row_h, row_gap = 6, 2.2 / 8, 0.30, 0.115
    y_top = (n_runs * (row_h + row_gap)) / 2
    for rr in range(n_runs):
        yb = y_top - (rr + 1) * (row_h + row_gap)
        held = (rr == 0)
        xoff = 0.30 if held else 0.0
        for k, rgb in enumerate(rgbs):
            sub.add_patch(plt.Rectangle((-1.1 + xoff + k * seg_w, yb), seg_w, row_h,
                                        fc=rgb, ec="none",
                                        alpha=0.55 if held else 1.0))
        sub.add_patch(plt.Rectangle((-1.1 + xoff, yb), 8 * seg_w, row_h,
                                    fill=False, ec="0.25", lw=0.6,
                                    linestyle="--" if held else "-"))
    ax.text(0.362 + sw_b / 2, label_y, "LORO\nclassification", transform=ax.transAxes,
            fontsize=5.6, ha="center", va="top", color="0.25", linespacing=1.3)

    sub = _square_inset(ax, 0.462, glyph_y, sw_b)
    _mini_ring(sub, rgbs, skip=225, r=1.05, s=9)
    aa = np.radians(235.0)
    sub.add_patch(FancyArrowPatch((0, 0), (0.78 * np.cos(aa), 0.78 * np.sin(aa)),
                                  arrowstyle="-|>", mutation_scale=5,
                                  lw=0.9, color="0.2"))
    ax.text(0.462 + sw_b / 2, label_y, "LOCO\ninterpolation", transform=ax.transAxes,
            fontsize=5.6, ha="center", va="top", color="0.25", linespacing=1.3)

    # pairwise representational distance: a similar pair sits close (short
    # double arrow), a dissimilar pair far (long double arrow)
    sub = _square_inset(ax, 0.562, glyph_y, sw_b)
    def _pair(y, i, j, x1, x2):
        sub.scatter([x1, x2], [y, y], s=16, c=[rgbs[i], rgbs[j]],
                    edgecolors="0.3", linewidths=0.4, zorder=5)
        sub.add_patch(FancyArrowPatch((x1 + 0.16, y), (x2 - 0.16, y),
                                      arrowstyle="<|-|>", mutation_scale=4,
                                      lw=0.8, color="0.3"))
    _pair(0.62, 0, 1, -0.72, 0.72)     # red-orange: similar, short
    _pair(-0.62, 0, 4, -1.15, 1.15)    # red-cyan: dissimilar, long
    ax.text(0.562 + sw_b / 2, label_y, "RDM\ngeometry", transform=ax.transAxes,
            fontsize=5.6, ha="center", va="top", color="0.25", linespacing=1.3)

    # Stage C: fitted hue rotation -> pre-image filter (Original / Filtered)
    sub = _square_inset(ax, 0.700, glyph_y, sw_c)
    _mini_ring(sub, rgbs, r=1.05, s=9)
    for h0, dd in ((90.0, -34.0), (180.0, -30.0), (315.0, 28.0)):
        a0, a1 = np.radians(h0), np.radians(h0 + dd)
        rr = 1.38
        sub.add_patch(FancyArrowPatch((rr * np.cos(a0), rr * np.sin(a0)),
                                      (rr * np.cos(a1), rr * np.sin(a1)),
                                      connectionstyle=f"arc3,rad={0.22 if dd > 0 else -0.22}",
                                      arrowstyle="-|>", mutation_scale=6,
                                      lw=1.0, color="0.2"))
    ax.text(0.700 + sw_c / 2, label_y,
            "fitted rotation\n($\\beta_s$, $\\beta_c$)", transform=ax.transAxes,
            fontsize=5.6, ha="center", va="top", color="0.25", linespacing=1.3)
    ax.annotate("", xy=(0.848, 0.50), xytext=(0.810, 0.50), xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", lw=0.8, color="0.35",
                                mutation_scale=7))
    sub = _square_inset(ax, 0.852, glyph_y, sw_c)
    sub.set_xlim(-1.6, 1.6); sub.set_ylim(-1.6, 1.6)
    # same color source as the ring dots: canonical STIM_LAB anchors; the
    # filtered swatch rotates the hue keeping that anchor's L*/chroma
    demo_idx = [1, 3, 6]           # 45 deg orange, 135 deg green, 270 deg purple
    for i, k in enumerate(demo_idx):
        xc = -1.05 + i * 1.05
        L0, a0, b0 = MEASURED_LAB[k]
        sub.add_patch(plt.Rectangle((xc - 0.42, 0.30), 0.84, 0.84,
                                    fc=lab_to_srgb(L0, a0, b0), ec="0.4", lw=0.4))
        ch = float(np.hypot(a0, b0))
        hp = np.degrees(np.arctan2(b0, a0)) - 28.0
        ap = ch * np.cos(np.radians(hp))
        bp = ch * np.sin(np.radians(hp))
        sub.add_patch(plt.Rectangle((xc - 0.42, -1.14), 0.84, 0.84,
                                    fc=lab_to_srgb(L0, ap, bp), ec="0.4", lw=0.4))
        sub.add_patch(FancyArrowPatch((xc, 0.24), (xc, -0.24),
                                      arrowstyle="-|>", mutation_scale=4,
                                      lw=0.8, color="0.35"))
    ax.text(0.852 + sw_c / 2, label_y, "pre-image\nfilter", transform=ax.transAxes,
            fontsize=5.6, ha="center", va="top", color="0.25", linespacing=1.3)


# ─── Assemble ─────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(7.0, 4.05))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.80, 0.62],
                          width_ratios=[0.88, 1.12],
                          left=0.075, right=0.985, top=0.935, bottom=0.035,
                          wspace=0.20, hspace=0.38)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        ax.text(-0.02, 1.07, letter, transform=ax.transAxes, fontsize=10,
                fontweight="bold", ha="left", va="bottom")

    fig.savefig(f"{OUT_STEM}.pdf", dpi=300)
    fig.savefig(f"{OUT_STEM}.png", dpi=300)
    print(f"Saved: {OUT_STEM}.pdf")
    print(f"Saved: {OUT_STEM}.png")


if __name__ == "__main__":
    main()
