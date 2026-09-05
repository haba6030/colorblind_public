#!/usr/bin/env python3
"""
generate_fig3_geometry_r6.py

Figure 5 (label `fig:geometry`, file stem `fig3_geometry`) — REVISION R6 version.

R6 (docs/PAPER/REVISION_PLAN_MOTION_GEOMETRY_2026-08-06.md) moves the DeltaRDM
heatmap (old panel A) out of this figure and into the pipeline schematic
(Figure 3, file stem `fig3_workflow`).  What remains here is the Procrustes
disparity panel only, so the panel letters A/B are dropped entirely.

2026-09-02 (MANUSCRIPT_EDITS_CONSOLIDATED.md §5.1): significance asterisks are
removed from the panel entirely.  The protan-V1 mark (symmetric-LOSO p = .045)
does not survive the head-motion-corrected pipeline (p = .077), and a single
remaining asterisk would visually assert per-participant regional attribution,
which §0.5 C4 withdraws.  Significance is stated only in the text and in
Supplementary §S2.

Provenance
----------
This is a direct port of `generate_fig3.py`, which produced the currently
committed `fig3_geometry.pdf/.png` and was deleted from the tree in commit
6f66e67.  Recover the original with:
    git show 6f66e67^:docs/PAPER/Figures/scripts/generate_fig3.py

Data (unchanged, no recomputation)
----------------------------------
    analysis/phase2_SRM_across_between/results/loo_consistent/
        20260218_163819/loo_consistent_results.json
  method: hc_only_srm_loo_consistent  (HC-only SRM + LOO-consistent reference
  + Crawford & Howell modified t).  `cvd_score` / `hc_loo_disparities` are
  orthogonal-Procrustes disparities (Frobenius residual of the normalized
  configurations; see rerun_loo_consistent.py:92-98).

Deliberate difference from the original panel B
-----------------------------------------------
  * y-axis label was "SRM disparity (correlation distance)".  The quantity is
    a Procrustes disparity, not a correlation distance -- the old label was
    wrong and contradicted the R6 caption.  Relabelled to
    "Procrustes disparity (SRM-aligned)".
  * the embedded axes title "Disparity by ROI" is removed: with a single panel
    it becomes a figure title, which the figure QC checklist forbids.
  * canvas is 180 mm x 85 mm (was 180 x 120 with three sub-axes), so the panel
    still prints at ~7 pt when included at \\textwidth.

Everything else -- colours, markers, HC band, jitter seed, y-limits -- is
byte-for-byte the original logic.

Output: fig3_geometry_r6.png (300 dpi), fig3_geometry_r6.pdf (vector)
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : docs/PAPER/Figures/scripts/generate_fig3_geometry_r6.py  (development repository, commit 53c81c2)
# Role      : Figure 5 (Procrustes disparity per participant and ROI).
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Paths -------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[3].parent   # repo root
LOO_JSON = (BASE / 'analysis/phase2_SRM_across_between/results/loo_consistent/'
                   '20260218_163819/loo_consistent_results.json')
OUT_DIR = BASE / 'docs/PAPER/Figures'

# --- Constants ---------------------------------------------------------------
ROIS = ['V1', 'V2', 'V3', 'hV4']
ROI_LABELS = ['V1', 'V2', 'V3', 'hV4']

# --- Style (identical to generate_fig3.py) -----------------------------------
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'pdf.fonttype': 42,
    'svg.fonttype': 'none',
})


def load_loo_results():
    with open(LOO_JSON) as f:
        return json.load(f)


def make_figure():
    fig_w_in = 180 / 25.4
    fig_h_in = 85 / 25.4
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    # single panel, legend in the reserved strip on the right
    ax_b = fig.add_axes([0.085, 0.175, 0.635, 0.775])

    loo = load_loo_results()

    x = np.arange(len(ROIS))
    width = 0.18
    offsets = {'sub-08': -width / 2, 'sub-09': width / 2}
    colors_cvd = {
        'sub-08': '#D55E00',   # vermillion (deutan)
        'sub-09': '#0072B2',   # blue (protan)
    }
    markers = {'sub-08': 'o', 'sub-09': 's'}
    labels = {
        'sub-08': 'Deutan',
        'sub-09': 'Protan',
    }

    # HC confidence band
    hc_means, hc_stds = [], []
    for roi_name in ROIS:
        s = loo['results'][roi_name]['summary']
        hc_means.append(s['hc_mean'])
        hc_stds.append(s['hc_std'])
    hc_means = np.array(hc_means)
    hc_stds = np.array(hc_stds)

    ax_b.fill_between(x, hc_means - hc_stds, hc_means + hc_stds,
                      alpha=0.18, color='#4CAF50', label='Control mean $\\pm$ 1 SD',
                      zorder=1)
    ax_b.plot(x, hc_means, color='#2E7D32', linewidth=1.2, linestyle='--',
              marker='D', markersize=4, label='Control mean', zorder=2,
              markerfacecolor='white', markeredgewidth=1.0)

    # HC individual LOO points (same jitter seed as the original)
    hc_sub_labels = ['sub-01', 'sub-02', 'sub-03', 'sub-04',
                     'sub-05', 'sub-06', 'sub-07']
    rng = np.random.default_rng(42)
    for roi_idx, roi_name in enumerate(ROIS):
        hc_disp = loo['results'][roi_name]['hc_loo_disparities']
        hc_vals = [hc_disp[s] for s in hc_sub_labels if s in hc_disp]
        jitter = rng.uniform(-0.08, 0.08, len(hc_vals))
        ax_b.scatter(np.full(len(hc_vals), roi_idx) + jitter, hc_vals,
                     color='#4CAF50', alpha=0.4, s=8, zorder=1, linewidths=0)

    # CVD subjects
    for sub in ['sub-08', 'sub-09']:
        scores = []
        for roi_name in ROIS:
            roi_data = loo['results'][roi_name]['individual_cvd'][sub]
            scores.append(roi_data['cvd_score'])

        xpos = x + offsets[sub]
        ax_b.plot(xpos, scores, color=colors_cvd[sub], linewidth=1.0,
                  linestyle='-', zorder=3)
        ax_b.scatter(xpos, scores, color=colors_cvd[sub], marker=markers[sub],
                     s=30, zorder=4, label=labels[sub])

        # No significance marks on the panel (2026-09-02, §5.1): the protan-V1
        # symmetric-LOSO p = .045 drops to .077 under the head-motion-corrected
        # pipeline, and a lone asterisk would visually assert per-participant
        # regional attribution, which the manuscript withdraws (§0.5 C4).
        # Region-wise tests for both pipelines live in Supplementary §S2.

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(ROI_LABELS, fontsize=7.5)
    ax_b.tick_params(axis='y', labelsize=7)
    ax_b.set_ylabel('Procrustes disparity (SRM-aligned)', fontsize=7.5,
                    labelpad=4)
    ax_b.set_xlabel('ROI', fontsize=7.5)

    ax_b.set_ylim(0.30, 1.05)
    ax_b.set_xlim(-0.5, len(ROIS) - 0.5)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)

    # legend outside, to the right of the axes
    ax_b.legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                bbox_transform=ax_b.transAxes, borderaxespad=0,
                frameon=False, fontsize=7, ncol=1,
                handlelength=1.6, handletextpad=0.5, labelspacing=0.7)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / 'fig3_geometry_r6.png'
    out_pdf = OUT_DIR / 'fig3_geometry_r6.pdf'
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.05,
                facecolor='white')
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05,
                facecolor='white')
    print(f'Saved: {out_png}')
    print(f'Saved: {out_pdf}')
    plt.close(fig)


if __name__ == '__main__':
    make_figure()
