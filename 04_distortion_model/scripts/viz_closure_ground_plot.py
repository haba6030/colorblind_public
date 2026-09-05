"""Closure ground plot — z-score composite loss landscape over (β_s, β_c) grid.

Reconstructs Phase B v6 Step 3 composite (s10b_v6_pca_rdm.py:301-304, :573, :603)
on the canonical 2-Component grid (BS_GRID × BC_GRID) using the *full 7-HC pool*
(no resample) for each final candidate's combo, and overlays the 300-resample
argmin cloud from v6 JSON storage.

Output:
  results/visualizations/closure_ground_plot/
    closure_ground_plot_S08_bs_dom.png   (γ_all + RDM_V1)
    closure_ground_plot_S08_bc_dom.png   (γ_OY + RDM_V2)
    closure_ground_plot_S09_bc_rot.png   (γ_all + RDM_V1)
    closure_ground_plot_summary.png      (3 panels side by side)
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/viz_closure_ground_plot.py  (development repository, commit 53c81c2)
# Role      : Loss-landscape reconstruction on the full control pool (Figure S1).
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

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from two_comp import forward_2comp, BS_GRID, BC_GRID
from neural_loss import load_amplitudes, load_hc_pool, ROI_K
from behav_loss import load_jnd_per_pair, PAIR_HUES, HC_JND_SUBJS
from utils_forward_model import create_basis_full, HUE_ANGLES
from s8_loo_train_test import jnd_baseline_from_pool

from s10b_v6_pca_rdm import (
    make_gamma_pair_atom,
    make_rdm_atom,
    grid_eval_2comp,
    zscore_grid,
    HC_SUBJS,
)

OUT_DIR = SCRIPT_DIR.parent / "results" / "visualizations" / "pipeline2_primary_4col"
OUT_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR = SCRIPT_DIR.parent / "results" / "s10_inclusion"

# Final candidates per PIPELINE_2_CLOSURE.md §5.1
CANDIDATES = [
    {
        "id": "S08_bs_dom",
        "subject": "sub-08",
        "family": "deutan",
        "label": "S08-βs-dom (38, −10)",
        "combo_key": "γALL|RDMV1|noLOCO",
        "gamma_atoms": ["ALL"],
        "rdm_rois": ["V1"],
        "fit_point": (38.0, -10.0),
        "title": "Sub-08 deutan · γ_all + RDM_V1",
    },
    {
        "id": "S08_bc_dom",
        "subject": "sub-08",
        "family": "deutan",
        "label": "S08-βc-dom (6, −42)",
        "combo_key": "γOY|RDMV2|noLOCO",
        "gamma_atoms": ["OY"],
        "rdm_rois": ["V2"],
        "fit_point": (6.0, -42.0),
        "title": "Sub-08 deutan · γ_OY + RDM_V2",
    },
    {
        "id": "S09_bc_rot",
        "subject": "sub-09",
        "family": "protan",
        "label": "S09-βc-rot (2, +24)",
        "combo_key": "γALL|RDMV1|noLOCO",
        "gamma_atoms": ["ALL"],
        "rdm_rois": ["V1"],
        "fit_point": (2.0, 24.0),
        "title": "Sub-09 protan · γ_all + RDM_V1",
    },
]


def build_composite_full_hc(cand):
    """Recompute composite on (BS_GRID × BC_GRID) using full 7-HC pool.

    Mirrors s10b_v6_pca_rdm.py Step 3 z-score composite:
      comp = (Σ_atom zscore_grid(atom_grid)) / sqrt(n_atoms)
    """
    subject = cand["subject"]
    family = cand["family"]

    cvd_jnd = load_jnd_per_pair(subject)

    atoms_grids = []
    atom_labels = []

    # γ atoms (use full HC_JND_SUBJS as train pool)
    for p in cand["gamma_atoms"]:
        fn = make_gamma_pair_atom(p, cvd_jnd, HC_JND_SUBJS)
        if fn is None:
            continue
        g = grid_eval_2comp(fn, family)
        atoms_grids.append(g)
        atom_labels.append(f"γ_{p}")

    # RDM atoms (use full 7-HC pool)
    for roi in cand["rdm_rois"]:
        cvd_amp = load_amplitudes(subject, roi)
        hc_amps = load_hc_pool(roi)
        pool_amps = {h: hc_amps[h] for h in HC_SUBJS if h in hc_amps}
        K = ROI_K[roi]
        C_b = create_basis_full(K, basis_type="fe")[HUE_ANGLES.astype(int)]
        fn = make_rdm_atom(roi, cvd_amp, pool_amps, C_b, K)
        if fn is None:
            continue
        g = grid_eval_2comp(fn, family)
        atoms_grids.append(g)
        atom_labels.append(f"RDM_{roi}")

    n_a = len(atoms_grids)
    z_sum = None
    z_each = []
    for g in atoms_grids:
        z = zscore_grid(g)
        z_each.append(z)
        z_sum = z if z_sum is None else z_sum + z
    comp = z_sum / np.sqrt(n_a)
    return comp, atoms_grids, z_each, atom_labels, n_a


def load_resample_argmins(cand):
    """Extract 300 (β_s, β_c) argmin points from v6 JSON storage."""
    path = JSON_DIR / f"s10b_v6_pca_rdm_results_{cand['subject']}.json"
    d = json.load(open(path))
    store = d["storage"].get(cand["combo_key"])
    if store is None:
        return np.array([]), np.array([])
    two = store["2comp"]
    bs = np.array([r["beta_s"] for r in two], dtype=float)
    bc = np.array([r["beta_c"] for r in two], dtype=float)
    return bs, bc


def plot_panel(ax, comp, atom_labels, n_a, fit_pt, bs_samples, bc_samples, title):
    # Composite landscape
    BC, BS = np.meshgrid(BC_GRID, BS_GRID)
    vmin = np.nanpercentile(comp, 1)
    vmax = np.nanpercentile(comp, 99)
    im = ax.pcolormesh(BC, BS, comp, cmap="viridis_r", shading="auto",
                       norm=Normalize(vmin=vmin, vmax=vmax))
    # Contours
    levels = np.nanpercentile(comp, [5, 15, 30, 50, 70])
    cs = ax.contour(BC, BS, comp, levels=levels, colors="white",
                    alpha=0.4, linewidths=0.7)

    # Resample argmin cloud
    if len(bs_samples) > 0:
        counts = Counter(zip(bs_samples, bc_samples))
        for (bs_v, bc_v), n in counts.items():
            ax.scatter(bc_v, bs_v, s=10 + 2 * n, c="white",
                       edgecolors="black", linewidth=0.4, alpha=0.55, zorder=3)

    # Closure fit point (canonical)
    fit_bs, fit_bc = fit_pt
    ax.scatter(fit_bc, fit_bs, marker="*", s=350, c="red",
               edgecolors="white", linewidth=1.2, zorder=5,
               label=f"closure fit ({fit_bs:.0f}, {fit_bc:.0f})")

    # Full-pool argmin
    flat = int(np.nanargmin(comp.ravel()))
    i, j = np.unravel_index(flat, comp.shape)
    ax.scatter(BC_GRID[j], BS_GRID[i], marker="x", s=120, c="orange",
               linewidth=2.5, zorder=4,
               label=f"full-7HC argmin ({BS_GRID[i]:.0f}, {BC_GRID[j]:.0f})")

    ax.set_xlabel("β_c (confusion-axis rotation, deg)", fontsize=10)
    ax.set_ylabel("β_s (S-cone axis rotation, deg)", fontsize=10)
    ax.set_title(f"{title}\nn_atoms={n_a}: {', '.join(atom_labels)} "
                 f"· weight = 1/√{n_a} ≈ {1/np.sqrt(n_a):.3f}",
                 fontsize=10)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_xlim(BC_GRID[0], BC_GRID[-1])
    ax.set_ylim(BS_GRID[0], BS_GRID[-1])
    return im


def main():
    # Per-candidate full figure (with z_each atoms)
    for cand in CANDIDATES:
        print(f"\n=== {cand['id']} ===", flush=True)
        comp, atoms_grids, z_each, atom_labels, n_a = build_composite_full_hc(cand)
        bs_samples, bc_samples = load_resample_argmins(cand)
        print(f"  composite shape={comp.shape}, "
              f"min={np.nanmin(comp):.3f}, max={np.nanmax(comp):.3f}", flush=True)
        print(f"  300-resample argmins: {len(bs_samples)}", flush=True)

        # Layout: top row = composite (large); bottom row = per-atom z-grids
        fig = plt.figure(figsize=(6 + 4 * n_a, 8.5))
        gs = fig.add_gridspec(2, max(n_a, 2), height_ratios=[1.4, 1], hspace=0.42, wspace=0.32)

        ax_main = fig.add_subplot(gs[0, :])
        im = plot_panel(ax_main, comp, atom_labels, n_a,
                        cand["fit_point"], bs_samples, bc_samples,
                        cand["title"])
        fig.colorbar(im, ax=ax_main, label="composite z (lower = better fit)",
                     shrink=0.85, pad=0.02)

        # Per-atom panels (full z-range so info-density mismatch L5 visible)
        BC, BS = np.meshgrid(BC_GRID, BS_GRID)
        for k, (z, lbl) in enumerate(zip(z_each, atom_labels)):
            ax = fig.add_subplot(gs[1, k])
            zmin = np.nanmin(z); zmax = np.nanmax(z)
            im2 = ax.pcolormesh(BC, BS, z, cmap="viridis_r", shading="auto",
                                norm=Normalize(vmin=zmin, vmax=zmax))
            ax.scatter(cand["fit_point"][1], cand["fit_point"][0],
                       marker="*", s=160, c="red", edgecolors="white",
                       linewidth=0.8, zorder=4)
            ax.set_title(f"{lbl}  (full z-range "
                         f"[{zmin:+.2f}, {zmax:+.2f}])", fontsize=9)
            ax.set_xlabel("β_c", fontsize=8)
            ax.set_ylabel("β_s", fontsize=8)
            fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.02)

        fig.suptitle(f"Closure z-score composite ground plot — {cand['label']}",
                     fontsize=12, y=0.995)
        out_path = OUT_DIR / f"closure_ground_plot_{cand['id']}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  → {out_path.name}", flush=True)

    # Summary 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.5))
    for ax, cand in zip(axes, CANDIDATES):
        comp, _, _, atom_labels, n_a = build_composite_full_hc(cand)
        bs_samples, bc_samples = load_resample_argmins(cand)
        im = plot_panel(ax, comp, atom_labels, n_a,
                        cand["fit_point"], bs_samples, bc_samples,
                        cand["title"])
        fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
    fig.suptitle("Phase 2 closure — z-score composite ground plot "
                 "(weight = 1/√n_a per atom; 7-HC pool; 300 resamples = white cloud)",
                 fontsize=12, y=1.02)
    out_path = OUT_DIR / "closure_ground_plot_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out_path}", flush=True)


if __name__ == "__main__":
    main()
