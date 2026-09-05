#!/usr/bin/env python3
"""
lambda_stability_loco.py -- Forward-model robustness axis 3: GCV ridge-lambda
stability (review §B, third check).

Axes 1 (per-color dominance) and 2 (residual structure) are already produced in
results/loco_reinforcement/{per_color_breakdown,residual_structure}.json. This
script fills the one missing robustness axis:

  Q1  Is the GCV-selected ridge alpha STABLE across LOCO folds / subjects, or does
      it wander the whole grid (which would make the hV4 encoding GO a
      lambda-coincidence)?
  Q2  Is the HC encoding rho a knife-edge on the GCV-chosen alpha, or a plateau?
      Recompute mean HC encoding rho at every FIXED grid alpha -- if the GO region
      (hV4) holds its rho across a broad alpha range, p~0.044 is not an alpha fluke.

Uses ONLY frozen primitives from utils_forward_model / loco_canonical -- no
reimplementation of the encoder. ENCODING decoder = 'gcv' (matches canonical).

Output (flat, no timestamp dir):
  results/loco_reinforcement/lambda_stability.json
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/lambda_stability_loco.py  (development repository, commit 53c81c2)
# Role      : Ridge-penalty grid stability (S6). Output: results/lambda_stability.json
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

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils_forward_model import (
    HC_SUBJECTS, CVD_SUBJECTS, ROIS, K_VALUES, ALPHA_GRID, HUE_ANGLES,
    load_amplitudes, create_basis_matrix, fit_W_ridge, gcv_select_alpha,
)

LOCAL_BASELINE = (
    Path(__file__).resolve().parents[3]
    / "analysis/phase1_procrustes_decoding/results/visualization/"
      "full_dataset_C010_with_residuals"
)


def loco_alpha_and_rho(amp, C8, alpha=None):
    """One LOCO pass. If alpha is None -> GCV-select per fold (return chosen alphas).
    If alpha is a float -> fixed ridge at that alpha. Returns (rho_per_color,
    alpha_per_color). alpha_per_color is the GCV pick (or the fixed alpha)."""
    amp = np.asarray(amp, float)
    n_runs, n_colors, V = amp.shape
    amp_mean = amp.mean(axis=0)
    rho = np.zeros(n_colors)
    alphas = np.zeros(n_colors)
    for c in range(n_colors):
        train = [k for k in range(n_colors) if k != c]
        C_train = np.tile(C8[train], (n_runs, 1))
        X_train = amp[:, train, :].reshape(-1, V)
        if alpha is None:
            a, _ = gcv_select_alpha(C_train, X_train)
        else:
            a = alpha
        alphas[c] = a
        W = fit_W_ridge(C_train, X_train, a)
        Y_pred = (C8[c:c + 1] @ W)[0]
        Y_act = amp_mean[c]
        if Y_pred.std() < 1e-10 or Y_act.std() < 1e-10:
            rho[c] = 0.0
        else:
            r = np.corrcoef(Y_pred, Y_act)[0, 1]
            rho[c] = r if np.isfinite(r) else 0.0
    return rho, alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_dir", default=str(LOCAL_BASELINE))
    ap.add_argument("--rois", nargs="+", default=ROIS)
    ap.add_argument("--out", default=str(
        Path(__file__).parent.parent / "results/loco_reinforcement/lambda_stability.json"))
    args = ap.parse_args()

    baseline = Path(args.baseline_dir)
    roi_label = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
    log_grid = np.log10(ALPHA_GRID)

    result = {"alpha_grid": ALPHA_GRID, "baseline_dir": str(baseline), "rois": {}}

    for roi in args.rois:
        K = K_VALUES[roi]
        C8 = create_basis_matrix(HUE_ANGLES, K, basis_type="fe")

        # ---- Q1: GCV-selected alpha per (subject, fold) for HC ----
        hc_alphas = []          # flat list over subjects x 8 folds
        per_subject = {}
        for s in HC_SUBJECTS:
            try:
                amp = load_amplitudes(baseline, s, roi)
            except FileNotFoundError:
                continue
            _, alphas = loco_alpha_and_rho(amp, C8, alpha=None)
            hc_alphas.extend(alphas.tolist())
            per_subject[s] = alphas.tolist()

        hc_alphas = np.array(hc_alphas)
        log_a = np.log10(hc_alphas)
        counts = Counter(hc_alphas.tolist())
        modal_alpha, modal_n = counts.most_common(1)[0]
        # bootstrap CI over folds for the log10(median alpha)
        rng = np.random.default_rng(0)
        boot_med = [np.median(log_a[rng.integers(0, len(log_a), len(log_a))])
                    for _ in range(2000)]
        ci = np.percentile(boot_med, [2.5, 97.5]).tolist()

        # ---- Q2: mean HC encoding rho at every FIXED grid alpha ----
        rho_vs_alpha = {}
        for a in ALPHA_GRID:
            per_sub_mean = []
            for s in HC_SUBJECTS:
                try:
                    amp = load_amplitudes(baseline, s, roi)
                except FileNotFoundError:
                    continue
                rho, _ = loco_alpha_and_rho(amp, C8, alpha=a)
                per_sub_mean.append(float(rho.mean()))
            rho_vs_alpha[a] = float(np.mean(per_sub_mean))
        # rho at GCV (per-fold selected) alpha
        gcv_rho = []
        for s in HC_SUBJECTS:
            try:
                amp = load_amplitudes(baseline, s, roi)
            except FileNotFoundError:
                continue
            rho, _ = loco_alpha_and_rho(amp, C8, alpha=None)
            gcv_rho.append(float(rho.mean()))
        gcv_rho_mean = float(np.mean(gcv_rho))

        rho_vals = np.array(list(rho_vs_alpha.values()))
        peak_rho = float(rho_vals.max())
        # fraction of grid where rho stays within 90% of peak = plateau width
        plateau = int(np.sum(rho_vals >= 0.9 * peak_rho))

        result["rois"][roi_label[roi]] = {
            "n_folds": int(len(hc_alphas)),
            "modal_alpha": float(modal_alpha),
            "modal_fraction": float(modal_n / len(hc_alphas)),
            "median_alpha": float(np.median(hc_alphas)),
            "log10_alpha_median_ci95": ci,
            "log10_alpha_sd": float(log_a.std()),
            "alpha_histogram": {str(a): int(counts.get(a, 0)) for a in ALPHA_GRID},
            "per_subject_alphas": per_subject,
            "rho_vs_fixed_alpha": {str(a): v for a, v in rho_vs_alpha.items()},
            "gcv_rho_mean": gcv_rho_mean,
            "peak_fixed_rho": peak_rho,
            "plateau_grid_points_within_90pct_peak": plateau,
        }
        print(f"[{roi_label[roi]}] modal alpha={modal_alpha:g} "
              f"({modal_n}/{len(hc_alphas)}={modal_n/len(hc_alphas):.0%}), "
              f"log10-alpha SD={log_a.std():.2f}, "
              f"GCV rho={gcv_rho_mean:.3f}, peak fixed rho={peak_rho:.3f} "
              f"@plateau {plateau}/{len(ALPHA_GRID)} grid pts")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
