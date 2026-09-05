#!/usr/bin/env python3
"""
basis_comparison.py — Systematic basis ablation for forward encoding model.

Tests FE-{2,3,4,6,8,12} and LF-{2,4} with ridge_gcv on LORO and LOCO.
Goal: find whether V1/V2 benefits from a different basis than hV4.

Runs locally — no BrainIAK/MPI needed.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/basis_comparison.py  (development repository, commit 53c81c2)
# Role      : Basis-count ablation for the hue-channel model.
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


import numpy as np
import json
from pathlib import Path
from itertools import product

# --- Import from existing utils ---
import sys
sys.path.insert(0, str(Path(__file__).parent))
from utils_forward_model import (
    HC_SUBJECTS, CVD_SUBJECTS, ALL_SUBJECTS, ROIS,
    HUE_ANGLES, N_RUNS, N_COLORS, ALPHA_GRID,
    load_amplitudes, create_basis_full, create_basis_matrix,
    fit_W_ridge, gcv_select_alpha, predict_patterns,
    voxel_pattern_correlation, circular_distance, decode_hue,
)

# ============================================================================
# Configuration
# ============================================================================

# Local data path
LOCAL_BASELINE_DIR = Path(__file__).resolve().parents[3] / \
    'analysis' / 'phase1_preprocess_decoding' / 'results' / 'full_dataset_C010'

# Bases to test: (label, basis_type, n_channels)
BASES = [
    ('FE-2',  'fe',  2),
    ('FE-3',  'fe',  3),
    ('FE-4',  'fe',  4),
    ('FE-6',  'fe',  6),   # baseline
    ('FE-8',  'fe',  8),
    ('FE-12', 'fe', 12),
    ('LF-2',  'lf',  2),
    ('LF-4',  'lf',  4),
]

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'results' / 'basis_comparison'


# ============================================================================
# LORO: Leave-One-Run-Out
# ============================================================================

def run_loro_single(amp, basis_label, basis_type, n_channels):
    """LORO for one subject x one ROI x one basis.

    Args:
        amp: (6, 8, V_s)

    Returns:
        dict with mean_voxel_corr, per_fold_voxel_corr, best_alphas
    """
    C = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)  # (8, K)
    fold_corrs = []
    best_alphas = []

    for test_run in range(N_RUNS):
        train_runs = [r for r in range(N_RUNS) if r != test_run]

        # Training data: 5 runs × 8 colors
        X_train = amp[train_runs].reshape(-1, amp.shape[2])   # (40, V_s)
        C_train = np.tile(C, (len(train_runs), 1))            # (40, K)

        # Test data: 1 run, mean across run = just the run itself
        X_test = amp[test_run]  # (8, V_s)

        # Fit ridge with GCV
        best_alpha, _ = gcv_select_alpha(C_train, X_train)
        W = fit_W_ridge(C_train, X_train, best_alpha)  # (K, V_s)

        # Predict and evaluate
        Y_hat = predict_patterns(W, C)  # (8, V_s)
        r = voxel_pattern_correlation(Y_hat, X_test)  # (8,)
        fold_corrs.append(np.mean(r))
        best_alphas.append(best_alpha)

    return {
        'mean_voxel_corr': float(np.mean(fold_corrs)),
        'per_fold': [float(x) for x in fold_corrs],
        'best_alphas': best_alphas,
    }


# ============================================================================
# LOCO: Leave-One-Color-Out
# ============================================================================

def run_loco_single(amp, basis_label, basis_type, n_channels):
    """LOCO for one subject x one ROI x one basis.

    Args:
        amp: (6, 8, V_s)

    Returns:
        dict with mean_voxel_corr, per_color_voxel_corr, mae, best_alphas
    """
    C_full = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)  # (8, K)
    color_corrs = []
    color_maes = []
    best_alphas = []

    basis_full = create_basis_full(n_channels, basis_type)  # (360, K)

    for test_color in range(N_COLORS):
        train_colors = [c for c in range(N_COLORS) if c != test_color]

        # Training data: 6 runs × 7 colors
        X_train = amp[:, train_colors, :].reshape(-1, amp.shape[2])  # (42, V_s)
        C_train = np.tile(C_full[train_colors], (N_RUNS, 1))        # (42, K)

        # Test data: mean across 6 runs for held-out color
        X_test = amp[:, test_color, :].mean(axis=0, keepdims=True)   # (1, V_s)
        C_test = C_full[test_color:test_color+1]                     # (1, K)

        # Fit ridge with GCV
        best_alpha, _ = gcv_select_alpha(C_train, X_train)
        W = fit_W_ridge(C_train, X_train, best_alpha)  # (K, V_s)

        # Predict held-out color pattern
        Y_hat = predict_patterns(W, C_test)  # (1, V_s)
        r = voxel_pattern_correlation(Y_hat, X_test)  # (1,)
        color_corrs.append(float(r[0]))

        # Decode hue for MAE
        pred_hue = decode_hue(W, basis_full, X_test)  # (1,)
        mae = circular_distance(HUE_ANGLES[test_color], pred_hue[0])
        color_maes.append(float(mae))

        best_alphas.append(best_alpha)

    return {
        'mean_voxel_corr': float(np.mean(color_corrs)),
        'per_color': [float(x) for x in color_corrs],
        'mean_mae': float(np.mean(color_maes)),
        'per_color_mae': [float(x) for x in color_maes],
        'best_alphas': best_alphas,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f'Data dir: {LOCAL_BASELINE_DIR}')
    print(f'Output dir: {OUTPUT_DIR}')
    print(f'Bases: {[b[0] for b in BASES]}')
    print(f'ROIs: {ROIS}')
    print(f'Subjects: {ALL_SUBJECTS}')
    print()

    # --- Collect results ---
    results = {}

    for basis_label, basis_type, n_channels in BASES:
        print(f'=== {basis_label} (type={basis_type}, K={n_channels}) ===')
        results[basis_label] = {}

        for roi in ROIS:
            results[basis_label][roi] = {'loro': {}, 'loco': {}}

            for subj in ALL_SUBJECTS:
                try:
                    amp = load_amplitudes(LOCAL_BASELINE_DIR, subj, roi)
                except FileNotFoundError:
                    print(f'  SKIP sub-{subj} {roi} (not found)')
                    continue

                # LORO
                loro = run_loro_single(amp, basis_label, basis_type, n_channels)
                results[basis_label][roi]['loro'][subj] = loro

                # LOCO
                loco = run_loco_single(amp, basis_label, basis_type, n_channels)
                results[basis_label][roi]['loco'][subj] = loco

            # Summary
            hc_loro = [results[basis_label][roi]['loro'][s]['mean_voxel_corr']
                       for s in HC_SUBJECTS
                       if s in results[basis_label][roi]['loro']]
            hc_loco = [results[basis_label][roi]['loco'][s]['mean_voxel_corr']
                       for s in HC_SUBJECTS
                       if s in results[basis_label][roi]['loco']]

            if hc_loro:
                print(f'  {roi}: LORO HC={np.mean(hc_loro):.3f}±{np.std(hc_loro):.3f}  '
                      f'LOCO HC={np.mean(hc_loco):.3f}±{np.std(hc_loco):.3f}')

    # --- Save full results ---
    out_path = OUTPUT_DIR / 'basis_comparison_full.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nFull results saved to {out_path}')

    # --- Print summary table ---
    print('\n' + '=' * 80)
    print('SUMMARY: HC Mean voxel_corr (± SD)')
    print('=' * 80)

    # LORO
    print('\n--- LORO ---')
    print(f'{"Basis":<8}', end='')
    for roi in ROIS:
        print(f'  {roi:>12}', end='')
    print()

    for basis_label, _, _ in BASES:
        print(f'{basis_label:<8}', end='')
        for roi in ROIS:
            vals = [results[basis_label][roi]['loro'][s]['mean_voxel_corr']
                    for s in HC_SUBJECTS
                    if s in results[basis_label][roi]['loro']]
            if vals:
                print(f'  {np.mean(vals):>5.3f}±{np.std(vals):.3f}', end='')
            else:
                print(f'  {"N/A":>12}', end='')
        print()

    # LOCO
    print('\n--- LOCO ---')
    print(f'{"Basis":<8}', end='')
    for roi in ROIS:
        print(f'  {roi:>12}', end='')
    print()

    for basis_label, _, _ in BASES:
        print(f'{basis_label:<8}', end='')
        for roi in ROIS:
            vals = [results[basis_label][roi]['loco'][s]['mean_voxel_corr']
                    for s in HC_SUBJECTS
                    if s in results[basis_label][roi]['loco']]
            if vals:
                print(f'  {np.mean(vals):>5.3f}±{np.std(vals):.3f}', end='')
            else:
                print(f'  {"N/A":>12}', end='')
        print()

    # LOCO MAE
    print('\n--- LOCO MAE (degrees) ---')
    print(f'{"Basis":<8}', end='')
    for roi in ROIS:
        print(f'  {roi:>12}', end='')
    print()

    for basis_label, _, _ in BASES:
        print(f'{basis_label:<8}', end='')
        for roi in ROIS:
            vals = [results[basis_label][roi]['loco'][s]['mean_mae']
                    for s in HC_SUBJECTS
                    if s in results[basis_label][roi]['loco']]
            if vals:
                print(f'  {np.mean(vals):>5.1f}±{np.std(vals):>4.1f}', end='')
            else:
                print(f'  {"N/A":>12}', end='')
        print()


if __name__ == '__main__':
    main()
