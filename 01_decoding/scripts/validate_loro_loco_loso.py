#!/usr/bin/env python3
"""
Validation: LORO + LOCO evaluation of 4 models.

Models:
  ols           — pinv(C) @ X, no regularisation
  ridge_gcv     — (C'C + alpha*I)^{-1} C'X, alpha via GCV per fold
  prior_only    — W = W0 (zero-shot, no fitting)
  prior_finetune — (C'C + lam*I)^{-1} (C'X + lam*W0), lam via inner CV

Evaluations:
  LORO  — hold out 1 run   (6-fold)
  LOCO  — hold out 1 color (8-fold)

Metrics per fold:
  voxel_corr, R2, rdm_corr, mae (LOCO only)

Reliability (once per subject x ROI):
  split-half RDM correlation, RDM noise ceiling

Usage:
    python validate_loro_loco_loso.py \
        --subject 01 \
        --rois V1 V2 V3 V4 \
        --baseline_dir /path/to/full_dataset_C010 \
        --prior_dir results/subject_weights \
        --output_dir results/validation
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/validate_loro_loco_loso.py  (development repository, commit 53c81c2)
# Role      : LORO / LOCO / LOSO validation of the ridge-GCV encoder (S7).
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
import pickle
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

from utils_forward_model import (
    ROIS, N_RUNS, N_COLORS, N_CHANNELS, ALPHA_GRID, LAMBDA_GRID,
    HUE_ANGLES, HC_SUBJECTS, CVD_SUBJECTS,
    load_amplitudes, create_basis_matrix, create_basis_full,
    fit_W_ols, fit_W_ridge, fit_W_prior_ridge,
    predict_patterns, decode_hue, circular_distance,
    voxel_pattern_correlation, explained_variance,
    compute_rdm, rdm_upper_tri, gcv_select_alpha,
    save_config, get_subject_group, project_new_subject,
    DEFAULT_BASELINE_DIR,
)


# ============================================================================
# Fold-level evaluation helpers
# ============================================================================

def evaluate_fold_loro(W, C_test, X_test):
    """Evaluate a single LORO fold (run prediction).

    Args:
        W: (K, V_s) encoding weights
        C_test: (8, K) design matrix for 8 conditions
        X_test: (8, V_s) actual patterns

    Returns:
        dict with voxel_corr, R2, rdm_corr
    """
    Y_hat = predict_patterns(W, C_test)  # (8, V_s)
    r = voxel_pattern_correlation(Y_hat, X_test)
    r2 = explained_variance(Y_hat, X_test)

    rdm_pred = compute_rdm(Y_hat)
    rdm_real = compute_rdm(X_test)
    rdm_r, _ = spearmanr(rdm_upper_tri(rdm_pred), rdm_upper_tri(rdm_real))
    if not np.isfinite(rdm_r):
        rdm_r = 0.0

    return {
        'voxel_corr': float(np.mean(r)),
        'voxel_corr_per_cond': [float(x) for x in r],
        'R2': float(np.mean(r2)),
        'rdm_corr': float(rdm_r),
    }


def evaluate_fold_loco(W, basis_full, C_test, X_test, true_hue):
    """Evaluate a single LOCO fold (colour interpolation).

    Args:
        W: (K, V_s) encoding weights
        basis_full: (360, K) full basis
        C_test: (1, K) design matrix for held-out colour
        X_test: (n_runs, V_s) held-out colour patterns (all runs)
        true_hue: scalar, true hue angle

    Returns:
        dict with voxel_corr, R2, mae, per-run details
    """
    # Pattern prediction: mean across runs for voxel correlation
    Y_hat_mean = predict_patterns(W, C_test)  # (1, V_s)
    X_mean = X_test.mean(axis=0, keepdims=True)  # (1, V_s)
    r = voxel_pattern_correlation(Y_hat_mean, X_mean)
    r2 = explained_variance(Y_hat_mean, X_mean)

    # Hue decoding per run
    pred_hues = decode_hue(W, basis_full, X_test)  # (n_runs,)
    errors = circular_distance(np.full(len(pred_hues), true_hue), pred_hues)

    return {
        'voxel_corr': float(np.mean(r)),
        'R2': float(np.mean(r2)),
        'mae': float(np.nanmean(errors)),
        'pred_hues': [float(h) for h in pred_hues],
        'errors': [float(e) for e in errors],
    }


# ============================================================================
# Model fitting per fold
# ============================================================================

def fit_model_loro(model_name, amp, W0, train_runs, lambda_grid,
                   n_channels=N_CHANNELS, basis_type='fe'):
    """Fit one model on LORO training runs.

    Args:
        model_name: 'ols', 'ridge_gcv', 'prior_only', 'prior_finetune'
        amp: (6, 8, V_s) full data
        W0: (K, V_s) prior weight (or None)
        train_runs: list of run indices
        lambda_grid: for prior_finetune inner CV
        n_channels: number of basis channels
        basis_type: 'fe' or 'lf'

    Returns:
        W: (K, V_s) encoding weights
        meta: dict of model-specific info
    """
    C = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)  # (8, K)
    V_s = amp.shape[2]
    X_train = amp[train_runs].reshape(-1, V_s)  # (n_train_runs*8, V_s)
    C_train = np.tile(C, (len(train_runs), 1))   # (n_train_runs*8, K)

    if model_name == 'ols':
        W = fit_W_ols(C_train, X_train)
        return W, {}

    elif model_name == 'ridge_gcv':
        best_alpha, gcv_scores = gcv_select_alpha(C_train, X_train)
        W = fit_W_ridge(C_train, X_train, best_alpha)
        return W, {'alpha': float(best_alpha)}

    elif model_name == 'prior_only':
        return W0.copy(), {}

    elif model_name == 'prior_finetune':
        # Inner LORO on training runs for lambda selection
        best_lam = _inner_cv_lambda(amp, W0, train_runs, lambda_grid,
                                    n_channels=n_channels, basis_type=basis_type)
        if best_lam == 0:
            W = fit_W_ols(C_train, X_train)
        else:
            W = fit_W_prior_ridge(C_train, X_train, W0, best_lam)
        return W, {'lambda': float(best_lam)}

    else:
        raise ValueError(f'Unknown model: {model_name}')


def fit_model_loco(model_name, amp, W0, train_colors, lambda_grid,
                   n_channels=N_CHANNELS, basis_type='fe'):
    """Fit one model on LOCO training colours.

    Args:
        model_name: model identifier
        amp: (6, 8, V_s)
        W0: (K, V_s) or None
        train_colors: list of colour indices
        lambda_grid: for prior_finetune
        n_channels: number of basis channels
        basis_type: 'fe' or 'lf'

    Returns:
        W, meta
    """
    C_full = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)
    C_train_row = C_full[train_colors]  # (7, K)
    V_s = amp.shape[2]

    # Pool all runs for training colours
    X_train = amp[:, train_colors, :].reshape(-1, V_s)  # (6*7, V_s)
    C_train = np.tile(C_train_row, (N_RUNS, 1))          # (42, K)

    if model_name == 'ols':
        W = fit_W_ols(C_train, X_train)
        return W, {}

    elif model_name == 'ridge_gcv':
        best_alpha, _ = gcv_select_alpha(C_train, X_train)
        W = fit_W_ridge(C_train, X_train, best_alpha)
        return W, {'alpha': float(best_alpha)}

    elif model_name == 'prior_only':
        return W0.copy(), {}

    elif model_name == 'prior_finetune':
        # Inner LORO for lambda on training colors
        best_lam = _inner_cv_lambda_loco(
            amp, W0, train_colors, lambda_grid,
            n_channels=n_channels, basis_type=basis_type)
        if best_lam == 0:
            W = fit_W_ols(C_train, X_train)
        else:
            W = fit_W_prior_ridge(C_train, X_train, W0, best_lam)
        return W, {'lambda': float(best_lam)}

    else:
        raise ValueError(f'Unknown model: {model_name}')


def _inner_cv_lambda(amp, W0, outer_train_runs, lambda_grid,
                     n_channels=N_CHANNELS, basis_type='fe'):
    """Inner LORO (over runs) for lambda selection."""
    C = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)
    inner_scores = {lam: [] for lam in lambda_grid}

    for val_run in outer_train_runs:
        inner_train = [r for r in outer_train_runs if r != val_run]
        X_tr = amp[inner_train].reshape(-1, amp.shape[2])
        C_tr = np.tile(C, (len(inner_train), 1))
        X_val = amp[val_run]  # (8, V_s)

        for lam in lambda_grid:
            if lam == 0:
                W = fit_W_ols(C_tr, X_tr)
            else:
                W = fit_W_prior_ridge(C_tr, X_tr, W0, lam)
            Y_hat = predict_patterns(W, C)
            r = voxel_pattern_correlation(Y_hat, X_val)
            inner_scores[lam].append(float(np.mean(r)))

    inner_means = {l: np.mean(s) for l, s in inner_scores.items()}
    return max(inner_means, key=inner_means.get)


def _inner_cv_lambda_loco(amp, W0, train_colors, lambda_grid,
                          n_channels=N_CHANNELS, basis_type='fe'):
    """Inner LORO (over runs) for lambda in LOCO context."""
    C_full = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)
    C_train_row = C_full[train_colors]
    inner_scores = {lam: [] for lam in lambda_grid}

    for val_run in range(N_RUNS):
        inner_train = [r for r in range(N_RUNS) if r != val_run]
        X_tr = amp[inner_train][:, train_colors, :].reshape(-1, amp.shape[2])
        C_tr = np.tile(C_train_row, (len(inner_train), 1))
        X_val = amp[val_run, train_colors, :]  # (7, V_s)

        for lam in lambda_grid:
            if lam == 0:
                W = fit_W_ols(C_tr, X_tr)
            else:
                W = fit_W_prior_ridge(C_tr, X_tr, W0, lam)
            Y_hat = predict_patterns(W, C_train_row)
            r = voxel_pattern_correlation(Y_hat, X_val)
            inner_scores[lam].append(float(np.mean(r)))

    inner_means = {l: np.mean(s) for l, s in inner_scores.items()}
    return max(inner_means, key=inner_means.get)


# ============================================================================
# LOCO-safe prior reconstruction (prevents data leakage)
# ============================================================================

def recompute_W0_excluding_colors(
    subj, roi, train_colors, baseline_dir, srm_dir, lambda_A=0.0,
):
    """Recompute W0 from HC group prior excluding held-out color(s).

    Rebuilds A_g from scratch using only train_colors, then projects
    to the target subject's voxel space.  SRM projection matrices R_i
    are NOT refit (they capture subspace geometry, not color-specific
    encoding).

    Args:
        subj: target subject ID (e.g. '01')
        roi: ROI name
        train_colors: list of colour indices included in training
        baseline_dir: path to C010 data
        srm_dir: directory with sub-{ID}_R.npy and srm_model.pkl
        lambda_A: regularisation for A_i fitting (0 = OLS, matching step_b)

    Returns:
        W0: (K, V_s) prior weight matrix built without held-out colors
    """
    roi_srm = Path(srm_dir) / roi
    C_full = create_basis_matrix(HUE_ANGLES)       # (8, K)
    C_train = C_full[train_colors]                  # (n_train, K)
    K = C_full.shape[1]

    # Precompute: (C_train^T C_train + lambda_A * I)^{-1}
    CtC_inv = np.linalg.inv(C_train.T @ C_train + lambda_A * np.eye(K))

    # --- Rebuild A_i per HC subject using only train_colors ---
    A_list = []
    for hc in HC_SUBJECTS:
        R_i = np.load(roi_srm / f'sub-{hc}_R.npy')          # (V_i, k)
        amp_i = load_amplitudes(baseline_dir, hc, roi)       # (6, 8, V_i)
        beta_i = amp_i.mean(axis=0)                          # (8, V_i)
        Z_i = R_i.T @ beta_i.T                               # (k, 8)
        Z_train = Z_i[:, train_colors]                        # (k, n_train)
        A_i = Z_train @ C_train @ CtC_inv                    # (k, K)
        A_list.append(A_i)

    A_g = np.mean(A_list, axis=0)                             # (k, K)

    # --- Get R_s for target subject ---
    if subj in HC_SUBJECTS:
        R_s = np.load(roi_srm / f'sub-{subj}_R.npy')
    else:
        with open(roi_srm / 'srm_model.pkl', 'rb') as f:
            srm_model = pickle.load(f)
        amp_s = load_amplitudes(baseline_dir, subj, roi)
        beta_s = amp_s.mean(axis=0)                           # (8, V_s)
        R_s = project_new_subject(srm_model, beta_s.T)        # (V_s, k)

    W0 = (R_s @ A_g).T                                        # (K, V_s)
    return W0


# ============================================================================
# Reliability metrics
# ============================================================================

def compute_reliability(amp):
    """Split-half and noise-ceiling metrics.

    Args:
        amp: (6, 8, V_s)

    Returns:
        dict with split_half_rdm, noise_ceiling_upper, noise_ceiling_lower
    """
    odd = amp[[0, 2, 4]].mean(axis=0)  # (8, V_s)
    even = amp[[1, 3, 5]].mean(axis=0)

    rdm_odd = compute_rdm(odd)
    rdm_even = compute_rdm(even)
    sh_r, _ = spearmanr(rdm_upper_tri(rdm_odd), rdm_upper_tri(rdm_even))
    if not np.isfinite(sh_r):
        sh_r = 0.0

    # Noise ceiling: upper = mean(corr(single_run_rdm, group_mean_rdm))
    grand_mean = amp.mean(axis=0)  # (8, V_s)
    rdm_grand = compute_rdm(grand_mean)
    vec_grand = rdm_upper_tri(rdm_grand)

    upper_rs = []
    for run in range(N_RUNS):
        rdm_run = compute_rdm(amp[run])
        r, _ = spearmanr(rdm_upper_tri(rdm_run), vec_grand)
        if np.isfinite(r):
            upper_rs.append(r)
    noise_upper = float(np.mean(upper_rs)) if upper_rs else 0.0

    # Lower: corr(LOO_mean_rdm, full_mean_rdm)
    lower_rs = []
    for run in range(N_RUNS):
        loo_runs = [r for r in range(N_RUNS) if r != run]
        loo_mean = amp[loo_runs].mean(axis=0)
        rdm_loo = compute_rdm(loo_mean)
        r, _ = spearmanr(rdm_upper_tri(rdm_loo), vec_grand)
        if np.isfinite(r):
            lower_rs.append(r)
    noise_lower = float(np.mean(lower_rs)) if lower_rs else 0.0

    return {
        'split_half_rdm': float(sh_r),
        'noise_ceiling_upper': noise_lower,   # LOO mean → higher (upper bound)
        'noise_ceiling_lower': noise_upper,   # single run → lower (lower bound)
    }


# ============================================================================
# Main evaluation loops
# ============================================================================

MODELS = ['ols', 'ridge_gcv', 'prior_only', 'prior_finetune']


def run_loro(amp, W0, lambda_grid, n_channels=N_CHANNELS, basis_type='fe',
             models=None):
    """LORO evaluation for specified models.

    Returns:
        results: {model_name: {mean_voxel_corr, mean_R2, mean_rdm_corr, folds}}
    """
    if models is None:
        models = MODELS
    C = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)
    basis_full = create_basis_full(n_channels, basis_type)
    results = {}

    for model in models:
        folds = []
        for test_run in range(N_RUNS):
            train_runs = [r for r in range(N_RUNS) if r != test_run]
            W, meta = fit_model_loro(model, amp, W0, train_runs, lambda_grid,
                                    n_channels=n_channels, basis_type=basis_type)
            X_test = amp[test_run]  # (8, V_s)
            fold_result = evaluate_fold_loro(W, C, X_test)
            fold_result.update(meta)
            fold_result['test_run'] = test_run
            folds.append(fold_result)

        results[model] = {
            'mean_voxel_corr': float(np.mean([f['voxel_corr'] for f in folds])),
            'mean_R2': float(np.mean([f['R2'] for f in folds])),
            'mean_rdm_corr': float(np.mean([f['rdm_corr'] for f in folds])),
            'folds': folds,
        }

    return results


def run_loco(amp, W0_full, lambda_grid, subj=None, roi=None,
             baseline_dir=None, srm_dir=None,
             n_channels=N_CHANNELS, basis_type='fe', models=None):
    """LOCO evaluation for specified models.

    For prior_only and prior_finetune, W0 is recomputed per fold
    excluding the held-out colour to prevent data leakage.

    Args:
        amp: (6, 8, V_s) amplitude data
        W0_full: (K, V_s) prior from all 8 colours (used for LORO only)
        lambda_grid: lambda candidates for prior_finetune
        subj: subject ID (needed for per-fold W0 recomputation)
        roi: ROI name (needed for per-fold W0 recomputation)
        baseline_dir: path to C010 data
        srm_dir: path to SRM projections
        n_channels: number of basis channels
        basis_type: 'fe' or 'lf'
        models: list of model names to evaluate

    Returns:
        results: {model_name: {mean_voxel_corr, mean_mae, folds}}
    """
    if models is None:
        models = MODELS
    C = create_basis_matrix(HUE_ANGLES, n_channels, basis_type)
    basis_full = create_basis_full(n_channels, basis_type)
    results = {}

    # Can we do leakage-free LOCO for prior models?
    can_recompute = (subj is not None and roi is not None
                     and baseline_dir is not None and srm_dir is not None)

    for model in models:
        folds = []
        for test_color in range(N_COLORS):
            train_colors = [c for c in range(N_COLORS) if c != test_color]

            # Per-fold W0: exclude held-out colour from group prior
            if model in ('prior_only', 'prior_finetune') and can_recompute:
                W0_fold = recompute_W0_excluding_colors(
                    subj, roi, train_colors, baseline_dir, srm_dir)
            else:
                W0_fold = W0_full

            W, meta = fit_model_loco(
                model, amp, W0_fold, train_colors, lambda_grid,
                n_channels=n_channels, basis_type=basis_type)

            C_test = C[test_color:test_color + 1]  # (1, K)
            X_test = amp[:, test_color, :]  # (6, V_s)
            true_hue = HUE_ANGLES[test_color]

            fold_result = evaluate_fold_loco(
                W, basis_full, C_test, X_test, true_hue)
            fold_result.update(meta)
            fold_result['test_color'] = int(test_color)
            fold_result['true_hue'] = int(true_hue)
            folds.append(fold_result)

        results[model] = {
            'mean_voxel_corr': float(np.mean([f['voxel_corr'] for f in folds])),
            'mean_R2': float(np.mean([f['R2'] for f in folds])),
            'mean_mae': float(np.mean([f['mae'] for f in folds])),
            'folds': folds,
        }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Validation: LORO + LOCO for 4 models')
    parser.add_argument('--subject', type=str, required=True)
    parser.add_argument('--rois', nargs='+', default=ROIS)
    parser.add_argument('--baseline_dir', type=str,
                        default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument('--prior_dir', type=str,
                        default='results/subject_weights')
    parser.add_argument('--srm_dir', type=str,
                        default='results/srm_projections')
    parser.add_argument('--output_dir', type=str,
                        default='results/validation')
    parser.add_argument('--lambda_grid', nargs='+', type=float,
                        default=LAMBDA_GRID)
    parser.add_argument('--basis_type', type=str, default='fe',
                        choices=['fe', 'lf'],
                        help='Basis type: fe (half-wave cos^2) or lf (Fourier)')
    parser.add_argument('--n_channels', type=int, default=N_CHANNELS,
                        help='Number of basis channels')
    parser.add_argument('--models', nargs='+', default=None,
                        help='Models to evaluate (default: all 4)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subj = args.subject
    group = get_subject_group(subj)

    # Determine active models
    active_models = args.models if args.models else list(MODELS)

    # Safety: prior models are incompatible with non-FE bases (W0 shape mismatch)
    if args.basis_type != 'fe':
        removed = [m for m in active_models
                   if m in ('prior_only', 'prior_finetune')]
        if removed:
            print(f'NOTE: Removing {removed} (incompatible with '
                  f'basis_type={args.basis_type})')
            active_models = [m for m in active_models
                             if m not in ('prior_only', 'prior_finetune')]

    print('=' * 60)
    print(f'Validation: sub-{subj} ({group})')
    print(f'ROIs: {args.rois}')
    print(f'Basis: {args.basis_type}-{args.n_channels}')
    print(f'Models: {active_models}')
    print('=' * 60)

    loro_all = {}
    loco_all = {}
    loco_leaked_all = {}
    reliability_all = {}

    for roi in args.rois:
        print(f'\n{"="*40} {roi} {"="*40}')

        try:
            amp = load_amplitudes(args.baseline_dir, subj, roi)
        except FileNotFoundError as e:
            print(f'  ERROR: {e}')
            continue

        # Load W0 (prior)
        W0_path = Path(args.prior_dir) / roi / f'sub-{subj}_W0.npy'
        if W0_path.exists():
            W0 = np.load(W0_path)
        else:
            print(f'  WARNING: W0 not found at {W0_path}, '
                  f'prior models will be skipped')
            W0 = None

        V_s = amp.shape[2]
        if V_s < 20:
            print(f'  WARNING: V_s={V_s} < 20 voxels — results may be noisy')

        # Reliability
        print(f'\n  --- Reliability ---')
        rel = compute_reliability(amp)
        reliability_all[roi] = rel
        print(f'  split-half RDM: {rel["split_half_rdm"]:.3f}')
        print(f'  noise ceiling: [{rel["noise_ceiling_lower"]:.3f}, '
              f'{rel["noise_ceiling_upper"]:.3f}]')

        # LORO
        print(f'\n  --- LORO (6-fold) ---')
        loro = run_loro(amp, W0, args.lambda_grid,
                        n_channels=args.n_channels, basis_type=args.basis_type,
                        models=active_models)
        loro_all[roi] = loro
        for m, res in loro.items():
            print(f'  {m:18s}: voxel_corr={res["mean_voxel_corr"]:.3f}  '
                  f'R2={res["mean_R2"]:.3f}  rdm_corr={res["mean_rdm_corr"]:.3f}')

        # LOCO — leaked (W0 from all 8 colors, for leakage quantification)
        # Only run leaked comparison when prior models are active
        has_prior_models = any(m in ('prior_only', 'prior_finetune')
                               for m in active_models)
        if has_prior_models:
            print(f'\n  --- LOCO (8-fold, LEAKED prior — diagnostic) ---')
            loco_leaked = run_loco(amp, W0, args.lambda_grid,
                                   n_channels=args.n_channels,
                                   basis_type=args.basis_type,
                                   models=active_models)
            loco_leaked_all[roi] = loco_leaked
            for m, res in loco_leaked.items():
                print(f'  {m:18s}: voxel_corr={res["mean_voxel_corr"]:.3f}  '
                      f'MAE={res["mean_mae"]:.1f}  R2={res["mean_R2"]:.3f}')

        # LOCO — clean (per-fold W0 recomputation, no leakage)
        print(f'\n  --- LOCO (8-fold, CLEAN prior) ---')
        loco = run_loco(amp, W0, args.lambda_grid,
                        subj=subj, roi=roi,
                        baseline_dir=args.baseline_dir,
                        srm_dir=args.srm_dir,
                        n_channels=args.n_channels,
                        basis_type=args.basis_type,
                        models=active_models)
        loco_all[roi] = loco
        for m, res in loco.items():
            print(f'  {m:18s}: voxel_corr={res["mean_voxel_corr"]:.3f}  '
                  f'MAE={res["mean_mae"]:.1f}  R2={res["mean_R2"]:.3f}')

        # Leakage delta for prior models
        for m in ('prior_only', 'prior_finetune'):
            if (roi in loco_leaked_all
                    and m in loco_leaked_all[roi] and m in loco):
                d_r = (loco_leaked_all[roi][m]['mean_voxel_corr']
                       - loco[m]['mean_voxel_corr'])
                d_mae = (loco_leaked_all[roi][m]['mean_mae']
                         - loco[m]['mean_mae'])
                print(f'  LEAKAGE {m:18s}: '
                      f'delta_r={d_r:+.4f}  delta_MAE={d_mae:+.1f}')

    # Save LORO results
    loro_output = {
        'subject': subj,
        'subject_group': group,
    }
    for roi in loro_all:
        loro_output[roi] = loro_all[roi]
        loro_output[roi]['reliability'] = reliability_all.get(roi, {})

    with open(output_dir / f'sub-{subj}_loro.json', 'w') as f:
        json.dump(loro_output, f, indent=2)

    # Save LOCO results (clean — leakage-free)
    loco_output = {
        'subject': subj,
        'subject_group': group,
    }
    for roi in loco_all:
        loco_output[roi] = loco_all[roi]
        loco_output[roi]['reliability'] = reliability_all.get(roi, {})

    with open(output_dir / f'sub-{subj}_loco.json', 'w') as f:
        json.dump(loco_output, f, indent=2)

    # Save LOCO results (leaked — for leakage quantification)
    loco_leaked_output = {
        'subject': subj,
        'subject_group': group,
    }
    for roi in loco_leaked_all:
        loco_leaked_output[roi] = loco_leaked_all[roi]

    with open(output_dir / f'sub-{subj}_loco_leaked.json', 'w') as f:
        json.dump(loco_leaked_output, f, indent=2)

    # Save reliability separately
    reliability_output = {
        'subject': subj,
        'subject_group': group,
    }
    reliability_output.update(reliability_all)
    with open(output_dir / f'sub-{subj}_reliability.json', 'w') as f:
        json.dump(reliability_output, f, indent=2)

    save_config(output_dir,
                step='validate_loro_loco',
                subject=subj,
                subject_group=group,
                rois=args.rois,
                basis_type=args.basis_type,
                n_channels=args.n_channels,
                baseline_dir=args.baseline_dir,
                prior_dir=args.prior_dir,
                srm_dir=args.srm_dir,
                models=active_models,
                lambda_grid=args.lambda_grid,
                loco_leakage_free=True,
                loco_leaked_saved=bool(loco_leaked_all))

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for roi in args.rois:
        if roi not in loco_all:
            continue
        print(f'\n{roi}:')
        print(f'  Reliability: sh_rdm={reliability_all[roi]["split_half_rdm"]:.3f}')
        for m in active_models:
            if m in loro_all.get(roi, {}):
                lr = loro_all[roi][m]
                lc = loco_all[roi][m]
                line = (f'  {m:18s}: LORO_r={lr["mean_voxel_corr"]:.3f}  '
                        f'LOCO_r={lc["mean_voxel_corr"]:.3f}  '
                        f'MAE={lc["mean_mae"]:.1f}')
                # Show leakage delta for prior models
                if m in ('prior_only', 'prior_finetune'):
                    lk = loco_leaked_all.get(roi, {}).get(m, {})
                    if lk:
                        d = lk['mean_voxel_corr'] - lc['mean_voxel_corr']
                        line += f'  (leak delta_r={d:+.4f})'
                print(line)

    # GO/NO-GO gate: LOCO voxel_corr > 0 (clean results only)
    # Use prior_finetune if available, otherwise best available model
    gate_model = ('prior_finetune' if 'prior_finetune' in active_models
                  else active_models[-1])
    print(f'\n--- GO/NO-GO GATE (clean LOCO, model={gate_model}) ---')
    for roi in args.rois:
        if roi not in loco_all:
            continue
        gm = loco_all[roi].get(gate_model, {})
        corr = gm.get('mean_voxel_corr', 0)
        status = 'GO' if corr > 0 else 'NO-GO'
        print(f'  {roi}: {gate_model} LOCO voxel_corr = {corr:.3f} -> {status}')


if __name__ == '__main__':
    main()
