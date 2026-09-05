#!/usr/bin/env python3
"""
Step D: Prior-Centred Ridge Fine-Tuning.

Nested CV for lambda selection:
  Outer: LORO (6-fold, hold-out 1 run)
    Inner: LORO on 5 remaining runs (5-fold)
      Evaluate each lambda -> pick best inner lambda per outer fold

  final_lambda = median of 6 outer best lambdas

Final W_s fit on all data with selected lambda.

Usage:
    python step_d_finetune.py \
        --subject 01 \
        --rois V1 V2 V3 V4 \
        --baseline_dir /path/to/full_dataset_C010 \
        --prior_dir results/subject_weights \
        --output_dir results/subject_weights
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/step_d_finetune.py  (development repository, commit 53c81c2)
# Role      : Step D (participant-level fine-tuning).
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
import numpy as np
from pathlib import Path

from utils_forward_model import (
    ROIS, N_RUNS, N_COLORS, N_CHANNELS,
    HUE_ANGLES, LAMBDA_GRID,
    load_amplitudes, create_basis_matrix, fit_W_prior_ridge,
    voxel_pattern_correlation, save_config, get_subject_group,
    DEFAULT_BASELINE_DIR,
)


def select_lambda_nested_cv(amp, W0, lambda_grid):
    """Nested LORO CV to select lambda for prior-centred ridge.

    Outer: 6-fold LORO (hold 1 run)
      Inner: 5-fold LORO on remaining runs
        For each lambda: train on 4 inner runs, evaluate on 1 inner run

    Args:
        amp: (6, 8, V_s) amplitudes
        W0: (K, V_s) prior weight
        lambda_grid: list of lambdas to test

    Returns:
        final_lambda: median of outer-fold best lambdas
        outer_best_lambdas: list of 6 best lambdas (one per outer fold)
        details: nested dict of scores
    """
    C = create_basis_matrix(HUE_ANGLES)  # (8, K)
    n_runs = amp.shape[0]

    outer_best_lambdas = []
    details = []

    for outer_run in range(n_runs):
        inner_runs = [r for r in range(n_runs) if r != outer_run]

        # Inner CV: LORO on inner_runs
        inner_scores = {lam: [] for lam in lambda_grid}

        for inner_val_idx, inner_val_run in enumerate(inner_runs):
            inner_train_runs = [r for r in inner_runs if r != inner_val_run]

            # Training data
            X_train = amp[inner_train_runs].reshape(-1, amp.shape[2])
            C_train = np.tile(C, (len(inner_train_runs), 1))

            # Validation data
            X_val = amp[inner_val_run]  # (8, V_s)

            for lam in lambda_grid:
                if lam == 0:
                    # OLS (no prior)
                    W = np.linalg.pinv(C_train) @ X_train
                else:
                    W = fit_W_prior_ridge(C_train, X_train, W0, lam)

                Y_hat = C @ W  # (8, V_s)
                r = voxel_pattern_correlation(Y_hat, X_val)
                inner_scores[lam].append(float(np.mean(r)))

        # Best inner lambda for this outer fold
        inner_means = {l: np.mean(s) for l, s in inner_scores.items()}
        best_lam = max(inner_means, key=inner_means.get)
        outer_best_lambdas.append(best_lam)

        details.append({
            'outer_run': outer_run,
            'best_lambda': best_lam,
            'inner_means': {str(k): v for k, v in inner_means.items()},
        })

    final_lambda = float(np.median(outer_best_lambdas))
    return final_lambda, outer_best_lambdas, details


def finetune_for_roi(subject, roi, baseline_dir, prior_dir, output_dir,
                     lambda_grid):
    """Fine-tune W_s for one subject x ROI.

    Returns:
        result: dict with lambda info and W shape
    """
    roi_prior = Path(prior_dir) / roi
    roi_out = Path(output_dir) / roi
    roi_out.mkdir(parents=True, exist_ok=True)

    # Load data
    amp = load_amplitudes(baseline_dir, subject, roi)  # (6, 8, V_s)
    W0 = np.load(roi_prior / f'sub-{subject}_W0.npy')   # (K, V_s)

    print(f'  amp: {amp.shape}, W0: {W0.shape}')

    # Nested CV for lambda
    final_lambda, outer_lambdas, details = select_lambda_nested_cv(
        amp, W0, lambda_grid)
    print(f'  Lambda selection: outer_best={outer_lambdas}, '
          f'final={final_lambda}')

    # Final fit on all data
    C = create_basis_matrix(HUE_ANGLES)  # (8, K)
    X_all = amp.reshape(-1, amp.shape[2])  # (48, V_s)
    C_all = np.tile(C, (N_RUNS, 1))       # (48, K)

    if final_lambda == 0:
        W_s = np.linalg.pinv(C_all) @ X_all
    else:
        W_s = fit_W_prior_ridge(C_all, X_all, W0, final_lambda)

    # Save
    np.save(roi_out / f'sub-{subject}_W.npy', W_s)

    # Diagnostics
    drift = np.linalg.norm(W_s - W0) / np.linalg.norm(W0)
    print(f'  W_s shape: {W_s.shape}, ||W-W0||/||W0|| = {drift:.4f}')

    result = {
        'final_lambda': final_lambda,
        'outer_best_lambdas': [float(l) for l in outer_lambdas],
        'W_shape': list(W_s.shape),
        'drift_from_prior': float(drift),
        'details': details,
    }

    with open(roi_out / f'sub-{subject}_lambda.json', 'w') as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Step D: Prior-Centred Ridge Fine-Tuning')
    parser.add_argument('--subject', type=str, required=True,
                        help='Subject ID (e.g., 01)')
    parser.add_argument('--rois', nargs='+', default=ROIS)
    parser.add_argument('--baseline_dir', type=str,
                        default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument('--prior_dir', type=str,
                        default='results/subject_weights')
    parser.add_argument('--output_dir', type=str,
                        default='results/subject_weights')
    parser.add_argument('--lambda_grid', nargs='+', type=float,
                        default=LAMBDA_GRID)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    subj = args.subject
    group = get_subject_group(subj)

    print('=' * 60)
    print(f'Step D: Fine-Tune W_s  (sub-{subj}, {group})')
    print(f'ROIs: {args.rois}')
    print(f'Lambda grid: {args.lambda_grid}')
    print('=' * 60)

    summary = {}
    for roi in args.rois:
        print(f'\n--- {roi} ---')
        try:
            result = finetune_for_roi(
                subj, roi, args.baseline_dir,
                args.prior_dir, args.output_dir, args.lambda_grid)
            summary[roi] = result
        except FileNotFoundError as e:
            print(f'  ERROR: {e}')
            continue

    save_config(output_dir,
                step='step_d_finetune',
                subject=subj,
                subject_group=group,
                rois=args.rois,
                baseline_dir=args.baseline_dir,
                prior_dir=args.prior_dir,
                lambda_grid=args.lambda_grid,
                summary=summary)

    print(f'\nStep D complete for sub-{subj}.')


if __name__ == '__main__':
    main()
