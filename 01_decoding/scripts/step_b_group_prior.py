#!/usr/bin/env python3
"""
Step B: Learn per-HC encoding A_i in SRM common space -> average to A_g.

For each HC subject:
  Z_i = R_i^T @ beta_i^T   (k, 8)  — common-space response
  A_i = Z_i @ C @ inv(C^T C + lambda_A * I)   (k, K) — common-space weight

A_g = mean(A_i)  — group prior in common space

Usage:
    python step_b_group_prior.py \
        --rois V1 V2 V3 V4 \
        --baseline_dir /path/to/full_dataset_C010 \
        --srm_dir results/srm_projections \
        --output_dir results/group_prior \
        --lambda_A 0
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/step_b_group_prior.py  (development repository, commit 53c81c2)
# Role      : Step B (group prior on channel weights).
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
import numpy as np
from pathlib import Path

from utils_forward_model import (
    HC_SUBJECTS, ROIS, K_VALUES, N_CHANNELS,
    HUE_ANGLES, load_amplitudes, create_basis_matrix, save_config,
    DEFAULT_BASELINE_DIR,
)


def build_group_prior_for_roi(roi, baseline_dir, srm_dir, output_dir,
                              lambda_A=0.0):
    """Build A_g for one ROI.

    Args:
        roi: ROI name
        baseline_dir: C010 data path
        srm_dir: directory with sub-{ID}_R.npy per ROI
        output_dir: where to save A_i and A_g
        lambda_A: regularisation for A_i fitting (0 = OLS)

    Returns:
        A_g: (k, K) group prior
        residuals: dict {subject: relative residual}
    """
    k = K_VALUES[roi]
    K = N_CHANNELS
    roi_srm = Path(srm_dir) / roi
    roi_out = Path(output_dir) / roi
    roi_out.mkdir(parents=True, exist_ok=True)

    # Design matrix for 8 hue conditions
    C = create_basis_matrix(HUE_ANGLES)  # (8, K)

    # Precompute: (C^T C + lambda_A * I)^{-1}
    CtC_reg_inv = np.linalg.inv(C.T @ C + lambda_A * np.eye(K))  # (K, K)

    A_list = []
    residuals = {}

    for subj in HC_SUBJECTS:
        # Load R_i
        R_i = np.load(roi_srm / f'sub-{subj}_R.npy')  # (V_s, k)

        # Run-averaged beta
        amp = load_amplitudes(baseline_dir, subj, roi)  # (6, 8, V_s)
        beta = amp.mean(axis=0)  # (8, V_s)

        # Project to common space: Z_i = R_i^T @ beta^T -> (k, 8)
        Z_i = R_i.T @ beta.T  # (k, 8)

        # Solve for A_i: A_i @ C^T = Z_i   ->   Z_i = A_i @ C^T
        # Ridge: A_i = Z_i @ C @ inv(C^T C + lambda_A * I)
        A_i = Z_i @ C @ CtC_reg_inv  # (k, K)

        # Residual check: ||Z_i - A_i @ C^T|| / ||Z_i||
        Z_hat = A_i @ C.T  # (k, 8)
        rel_residual = np.linalg.norm(Z_i - Z_hat) / np.linalg.norm(Z_i)
        residuals[subj] = float(rel_residual)

        np.save(roi_out / f'sub-{subj}_A.npy', A_i)
        A_list.append(A_i)

        print(f'  sub-{subj}: A_i shape={A_i.shape}, '
              f'residual={rel_residual:.4f}')

    # Group prior: simple mean
    A_g = np.mean(A_list, axis=0)  # (k, K)
    np.save(roi_out / 'A_g.npy', A_g)

    print(f'  A_g shape: {A_g.shape}')
    return A_g, residuals


def main():
    parser = argparse.ArgumentParser(
        description='Step B: Group Prior A_g in Common Space')
    parser.add_argument('--rois', nargs='+', default=ROIS)
    parser.add_argument('--baseline_dir', type=str,
                        default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument('--srm_dir', type=str,
                        default='results/srm_projections')
    parser.add_argument('--output_dir', type=str,
                        default='results/group_prior')
    parser.add_argument('--lambda_A', type=float, default=0.0,
                        help='Regularisation for A_i fitting (0=OLS)')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Step B: Group Prior Construction (A_g)')
    print(f'ROIs: {args.rois}')
    print(f'lambda_A: {args.lambda_A}')
    print('=' * 60)

    summary = {}
    for roi in args.rois:
        print(f'\n--- {roi} ---')
        A_g, residuals = build_group_prior_for_roi(
            roi, args.baseline_dir, args.srm_dir, args.output_dir,
            args.lambda_A)
        summary[roi] = {
            'A_g_shape': list(A_g.shape),
            'residuals': residuals,
            'mean_residual': float(np.mean(list(residuals.values()))),
        }

        # Flag if any subject has high residual
        for subj, res in residuals.items():
            if res > 0.3:
                print(f'  WARNING: sub-{subj} residual {res:.3f} > 0.3')

    save_config(output_dir,
                step='step_b_group_prior',
                rois=args.rois,
                baseline_dir=args.baseline_dir,
                srm_dir=args.srm_dir,
                lambda_A=args.lambda_A,
                hc_subjects=HC_SUBJECTS,
                summary=summary)

    print('\nStep B complete.')


if __name__ == '__main__':
    main()
