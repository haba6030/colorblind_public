#!/usr/bin/env python3
"""
Step A: Fit SRM on HC subjects -> R_i projection matrices per ROI.

Uses BrainIAK SRM with beta-based approach (run-averaged betas as input).
HC-only training; CVD subjects are projected later (step_c).

Usage:
    mpirun -np 1 python step_a_fit_srm.py \
        --rois V1 V2 V3 V4 \
        --baseline_dir /path/to/full_dataset_C010 \
        --output_dir results/srm_projections
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/step_a_fit_srm.py  (development repository, commit 53c81c2)
# Role      : Forward-model pipeline step A (control SRM).
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
import pickle
import numpy as np
from pathlib import Path

from utils_forward_model import (
    HC_SUBJECTS, ROIS, K_VALUES, N_RUNS,
    load_amplitudes, save_config, DEFAULT_BASELINE_DIR,
)

from brainiak.funcalign.srm import SRM


def fit_srm_for_roi(roi, baseline_dir, output_dir, n_iter=10):
    """Fit SRM on HC run-averaged betas for one ROI.

    Args:
        roi: ROI name
        baseline_dir: path to C010 data
        output_dir: where to save R_i, shared_response, model
        n_iter: SRM iterations

    Returns:
        k: number of features used
        voxel_counts: list of V_s per HC subject
    """
    k = K_VALUES[roi]
    roi_dir = Path(output_dir) / roi
    roi_dir.mkdir(parents=True, exist_ok=True)

    # Load HC betas: run-average -> (8, V_s) per subject
    srm_input = []
    voxel_counts = []
    for subj in HC_SUBJECTS:
        amp = load_amplitudes(baseline_dir, subj, roi)  # (6, 8, V_s)
        beta = amp.mean(axis=0)  # (8, V_s)
        srm_input.append(beta.T)  # BrainIAK expects (V_s, 8)
        voxel_counts.append(beta.shape[1])
        print(f'  sub-{subj}: V_s={beta.shape[1]}')

    # Fit SRM
    print(f'  Fitting SRM: k={k}, n_iter={n_iter}, '
          f'n_subjects={len(HC_SUBJECTS)}')
    srm = SRM(n_iter=n_iter, features=k)
    srm.fit(srm_input)

    # Save R_i for each HC subject
    for i, subj in enumerate(HC_SUBJECTS):
        R_i = srm.w_[i]  # (V_s, k)
        np.save(roi_dir / f'sub-{subj}_R.npy', R_i)

        # Verify near-orthonormality
        orth_check = R_i.T @ R_i
        diag_ok = np.allclose(np.diag(orth_check), 1.0, atol=0.05)
        offdiag = orth_check - np.diag(np.diag(orth_check))
        offdiag_ok = np.allclose(offdiag, 0.0, atol=0.05)
        if not (diag_ok and offdiag_ok):
            print(f'  WARNING: sub-{subj} R not orthonormal '
                  f'(diag={np.diag(orth_check)}, '
                  f'max_offdiag={np.abs(offdiag).max():.4f})')

    # Save shared response and model
    np.save(roi_dir / 'shared_response.npy', srm.s_)
    with open(roi_dir / 'srm_model.pkl', 'wb') as f:
        pickle.dump(srm, f)

    print(f'  Shared response shape: {srm.s_.shape}')  # (k, 8)
    return k, voxel_counts


def main():
    parser = argparse.ArgumentParser(description='Step A: Fit SRM on HC')
    parser.add_argument('--rois', nargs='+', default=ROIS)
    parser.add_argument('--baseline_dir', type=str,
                        default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument('--output_dir', type=str,
                        default='results/srm_projections')
    parser.add_argument('--n_iter', type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Step A: SRM Fitting (HC-only)')
    print(f'ROIs: {args.rois}')
    print(f'Baseline dir: {args.baseline_dir}')
    print(f'Output dir: {args.output_dir}')
    print('=' * 60)

    summary = {}
    for roi in args.rois:
        print(f'\n--- {roi} ---')
        k, vcounts = fit_srm_for_roi(
            roi, args.baseline_dir, args.output_dir, args.n_iter)
        summary[roi] = {
            'k': k,
            'voxel_counts': {s: v for s, v in zip(HC_SUBJECTS, vcounts)},
        }

    save_config(output_dir,
                step='step_a_fit_srm',
                rois=args.rois,
                baseline_dir=args.baseline_dir,
                n_iter=args.n_iter,
                k_values={r: K_VALUES[r] for r in args.rois},
                hc_subjects=HC_SUBJECTS,
                summary=summary)

    print('\nStep A complete.')


if __name__ == '__main__':
    main()
