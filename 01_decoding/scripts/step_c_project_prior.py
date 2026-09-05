#!/usr/bin/env python3
"""
Step C: Project group prior A_g into each subject's voxel space.

  W_{0,s} = (R_s @ A_g)^T   -->  shape (K, V_s)

For HC: R_s from step_a (saved as sub-{ID}_R.npy).
For CVD: R_s via SVD projection of new subject into SRM space.

Usage:
    mpirun -np 1 python step_c_project_prior.py \
        --subjects 01 02 03 04 05 06 07 08 09 10 \
        --rois V1 V2 V3 V4 \
        --baseline_dir /path/to/full_dataset_C010 \
        --srm_dir results/srm_projections \
        --prior_dir results/group_prior \
        --output_dir results/subject_weights
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/step_c_project_prior.py  (development repository, commit 53c81c2)
# Role      : Step C (project prior to each participant).
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
    HC_SUBJECTS, CVD_SUBJECTS, ALL_SUBJECTS, ROIS, N_CHANNELS,
    load_amplitudes, project_new_subject, save_config,
    DEFAULT_BASELINE_DIR,
)


def project_prior_for_roi(roi, subjects, baseline_dir, srm_dir, prior_dir,
                          output_dir):
    """Project A_g to W0 for each subject in one ROI.

    Args:
        roi: ROI name
        subjects: list of subject IDs
        baseline_dir, srm_dir, prior_dir, output_dir: paths

    Returns:
        shapes: dict {subject: W0_shape}
    """
    roi_srm = Path(srm_dir) / roi
    roi_prior = Path(prior_dir) / roi
    roi_out = Path(output_dir) / roi
    roi_out.mkdir(parents=True, exist_ok=True)

    # Load A_g
    A_g = np.load(roi_prior / 'A_g.npy')  # (k, K)

    # Load SRM model (needed for CVD projection)
    srm_model = None
    srm_pkl = roi_srm / 'srm_model.pkl'
    if srm_pkl.exists():
        with open(srm_pkl, 'rb') as f:
            srm_model = pickle.load(f)

    shapes = {}

    for subj in subjects:
        # Get R_s
        if subj in HC_SUBJECTS:
            R_s = np.load(roi_srm / f'sub-{subj}_R.npy')  # (V_s, k)
        else:
            # CVD: project into SRM via SVD
            if srm_model is None:
                print(f'  sub-{subj}: SKIP — no SRM model for CVD projection')
                continue
            amp = load_amplitudes(baseline_dir, subj, roi)  # (6, 8, V_s)
            beta = amp.mean(axis=0)  # (8, V_s)
            R_s = project_new_subject(srm_model, beta.T)  # (V_s, k)

        # W0 = (R_s @ A_g)^T  ->  (K, V_s)
        W0 = (R_s @ A_g).T  # (V_s, k) @ (k, K) = (V_s, K) -> T -> (K, V_s)

        np.save(roi_out / f'sub-{subj}_W0.npy', W0)
        shapes[subj] = list(W0.shape)

        print(f'  sub-{subj}: W0 {W0.shape}  '
              f'(R_s {R_s.shape}, A_g {A_g.shape})')

    return shapes


def main():
    parser = argparse.ArgumentParser(
        description='Step C: Project Group Prior to Subject Space')
    parser.add_argument('--subjects', nargs='+', default=ALL_SUBJECTS)
    parser.add_argument('--rois', nargs='+', default=ROIS)
    parser.add_argument('--baseline_dir', type=str,
                        default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument('--srm_dir', type=str,
                        default='results/srm_projections')
    parser.add_argument('--prior_dir', type=str,
                        default='results/group_prior')
    parser.add_argument('--output_dir', type=str,
                        default='results/subject_weights')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Step C: Project Prior -> W_{0,s}')
    print(f'Subjects: {args.subjects}')
    print(f'ROIs: {args.rois}')
    print('=' * 60)

    summary = {}
    for roi in args.rois:
        print(f'\n--- {roi} ---')
        shapes = project_prior_for_roi(
            roi, args.subjects, args.baseline_dir,
            args.srm_dir, args.prior_dir, args.output_dir)
        summary[roi] = shapes

    save_config(output_dir,
                step='step_c_project_prior',
                subjects=args.subjects,
                rois=args.rois,
                baseline_dir=args.baseline_dir,
                srm_dir=args.srm_dir,
                prior_dir=args.prior_dir,
                summary=summary)

    print('\nStep C complete.')


if __name__ == '__main__':
    main()
