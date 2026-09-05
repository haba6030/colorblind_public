#!/usr/bin/env python3
"""
Test 2C: Cross-Validation for Optimal k Selection

Purpose: Select optimal SRM dimensionality k via LOSO CV.
For each fold (leave-one-HC-out) × ROI, test k ∈ {2, 3, 4, 5, 6}.
Metric: reconstruction error of left-out subject + RDM reliability.

Usage:
    python run_k_selection_cv.py --fold 0 --roi V1
    python run_k_selection_cv.py --fold 0  # all ROIs
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase2_SRM_across_between/validation/2C_optimal_k_selection/run_k_selection_cv.py  (development repository, commit 53c81c2)
# Role      : LOSO cross-validation over k = 2..6 (S4).
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


import sys
import argparse
import json
import time
import socket
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes

try:
    from brainiak.funcalign.srm import SRM
except ImportError:
    print("ERROR: brainiak not installed")
    sys.exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

HC_SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROIS = ['V1', 'V2', 'V3', 'hV4']
ROI_DIR_MAP = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'hV4': 'V4'}
K_CANDIDATES = [2, 3, 4, 5, 6]

if socket.gethostname().startswith('node'):
    DATA_DIR = Path("/scratch/connectome/haba6030/colorBlind/derivatives/full_dataset_C010")
    OUTPUT_BASE = Path("/scratch/connectome/haba6030/colorBlind/analysis/phase2_SRM_across_between/validation/2C_optimal_k_selection/results")
else:
    DATA_DIR = Path("/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/colorBlind_analysis/analysis/phase1_preprocess_decoding/results/full_dataset_C010")
    OUTPUT_BASE = Path("/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/colorBlind_analysis/analysis/phase2_SRM_across_between/validation/2C_optimal_k_selection/results")


def load_amplitudes(subject, roi):
    roi_dir = ROI_DIR_MAP.get(roi, roi)
    path = DATA_DIR / subject / roi_dir / "amplitudes_procrustes.npy"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")
    return np.load(path)


def project_new_subject(srm_model, new_data):
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def evaluate_k_for_fold(fold_idx, roi, k):
    """
    Evaluate one k value for one LOSO fold.

    Returns:
        Dict with reconstruction_error, rdm_reliability, left_out_rdm_corr
    """
    left_out = HC_SUBJECTS[fold_idx]
    train_hc = [s for i, s in enumerate(HC_SUBJECTS) if i != fold_idx]

    # Load training data
    train_srm_data = []
    train_full = {}
    for s in train_hc:
        amp = load_amplitudes(s, roi)
        train_full[s] = amp
        beta = amp.mean(axis=0)  # (8, n_voxels)
        train_srm_data.append(beta.T)  # (n_voxels, 8)

    # Load left-out data
    left_amp = load_amplitudes(left_out, roi)
    left_beta = left_amp.mean(axis=0)  # (8, n_voxels)

    # Fit SRM
    srm = SRM(n_iter=10, features=k)
    srm.fit(train_srm_data)

    # Project left-out subject
    W_left = project_new_subject(srm, left_beta.T)

    # Reconstruction error: X ≈ W @ S → error = ||X - W @ S||
    S = srm.s_  # (k, 8)
    X_left = left_beta.T  # (n_voxels, 8)
    reconstructed = W_left @ S  # (n_voxels, 8)
    recon_error = float(np.mean((X_left - reconstructed) ** 2))

    # RDM reliability of left-out subject in shared space
    # Project each run separately and compute run-pair RDM correlations
    left_aligned_runs = left_amp @ W_left  # (6, 8, k)
    run_rdms = []
    for run in range(6):
        patterns = left_aligned_runs[run]  # (8, k)
        rdm = 1 - np.corrcoef(patterns)
        run_rdms.append(rdm)

    mask = np.triu(np.ones((8, 8), dtype=bool), k=1)
    pairwise_corrs = []
    for i in range(6):
        for j in range(i + 1, 6):
            r, _ = spearmanr(run_rdms[i][mask], run_rdms[j][mask])
            if np.isfinite(r):
                pairwise_corrs.append(r)

    rdm_reliability = float(np.mean(pairwise_corrs)) if pairwise_corrs else 0.0

    # Left-out subject's RDM correlation with training subjects' mean RDM
    train_rdms = []
    for i, s in enumerate(train_hc):
        W_i = srm.w_[i]
        aligned = train_full[s] @ W_i  # (6, 8, k)
        avg_pattern = aligned.mean(axis=0)  # (8, k)
        rdm = 1 - np.corrcoef(avg_pattern)
        train_rdms.append(rdm[mask])

    mean_train_rdm = np.mean(train_rdms, axis=0)
    left_avg_pattern = left_aligned_runs.mean(axis=0)  # (8, k)
    left_rdm = (1 - np.corrcoef(left_avg_pattern))[mask]

    cross_subj_r, _ = spearmanr(left_rdm, mean_train_rdm)

    return {
        'k': k,
        'fold_idx': fold_idx,
        'left_out': left_out,
        'reconstruction_error': recon_error,
        'rdm_reliability': rdm_reliability,
        'cross_subject_rdm_corr': float(cross_subj_r) if np.isfinite(cross_subj_r) else 0.0
    }


def run_k_selection(fold_idx, roi):
    """Run all k candidates for one fold-ROI pair."""
    print(f"\n{'='*60}")
    print(f"k-Selection CV: Fold {fold_idx} (leave out {HC_SUBJECTS[fold_idx]}), ROI={roi}")
    print(f"k candidates: {K_CANDIDATES}")
    print(f"{'='*60}")

    results = {}
    for k in K_CANDIDATES:
        print(f"\n  k={k}...")
        try:
            r = evaluate_k_for_fold(fold_idx, roi, k)
            results[k] = r
            print(f"    Recon error: {r['reconstruction_error']:.6f}, "
                  f"RDM rel: {r['rdm_reliability']:.3f}, "
                  f"Cross-subj RDM: {r['cross_subject_rdm_corr']:.3f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            results[k] = {'error': str(e)}

    # Find best k
    valid = {k: v for k, v in results.items() if 'error' not in v}
    if valid:
        # Best = highest RDM reliability
        best_k_rdm = max(valid, key=lambda k: valid[k]['rdm_reliability'])
        # Best = lowest reconstruction error
        best_k_recon = min(valid, key=lambda k: valid[k]['reconstruction_error'])
        print(f"\n  Best k (RDM reliability): {best_k_rdm}")
        print(f"  Best k (recon error): {best_k_recon}")
    else:
        best_k_rdm = best_k_recon = None

    return {
        'roi': roi,
        'fold_idx': fold_idx,
        'left_out': HC_SUBJECTS[fold_idx],
        'results_by_k': {str(k): v for k, v in results.items()},
        'best_k_rdm': best_k_rdm,
        'best_k_recon': best_k_recon
    }


def main():
    parser = argparse.ArgumentParser(description='Test 2C: k-Selection CV')
    parser.add_argument('--fold', type=int, required=True, help='LOSO fold (0-6)')
    parser.add_argument('--roi', type=str, default=None, help='ROI. Omit for all.')
    args = parser.parse_args()

    rois = [args.roi] if args.roi else ROIS
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    start = time.time()
    all_results = {}

    for roi in rois:
        try:
            all_results[roi] = run_k_selection(args.fold, roi)
        except Exception as e:
            print(f"ERROR in {roi}: {e}")
            import traceback
            traceback.print_exc()

    roi_tag = args.roi if args.roi else "all"
    out_file = OUTPUT_BASE / f"k_selection_fold{args.fold}_{roi_tag}_results.json"
    with open(out_file, 'w') as f:
        json.dump({
            'config': {
                'test': '2C_optimal_k_selection',
                'data_dir': str(DATA_DIR),
                'dataset': 'full_dataset_C010',
                'input_file': 'amplitudes_procrustes.npy',
                'hc_subjects': HC_SUBJECTS,
                'k_candidates': K_CANDIDATES,
                'srm_n_iter': 10,
                'method': 'LOSO CV (HC only)',
                'metrics': ['reconstruction_error', 'rdm_reliability', 'cross_subject_rdm_corr'],
            },
            'settings': {'fold': args.fold, 'rois': rois, 'k_candidates': K_CANDIDATES,
                         'timestamp': timestamp, 'elapsed': time.time() - start},
            'results': all_results
        }, f, indent=2)

    print(f"\nSaved: {out_file}")
    print(f"Elapsed: {time.time() - start:.1f}s")


if __name__ == '__main__':
    main()
