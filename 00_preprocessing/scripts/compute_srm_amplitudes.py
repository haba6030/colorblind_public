#!/usr/bin/env python3
"""
Compute SRM Per-Run Amplitudes

Trains HC-only SRM and saves amplitudes_srm.npy (6, 8, k) per subject per ROI,
alongside existing amplitudes_procrustes.npy for reusability.

HC subjects: SRM W_i applied directly
CVD subjects: SVD projection (same as rerun_loo_consistent.py)

Requires: conda activate srm (BrainIAK)
Usage:    mpirun -np 1 python compute_srm_amplitudes.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase3_decoder_comparing/analysis/compute_srm_amplitudes.py  (development repository, commit 53c81c2)
# Role      : SRM-aligned amplitudes (`amplitudes_srm.npy`) used by the cross-participant decoder comparison.
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
import sys
from pathlib import Path
from datetime import datetime

try:
    from brainiak.funcalign.srm import SRM
except ImportError:
    print("ERROR: brainiak not available. Use: conda activate srm")
    sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================

HC_SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
CVD_SUBJECTS = ['sub-08', 'sub-09', 'sub-10']
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS

ROIS = ['V1', 'V2', 'V3', 'V4']
ROI_DISPLAY = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}
K_VALUES = {'V1': 4, 'V2': 4, 'V3': 3, 'V4': 3}

import socket
if socket.gethostname().startswith('node'):
    DATA_DIR = Path("/scratch/connectome/haba6030/colorBlind/derivatives/full_dataset_C010")
else:
    DATA_DIR = Path("/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/colorBlind_analysis"
                    "/analysis/phase1_preprocess_decoding/results/full_dataset_C010")


# ============================================================================
# CORE FUNCTIONS (from rerun_loo_consistent.py)
# ============================================================================

def project_new_subject(srm_model, new_data):
    """Project new subject into SRM space via SVD (rerun_loo_consistent.py:85-89)"""
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def compute_srm_amplitudes(roi, k):
    """
    Train HC-only SRM and compute per-run amplitudes for all subjects.

    Returns dict: {subject: (6, 8, k) array}
    """
    roi_dir = roi  # V4 on disk for hV4
    roi_display = ROI_DISPLAY.get(roi, roi)

    print(f"\n{'='*60}")
    print(f"Computing SRM amplitudes: {roi_display}, k={k}")
    print(f"{'='*60}")

    # Load all subjects' data
    hc_data = {}
    for s in HC_SUBJECTS:
        path = DATA_DIR / s / roi_dir / "amplitudes_procrustes.npy"
        hc_data[s] = np.load(path)  # (6, 8, n_voxels)
        print(f"  Loaded {s}: {hc_data[s].shape}")

    cvd_data = {}
    for s in CVD_SUBJECTS:
        path = DATA_DIR / s / roi_dir / "amplitudes_procrustes.npy"
        cvd_data[s] = np.load(path)
        print(f"  Loaded {s}: {cvd_data[s].shape}")

    # Train HC-only SRM on mean-across-runs betas
    hc_betas = [hc_data[s].mean(axis=0).T for s in HC_SUBJECTS]  # list of (n_voxels, 8)
    print(f"\n  Training SRM on {len(HC_SUBJECTS)} HC subjects (k={k})...")
    srm = SRM(n_iter=10, features=k)
    srm.fit(hc_betas)
    print(f"  SRM training complete.")

    results = {}

    # Apply W_i to per-run HC data → (6, 8, k)
    for i, s in enumerate(HC_SUBJECTS):
        W_i = srm.w_[i]  # (n_voxels, k)
        data = hc_data[s]  # (6, 8, n_voxels)
        aligned = data @ W_i  # (6, 8, n_voxels) @ (n_voxels, k) → (6, 8, k)
        results[s] = aligned
        print(f"  {s} HC: {data.shape} → {aligned.shape}")

    # Project CVD via SVD, apply to per-run data → (6, 8, k)
    for s in CVD_SUBJECTS:
        data = cvd_data[s]  # (6, 8, n_voxels)
        beta = data.mean(axis=0)  # (8, n_voxels)
        W_cvd = project_new_subject(srm, beta.T)  # (n_voxels, k)
        aligned = data @ W_cvd  # (6, 8, k)
        results[s] = aligned
        print(f"  {s} CVD: {data.shape} → {aligned.shape}")

    return results, srm


def save_srm_amplitudes(roi, k, results, srm):
    """Save amplitudes_srm.npy and srm_config.json per subject."""
    roi_dir = roi

    for s, aligned in results.items():
        out_dir = DATA_DIR / s / roi_dir
        out_path = out_dir / "amplitudes_srm.npy"
        np.save(out_path, aligned)
        print(f"  Saved: {out_path} ({aligned.shape})")

        # Save config per subject-ROI
        config = {
            'description': 'SRM-projected per-run amplitudes',
            'source': 'amplitudes_procrustes.npy',
            'srm_k': k,
            'srm_n_iter': 10,
            'training_subjects': HC_SUBJECTS,
            'projection_method': 'w_' if s in HC_SUBJECTS else 'SVD (project_new_subject)',
            'shape': list(aligned.shape),
            'roi': ROI_DISPLAY.get(roi, roi),
            'created': datetime.now().isoformat(),
        }
        config_path = out_dir / "srm_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)


def verify_results(roi, k):
    """Verify saved amplitudes_srm.npy files."""
    roi_dir = roi
    roi_display = ROI_DISPLAY.get(roi, roi)
    print(f"\n  Verification for {roi_display}:")

    for s in ALL_SUBJECTS:
        path = DATA_DIR / s / roi_dir / "amplitudes_srm.npy"
        if path.exists():
            data = np.load(path)
            expected_shape = (6, 8, k)
            status = "OK" if data.shape == expected_shape else f"MISMATCH (expected {expected_shape})"
            print(f"    {s}: {data.shape} {status}")
        else:
            print(f"    {s}: MISSING")


def main():
    print("="*60)
    print("Compute SRM Per-Run Amplitudes")
    print(f"Data: {DATA_DIR}")
    print(f"K values: {K_VALUES}")
    print("="*60)

    for roi in ROIS:
        k = K_VALUES[roi]
        results, srm = compute_srm_amplitudes(roi, k)
        save_srm_amplitudes(roi, k, results, srm)
        verify_results(roi, k)

    print(f"\n{'='*60}")
    print("All SRM amplitudes computed and saved.")
    print("="*60)


if __name__ == '__main__':
    main()
