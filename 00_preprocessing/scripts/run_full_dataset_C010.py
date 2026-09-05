#!/usr/bin/env python3
"""
Full Dataset Pipeline: P3 (C010 + Motion/Tissue + WM aCompCor)

Runs the validated Phase 2 P3 configuration on all subjects and ROIs.
This uses the exact same code that was tested and validated in Phase 2.

Author: Claude Code
Date: 2026-02-08
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase1_procrustes_decoding/run_full_dataset_C010.py  (development repository, commit 53c81c2)
# Role      : Per-run GLM -> (6 runs x 8 hues x voxels) amplitudes per ROI, plus run-to-run Procrustes alignment (dataset C010).
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
import pandas as pd
import nibabel as nib
from pathlib import Path
# nilearn not needed - using direct numpy indexing
from scipy.stats import spearmanr
from scipy.linalg import orthogonal_procrustes
import json
import sys
import argparse

# ============================================================================
# Configuration
# ============================================================================

# Paths (server)
BASE_DIR = Path("/scratch/connectome/haba6030/colorBlind")
FMRIPREP_DIR = Path("/storage/connectome/haba6030/fmriprep_out_method3_header_mi")
EVENT_DIR = Path("/storage/connectome/haba6030/bids_editted")
ROI_MASKS_DIR = BASE_DIR / "analysis" / "roi_masks" / "method3_header_mi"
OUTPUT_DIR = BASE_DIR / "derivatives" / "full_dataset_C010_with_residuals"  # NEW name to avoid overwrite

# Pipeline settings
TR = 1.5
N_SCANS = 288
N_RUNS = 6
N_COLORS = 8
FIR_DELAYS = np.arange(8) * TR

# C010 Configuration: Drift Only (NO confounds)
MOTION_TISSUE = False
WM_ACOMPCOR = False

print(f"Pipeline Configuration: C010 (Drift Only)")
print(f"  2nd-level drift: YES")
print(f"  Motion/Tissue confounds: NO")
print(f"  WM aCompCor: NO")
print()


# ============================================================================
# Functions from Phase 2 (Validated)
# ============================================================================

def get_roi_mask_path(subject_id, roi_name):
    """Get ROI mask path (server)"""
    roi_map = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}
    roi_prefix = roi_map.get(roi_name)
    if roi_prefix is None:
        raise ValueError(f"Unknown ROI: {roi_name}")

    # Server path structure
    mask_filename = f"{roi_prefix}_mask_thr50_intnearest_binTrue_maskfunc_gmTrue_subjFalse.nii.gz"
    mask_path = ROI_MASKS_DIR / f"sub-{subject_id}" / "roi_pipeline" / mask_filename
    return mask_path


def load_bold_data(subject_id, roi_name, run_idx):
    """Load BOLD data for single run (EXACT Phase 2 implementation)"""
    # Get ROI mask
    roi_mask_path = get_roi_mask_path(subject_id, roi_name)
    if not roi_mask_path.exists():
        raise FileNotFoundError(f"ROI mask not found: {roi_mask_path}")

    # BOLD file
    func_file = FMRIPREP_DIR / f"sub-{subject_id}" / "func" / \
                f"sub-{subject_id}_task-rsvp_run-{run_idx}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

    if not func_file.exists():
        raise FileNotFoundError(f"BOLD file not found: {func_file}")

    # Load BOLD and mask (CRITICAL: use numpy indexing, NOT nilearn.masking.apply_mask)
    func_img = nib.load(func_file)
    roi_mask_img = nib.load(roi_mask_path)

    func_data_4d = func_img.get_fdata()
    roi_mask = roi_mask_img.get_fdata() > 0

    func_data = func_data_4d[roi_mask].T  # (n_scans, n_voxels)

    # Events file (server structure)
    events_file = EVENT_DIR / f"sub-{subject_id}" / "func" / \
                  f"sub-{subject_id}_task-rsvp_run-{run_idx}_events.tsv"

    if not events_file.exists():
        raise FileNotFoundError(f"Events file not found: {events_file}")

    events = pd.read_csv(events_file, sep='\t')

    # CRITICAL: Adjust for dropped volumes (3 TRs)
    events['onset'] = events['onset'] - (3 * TR)

    return func_data, events


def load_motion_confounds(confounds_path):
    """Load motion + tissue confounds (from Phase 2)"""
    confounds_df = pd.read_csv(confounds_path, sep='\t')
    confound_cols = []

    # Motion parameters (6 DOF)
    motion_cols = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    confound_cols.extend(motion_cols)

    # Tissue signals
    tissue_cols = ['csf', 'white_matter']
    confound_cols.extend(tissue_cols)

    # Cosine drift (important!)
    cosine_cols = [c for c in confounds_df.columns if c.startswith('cosine')]
    confound_cols.extend(cosine_cols)

    # Extract available columns
    available_cols = [c for c in confound_cols if c in confounds_df.columns]

    if len(available_cols) == 0:
        return None, 0

    confounds = confounds_df[available_cols].values

    # Replace NaN/Inf
    if not np.all(np.isfinite(confounds)):
        confounds = np.nan_to_num(confounds, nan=0.0, posinf=0.0, neginf=0.0)

    return confounds, confounds.shape[1]


def load_wm_acompcor_only(confounds_path):
    """Load WM aCompCor (a_comp_cor_05~09) (from Phase 2)"""
    confounds_df = pd.read_csv(confounds_path, sep='\t')

    # WM aCompCor columns (indices 5-9)
    wm_cols = [f'a_comp_cor_{i:02d}' for i in range(5, 10)]
    available_cols = [c for c in wm_cols if c in confounds_df.columns]

    if len(available_cols) == 0:
        return None, 0

    confounds = confounds_df[available_cols].values

    # Replace NaN/Inf only (Phase 2 does NOT filter constant columns)
    if not np.all(np.isfinite(confounds)):
        confounds = np.nan_to_num(confounds, nan=0.0, posinf=0.0, neginf=0.0)

    return confounds, confounds.shape[1]


def create_drift_regressors(n_scans, run_idx, n_runs):
    """Create per-run linear + constant drift (from Phase 2)"""
    drift_cols = np.zeros((n_scans, n_runs * 2))

    # Linear regressor for this run (1-indexed in Phase 2)
    drift_cols[:, run_idx - 1] = np.linspace(-0.5, 0.5, n_scans)

    # Constant regressor for this run
    drift_cols[:, n_runs + run_idx - 1] = 1.0

    return drift_cols


def convolve_hrf_with_events(events, n_scans, tr, hrf, hrf_deriv):
    """Convolve HRF with event onsets (EXACT Phase 2 code)"""
    frame_times = np.arange(n_scans) * tr
    X_hrf = np.zeros((n_scans, N_COLORS * 2))  # HRF + derivative

    for color_idx in range(N_COLORS):
        color_name = f'color_{color_idx + 1}'
        color_events = events[events['trial_type'] == color_name]

        if len(color_events) == 0:
            continue

        for _, event in color_events.iterrows():
            onset = event['onset']
            # HRF
            hrf_signal = np.interp(frame_times - onset, np.arange(len(hrf)) * tr, hrf, left=0, right=0)
            X_hrf[:, color_idx] += hrf_signal

            # Derivative
            deriv_signal = np.interp(frame_times - onset, np.arange(len(hrf_deriv)) * tr, hrf_deriv, left=0, right=0)
            X_hrf[:, N_COLORS + color_idx] += deriv_signal

    return X_hrf


def build_2nd_level_design_matrix(events, n_scans, tr, hrf, hrf_deriv,
                                  run_idx, n_runs, confounds=None):
    """Build 2nd-level design matrix (EXACT Phase 2 code)"""
    X_components = []

    # 1. HRF regressors (8 colors)
    X_hrf = convolve_hrf_with_events(events, n_scans, tr, hrf, hrf_deriv)
    X_components.append(X_hrf[:, :N_COLORS])   # HRF
    X_components.append(X_hrf[:, N_COLORS:])   # Derivative

    # 2. Drift regressors (C010: 2nd-level drift)
    drift_cols = create_drift_regressors(n_scans, run_idx, n_runs)
    X_components.append(drift_cols)

    # 3. Confounds
    if confounds is not None:
        X_components.append(confounds)

    X = np.hstack(X_components)
    return X


def compute_split_half_reliability(amplitudes, n_iterations=100):
    """Compute random split-half reliability (from Phase 2)"""
    n_runs, n_colors, n_voxels = amplitudes.shape

    if n_runs < 4:
        return {'corrected': np.nan, 'raw': np.nan}

    correlations = []
    np.random.seed(42)

    for _ in range(n_iterations):
        run_indices = np.random.permutation(n_runs)
        split_point = n_runs // 2

        half1_runs = run_indices[:split_point]
        half2_runs = run_indices[split_point:2*split_point]

        half1_patterns = amplitudes[half1_runs].mean(axis=0)
        half2_patterns = amplitudes[half2_runs].mean(axis=0)

        rdm1 = 1 - np.corrcoef(half1_patterns)
        rdm2 = 1 - np.corrcoef(half2_patterns)

        mask = np.triu(np.ones_like(rdm1, dtype=bool), k=1)
        rdm1_vec = rdm1[mask]
        rdm2_vec = rdm2[mask]

        r, _ = spearmanr(rdm1_vec, rdm2_vec)
        correlations.append(r)

    correlations = np.array(correlations)
    correlations = correlations[np.isfinite(correlations)]

    r_half = np.mean(correlations)
    r_corrected = 2 * r_half / (1 + r_half) if r_half < 1 else 1.0

    return {'corrected': r_corrected, 'raw': r_half}


def compute_split_half_odd_even(amplitudes):
    """Compute odd/even split-half reliability (from Phase 2)"""
    n_runs, n_colors, n_voxels = amplitudes.shape

    if n_runs < 4:
        return {'corrected': np.nan, 'raw': np.nan}

    odd_indices = np.arange(0, n_runs, 2)
    even_indices = np.arange(1, n_runs, 2)

    odd_patterns = amplitudes[odd_indices].mean(axis=0)
    even_patterns = amplitudes[even_indices].mean(axis=0)

    rdm_odd = 1 - np.corrcoef(odd_patterns)
    rdm_even = 1 - np.corrcoef(even_patterns)

    mask = np.triu(np.ones_like(rdm_odd, dtype=bool), k=1)
    rdm_odd_vec = rdm_odd[mask]
    rdm_even_vec = rdm_even[mask]

    r_half, _ = spearmanr(rdm_odd_vec, rdm_even_vec)
    r_corrected = 2 * r_half / (1 + r_half) if r_half < 1 else 1.0

    return {'corrected': r_corrected, 'raw': r_half}


def apply_procrustes_alignment(amplitudes_raw):
    """Apply Procrustes alignment (EXACT Phase 2 code)"""
    n_runs, n_colors, n_voxels = amplitudes_raw.shape

    reference = amplitudes_raw[0]
    aligned = np.zeros_like(amplitudes_raw)
    aligned[0] = reference

    disparities = np.zeros(n_runs)

    for run_idx in range(1, n_runs):
        target = amplitudes_raw[run_idx]
        R, scale = orthogonal_procrustes(target.T, reference.T)
        aligned_run = (target.T @ R).T
        # CRITICAL: Phase 2 normalizes by n_voxels
        disparity = np.sum((aligned_run - reference) ** 2) / n_voxels
        disparities[run_idx] = disparity
        aligned[run_idx] = aligned_run

    return aligned, disparities


def compute_metrics(amplitudes_raw, amplitudes_proc, disparities):
    """Compute all metrics (from Phase 2)"""
    # Noise ceiling
    results_raw_random = compute_split_half_reliability(amplitudes_raw, n_iterations=100)
    results_raw_oddeven = compute_split_half_odd_even(amplitudes_raw)

    results_proc_random = compute_split_half_reliability(amplitudes_proc, n_iterations=100)
    results_proc_oddeven = compute_split_half_odd_even(amplitudes_proc)

    # Method difference
    raw_method_diff = abs(results_raw_random['corrected'] - results_raw_oddeven['corrected'])
    proc_method_diff = abs(results_proc_random['corrected'] - results_proc_oddeven['corrected'])

    # RDM reliability
    raw_rdm_reliability = (results_raw_random['raw'] + results_raw_oddeven['raw']) / 2
    proc_rdm_reliability = (results_proc_random['raw'] + results_proc_oddeven['raw']) / 2

    # Temporal autocorrelation
    n_runs, n_colors, n_voxels = amplitudes_raw.shape
    autocorr_values = []
    for voxel_idx in range(min(100, n_voxels)):
        voxel_timecourse = amplitudes_raw[:, :, voxel_idx].flatten()
        for run_idx in range(n_runs - 1):
            start = run_idx * n_colors
            end = start + n_colors
            next_start = (run_idx + 1) * n_colors
            next_end = next_start + n_colors

            r = np.corrcoef(voxel_timecourse[start:end], voxel_timecourse[next_start:next_end])[0, 1]
            if np.isfinite(r):
                autocorr_values.append(r)

    temporal_autocorr = np.mean(autocorr_values) if autocorr_values else 0.0

    # Drift magnitude
    drift_magnitudes = []
    for voxel_idx in range(n_voxels):
        voxel_means = amplitudes_raw[:, :, voxel_idx].mean(axis=1)
        slope = np.polyfit(np.arange(n_runs), voxel_means, 1)[0]
        drift_magnitudes.append(abs(slope))
    drift_magnitude = np.mean(drift_magnitudes)

    # Procrustes disparity
    procrustes_disparity = disparities.mean()

    return {
        'raw_noise_ceiling': {
            'random': results_raw_random['corrected'],
            'oddeven': results_raw_oddeven['corrected'],
            'method_diff': raw_method_diff
        },
        'procrustes_noise_ceiling': {
            'random': results_proc_random['corrected'],
            'oddeven': results_proc_oddeven['corrected'],
            'method_diff': proc_method_diff
        },
        'raw_rdm_reliability': raw_rdm_reliability,
        'procrustes_rdm_reliability': proc_rdm_reliability,
        'temporal_autocorrelation': temporal_autocorr,
        'drift_magnitude': drift_magnitude,
        'procrustes_disparity': procrustes_disparity
    }


# ============================================================================
# Main Pipeline (from Phase 2, adapted for single subject/ROI)
# ============================================================================

def run_subject_roi(subject_id, roi_name):
    """Run P3 pipeline for one subject-ROI pair"""

    print(f"\n{'='*80}")
    print(f"Processing: sub-{subject_id} {roi_name} (P3 Configuration)")
    print(f"{'='*80}\n")

    # Create output directory
    output_dir = OUTPUT_DIR / f"sub-{subject_id}" / roi_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 1st-level HRF estimation (EXACT Phase 2 code)
    print("Step 1: 1st-level HRF estimation...")

    all_func_data = []
    all_design_matrices = []

    for run_idx in range(1, N_RUNS + 1):
        print(f"  Run {run_idx}...")
        func_data, events = load_bold_data(subject_id, roi_name, run_idx)
        n_scans = func_data.shape[0]

        # Build FIR design matrix (no drift in 1st-level)
        X_fir = np.zeros((n_scans, len(FIR_DELAYS) * N_COLORS))

        for color_idx in range(N_COLORS):
            color_name = f'color_{color_idx + 1}'
            color_events = events[events['trial_type'] == color_name]

            for delay_idx, delay in enumerate(FIR_DELAYS):
                regressor_idx = color_idx * len(FIR_DELAYS) + delay_idx
                regressor = np.zeros(n_scans)

                for _, event in color_events.iterrows():
                    onset_tr = int(np.round(event['onset'] / TR))
                    target_tr = onset_tr + delay_idx

                    if 0 <= target_tr < n_scans:
                        regressor[target_tr] = 1.0

                X_fir[:, regressor_idx] = regressor

        all_func_data.append(func_data)
        all_design_matrices.append(X_fir)

    # Concatenate across runs
    Y_concat = np.vstack(all_func_data)
    X_concat = np.vstack(all_design_matrices)

    # Estimate FIR betas
    betas_fir = np.linalg.lstsq(X_concat, Y_concat, rcond=None)[0]

    # Average across voxels to get ROI HRF
    roi_hrf_fir = betas_fir.mean(axis=1).reshape(N_COLORS, len(FIR_DELAYS))

    # Average across colors
    roi_hrf = roi_hrf_fir.mean(axis=0)

    # Compute derivative
    roi_hrf_deriv = np.gradient(roi_hrf)

    print(f"  ✓ ROI HRF estimated: shape {roi_hrf.shape}")

    # Step 2: 2nd-level amplitude estimation (C010: drift only)
    print("\nStep 2: 2nd-level amplitude estimation (C010: Drift Only, No Confounds)...")

    amplitudes_raw = []
    all_residuals = []  # NEW: Store residuals for noise covariance estimation

    for run_idx in range(1, N_RUNS + 1):
        print(f"  Run {run_idx}...")
        func_data, events = load_bold_data(subject_id, roi_name, run_idx)
        n_scans = func_data.shape[0]  # Get actual number of scans for this run

        # Load confounds
        confounds_path = FMRIPREP_DIR / f"sub-{subject_id}" / "func" / \
                        f"sub-{subject_id}_task-rsvp_run-{run_idx}_desc-confounds_timeseries.tsv"

        all_confounds = []

        # Motion/Tissue confounds
        if MOTION_TISSUE:
            motion_conf, n_motion = load_motion_confounds(confounds_path)
            if motion_conf is not None:
                all_confounds.append(motion_conf)
                print(f"    Loaded {n_motion} motion/tissue confounds")

        # WM aCompCor
        if WM_ACOMPCOR:
            wm_conf, n_wm = load_wm_acompcor_only(confounds_path)
            if wm_conf is not None:
                all_confounds.append(wm_conf)
                print(f"    Loaded {n_wm} WM aCompCor confounds")

        confounds = np.hstack(all_confounds) if all_confounds else None

        # Build design matrix (use actual n_scans, not constant)
        X = build_2nd_level_design_matrix(
            events, n_scans, TR, roi_hrf, roi_hrf_deriv,
            run_idx, N_RUNS, confounds
        )

        print(f"    Design matrix: {X.shape}")

        # Estimate amplitudes
        betas, _, _, _ = np.linalg.lstsq(X, func_data, rcond=None)
        amplitudes = betas[:N_COLORS, :]
        amplitudes_raw.append(amplitudes)

        # NEW: Compute residuals (for noise covariance estimation)
        Y_hat = X @ betas  # Predicted signal
        residuals = func_data - Y_hat  # Residuals (noise)
        all_residuals.append(residuals)
        print(f"    Residuals: {residuals.shape}")

    amplitudes_raw = np.array(amplitudes_raw)  # (6, 8, n_voxels)
    all_residuals = np.vstack(all_residuals)  # (total_scans, n_voxels)
    print(f"\n✓ Amplitudes computed: {amplitudes_raw.shape}")
    print(f"✓ Residuals computed: {all_residuals.shape}")

    # Step 3: Procrustes alignment
    print("\nStep 3: Procrustes alignment...")
    amplitudes_proc, disparities = apply_procrustes_alignment(amplitudes_raw)
    print(f"  ✓ Mean disparity: {disparities.mean():.6f}")

    # Step 4: Compute metrics
    print("\nStep 4: Computing metrics...")
    metrics = compute_metrics(amplitudes_raw, amplitudes_proc, disparities)

    print(f"\n  Noise Ceiling (Raw):")
    print(f"    Method Diff: {metrics['raw_noise_ceiling']['method_diff']:.4f}")
    print(f"  RDM Reliability:")
    print(f"    Raw: {metrics['raw_rdm_reliability']:.4f}")
    print(f"    Procrustes: {metrics['procrustes_rdm_reliability']:.4f}")

    # Step 5: Save outputs
    print("\nStep 5: Saving outputs...")

    np.save(output_dir / 'amplitudes_raw.npy', amplitudes_raw)
    np.save(output_dir / 'amplitudes_procrustes.npy', amplitudes_proc)
    np.save(output_dir / 'procrustes_disparities.npy', disparities)
    np.save(output_dir / 'roi_hrf.npy', roi_hrf)
    np.save(output_dir / 'roi_hrf_deriv.npy', roi_hrf_deriv)
    np.save(output_dir / 'residuals_2nd_level.npy', all_residuals)  # NEW: Save residuals

    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    config = {
        'subject': subject_id,
        'roi': roi_name,
        'n_voxels': amplitudes_raw.shape[2],
        'pipeline': 'P3',
        'motion_tissue': MOTION_TISSUE,
        'wm_acompcor': WM_ACOMPCOR
    }

    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Outputs saved to: {output_dir}")
    print(f"\n{'='*80}")
    print(f"COMPLETED: sub-{subject_id} {roi_name}")
    print(f"{'='*80}\n")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Full Dataset P3 Pipeline')
    parser.add_argument('--subject', required=True, help='Subject ID (e.g., 02)')
    parser.add_argument('--roi', required=True, choices=['V1', 'V2', 'V3', 'V4'], help='ROI name')

    args = parser.parse_args()

    try:
        metrics = run_subject_roi(args.subject, args.roi)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
