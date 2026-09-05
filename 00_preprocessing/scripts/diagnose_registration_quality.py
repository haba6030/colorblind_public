#!/usr/bin/env python3
"""
Registration Quality Diagnosis - Visual Inspection

Compare registration quality across methods by overlaying Wang atlas ROIs
on BOLD functional references.

Usage:
    python diagnose_registration_quality.py --subject 01 --run 1 --method method3_header_mi
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/diagnose_registration_quality.py  (development repository, commit 53c81c2)
# Role      : Registration diagnostics used for the visual inspection reported in S2.
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
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn import image, plotting
from nilearn.glm.first_level import make_first_level_design_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import json
from datetime import datetime


# ROI definitions (Wang atlas)
ROI_DEFINITIONS = {
    'V1': {
        'files': [
            'perc_VTPM_vol_roi1_lh.nii.gz',  # V1v left
            'perc_VTPM_vol_roi1_rh.nii.gz',  # V1v right
            'perc_VTPM_vol_roi2_lh.nii.gz',  # V1d left
            'perc_VTPM_vol_roi2_rh.nii.gz'   # V1d right
        ]
    },
    'V2': {
        'files': [
            'perc_VTPM_vol_roi3_lh.nii.gz',  # V2v left
            'perc_VTPM_vol_roi3_rh.nii.gz',  # V2v right
            'perc_VTPM_vol_roi4_lh.nii.gz',  # V2d left
            'perc_VTPM_vol_roi4_rh.nii.gz'   # V2d right
        ]
    },
    'V3': {
        'files': [
            'perc_VTPM_vol_roi5_lh.nii.gz',  # V3v left
            'perc_VTPM_vol_roi5_rh.nii.gz',  # V3v right
            'perc_VTPM_vol_roi6_lh.nii.gz',  # V3d left
            'perc_VTPM_vol_roi6_rh.nii.gz'   # V3d right
        ]
    },
    'hV4': {
        'files': [
            'perc_VTPM_vol_roi7_lh.nii.gz',  # hV4 left
            'perc_VTPM_vol_roi7_rh.nii.gz'   # hV4 right
        ]
    }
}


def load_and_combine_roi(atlas_dir, roi_name, threshold=50):
    """
    Load Wang atlas ROI files and combine into single binary mask

    Parameters:
    -----------
    atlas_dir : Path
        Directory containing Wang atlas files
    roi_name : str
        ROI name (V1, V2, V3, hV4)
    threshold : float
        Probability threshold (0-100 scale)

    Returns:
    --------
    roi_mask_img : nib.Nifti1Image
        Combined binary ROI mask
    combined_prob : np.ndarray
        Combined probability map (0-100 scale)
    """
    roi_files = ROI_DEFINITIONS[roi_name]['files']

    # Load first file as template
    first_file = atlas_dir / roi_files[0]
    template_img = nib.load(first_file)
    combined_data = np.zeros(template_img.shape, dtype=np.float32)

    # Combine all ROI files
    for roi_file in roi_files:
        roi_path = atlas_dir / roi_file
        if not roi_path.exists():
            print(f"⚠️  WARNING: ROI file not found: {roi_file}")
            continue

        roi_img = nib.load(roi_path)
        roi_data = roi_img.get_fdata()

        # Wang atlas stores probabilities as 0-100
        combined_data = np.maximum(combined_data, roi_data)

    # Apply threshold and binarize
    binary_mask = (combined_data >= threshold).astype(np.float32)

    # Create new image
    roi_mask_img = nib.Nifti1Image(binary_mask, template_img.affine, template_img.header)

    # Return both mask and probability map
    return roi_mask_img, combined_data


def compute_roi_alignment_metrics(roi_mask, roi_prob_map, bold_mean, bold_brain_mask, method_name, roi_name):
    """
    Compute quantitative alignment metrics between ROI atlas and BOLD

    Parameters:
    -----------
    roi_mask : nib.Nifti1Image
        Binary ROI mask (thresholded)
    roi_prob_map : np.ndarray
        ROI probability map (0-100 scale)
    bold_mean : nib.Nifti1Image
        Mean BOLD image
    bold_brain_mask : nib.Nifti1Image
        BOLD brain mask
    method_name : str
        Preprocessing method name
    roi_name : str
        ROI name

    Returns:
    --------
    metrics : dict
        Alignment metrics
    """
    # Get data
    roi_data = roi_mask.get_fdata()
    roi_binary = (roi_data > 0).astype(np.float32)
    bold_data = bold_mean.get_fdata()
    brain_mask_data = bold_brain_mask.get_fdata()
    brain_binary = (brain_mask_data > 0).astype(np.float32)

    # Ensure same shape - resample brain mask AND BOLD mean if needed
    if roi_binary.shape != brain_binary.shape:
        print(f"  Warning: Shape mismatch - ROI {roi_binary.shape} vs Brain {brain_binary.shape}")
        print(f"  Resampling brain mask and BOLD mean to match ROI...")

        # Resample brain mask to match ROI
        brain_mask_resampled = image.resample_img(
            bold_brain_mask,
            target_affine=roi_mask.affine,
            target_shape=roi_mask.shape[:3],
            interpolation='nearest',
            force_resample=True,
            copy_header=True
        )
        brain_binary = brain_mask_resampled.get_fdata()
        brain_binary = (brain_binary > 0).astype(np.float32)

        # Also resample BOLD mean image to match ROI
        bold_mean_resampled = image.resample_img(
            bold_mean,
            target_affine=roi_mask.affine,
            target_shape=roi_mask.shape[:3],
            interpolation='linear',  # Use linear for intensity values
            force_resample=True,
            copy_header=True
        )
        bold_data = bold_mean_resampled.get_fdata()

    # 1. Basic counts
    roi_voxels = int(np.sum(roi_binary))
    bold_brain_voxels = int(np.sum(brain_binary))

    # 2. Intersection (ROI ∩ BOLD brain)
    intersection = roi_binary * brain_binary
    intersection_voxels = int(np.sum(intersection))

    # 3. ROI coverage ratio (what % of ROI is covered by BOLD?)
    roi_coverage_ratio = intersection_voxels / roi_voxels if roi_voxels > 0 else 0.0

    # 4. BOLD-in-ROI ratio (what % of BOLD brain is in ROI?)
    bold_in_roi_ratio = intersection_voxels / bold_brain_voxels if bold_brain_voxels > 0 else 0.0

    # 5. Mean BOLD signal in ROI
    bold_in_roi = bold_data * intersection
    mean_signal_in_roi = np.mean(bold_in_roi[bold_in_roi > 0]) if np.any(bold_in_roi > 0) else 0.0
    std_signal_in_roi = np.std(bold_in_roi[bold_in_roi > 0]) if np.any(bold_in_roi > 0) else 0.0

    # 6. Mean BOLD signal in whole brain (for comparison)
    bold_in_brain = bold_data * brain_binary
    mean_signal_in_brain = np.mean(bold_in_brain[bold_in_brain > 0]) if np.any(bold_in_brain > 0) else 0.0

    # 7. Signal ratio (ROI vs whole brain)
    signal_ratio = mean_signal_in_roi / mean_signal_in_brain if mean_signal_in_brain > 0 else 0.0

    # 8. Weighted overlap (using ROI probabilities)
    # Resample prob_map to match roi_mask shape
    if roi_prob_map.shape != roi_binary.shape:
        # roi_prob_map is already in atlas space, roi_mask has been resampled
        # We need to use the intersection with probabilities
        weighted_overlap = intersection_voxels  # Fallback to binary
        weighted_overlap_sum = intersection_voxels
    else:
        weighted_intersection = roi_prob_map * brain_binary / 100.0  # Convert to 0-1 scale
        weighted_overlap_sum = float(np.sum(weighted_intersection))
        weighted_overlap = weighted_overlap_sum

    metrics = {
        'method': method_name,
        'roi': roi_name,
        'roi_voxels': roi_voxels,
        'bold_brain_voxels': bold_brain_voxels,
        'intersection_voxels': intersection_voxels,
        'roi_coverage_ratio': float(roi_coverage_ratio),
        'bold_in_roi_ratio': float(bold_in_roi_ratio),
        'mean_signal_in_roi': float(mean_signal_in_roi),
        'std_signal_in_roi': float(std_signal_in_roi),
        'mean_signal_in_brain': float(mean_signal_in_brain),
        'signal_ratio_roi_vs_brain': float(signal_ratio),
        'weighted_overlap': float(weighted_overlap)
    }

    return metrics


def run_lightweight_glm(bold_4d, events_df, roi_mask, brain_mask, tr=1.5):
    """
    Run lightweight GLM with canonical HRF for diagnostic purposes

    Parameters:
    -----------
    bold_4d : nib.Nifti1Image
        4D BOLD image
    events_df : pd.DataFrame
        Events with columns: onset, duration, trial_type
    roi_mask : nib.Nifti1Image
        Binary ROI mask
    brain_mask : nib.Nifti1Image
        BOLD brain mask
    tr : float
        Repetition time in seconds

    Returns:
    --------
    glm_metrics : dict
        Dictionary with GLM quality metrics
    """
    # Compute ROI ∩ Brain intersection
    roi_data = roi_mask.get_fdata()
    brain_data = brain_mask.get_fdata()

    # Check shape compatibility - resample brain mask if needed
    if roi_data.shape != brain_data.shape:
        # Resample brain mask to match ROI
        brain_mask_resampled = image.resample_img(
            brain_mask,
            target_affine=roi_mask.affine,
            target_shape=roi_mask.shape[:3],
            interpolation='nearest',
            force_resample=True,
            copy_header=True
        )
        brain_data = brain_mask_resampled.get_fdata()

    intersection_mask = (roi_data > 0) & (brain_data > 0)
    n_roi_voxels = int(np.sum(intersection_mask))

    if n_roi_voxels == 0:
        return {
            'glm_n_roi_voxels': 0,
            'glm_n_valid_voxels': 0,
            'glm_valid_ratio': 0.0,
            'glm_mean_amplitude': 0.0,
            'glm_std_amplitude': 0.0,
            'glm_mean_variance_across_colors': 0.0,
            'glm_error': 'Empty ROI intersection'
        }

    # Extract ROI timeseries
    bold_data = bold_4d.get_fdata()  # (x, y, z, time)
    n_scans = bold_data.shape[-1]

    # Check BOLD spatial dimensions match ROI - resample if needed
    if bold_data.shape[:3] != roi_data.shape:
        print(f"    Resampling 4D BOLD from {bold_data.shape[:3]} to {roi_data.shape}...")

        # Resample 4D BOLD to match ROI space
        bold_4d_resampled = image.resample_img(
            bold_4d,
            target_affine=roi_mask.affine,
            target_shape=roi_mask.shape[:3],
            interpolation='linear',
            force_resample=True,
            copy_header=True
        )
        bold_data = bold_4d_resampled.get_fdata()
        n_scans = bold_data.shape[-1]

    # Reshape to (time, voxels)
    bold_2d = bold_data.reshape(-1, n_scans).T  # (time, n_all_voxels)

    # Apply intersection mask
    intersection_flat = intersection_mask.flatten()
    roi_timeseries = bold_2d[:, intersection_flat]  # (time, n_roi_voxels)

    # Create frame times
    frame_times = np.arange(n_scans) * tr

    # Filter events: only keep color events (exclude 'blank')
    if 'trial_type' in events_df.columns:
        color_events = events_df[events_df['trial_type'].str.startswith('color_')].copy()
    elif 'color_name' in events_df.columns:
        # Alternative column name
        color_events = events_df[events_df['color_name'].notna()].copy()
        color_events['trial_type'] = 'color_' + color_events['color_name'].astype(str)
    else:
        return {
            'glm_n_roi_voxels': n_roi_voxels,
            'glm_n_valid_voxels': 0,
            'glm_valid_ratio': 0.0,
            'glm_mean_amplitude': 0.0,
            'glm_std_amplitude': 0.0,
            'glm_mean_variance_across_colors': 0.0,
            'glm_error': 'No trial_type or color_name column in events'
        }

    if len(color_events) == 0:
        return {
            'glm_n_roi_voxels': n_roi_voxels,
            'glm_n_valid_voxels': 0,
            'glm_valid_ratio': 0.0,
            'glm_mean_amplitude': 0.0,
            'glm_std_amplitude': 0.0,
            'glm_mean_variance_across_colors': 0.0,
            'glm_error': 'No color events found'
        }

    # Build design matrix with nilearn (canonical HRF)
    try:
        design_matrix = make_first_level_design_matrix(
            frame_times,
            events=color_events,
            hrf_model='spm',  # SPM canonical HRF
            drift_model=None,  # No drift - keep lightweight
            high_pass=None     # No high-pass filter
        )
    except Exception as e:
        return {
            'glm_n_roi_voxels': n_roi_voxels,
            'glm_n_valid_voxels': 0,
            'glm_valid_ratio': 0.0,
            'glm_mean_amplitude': 0.0,
            'glm_std_amplitude': 0.0,
            'glm_mean_variance_across_colors': 0.0,
            'glm_error': f'Design matrix creation failed: {str(e)}'
        }

    # Extract only color columns (drop 'constant')
    color_cols = [col for col in design_matrix.columns
                  if col.startswith('color_')]

    if len(color_cols) == 0:
        return {
            'glm_n_roi_voxels': n_roi_voxels,
            'glm_n_valid_voxels': 0,
            'glm_valid_ratio': 0.0,
            'glm_mean_amplitude': 0.0,
            'glm_std_amplitude': 0.0,
            'glm_mean_variance_across_colors': 0.0,
            'glm_error': 'No color columns in design matrix'
        }

    X = design_matrix[color_cols].values  # (n_scans, n_colors)

    # Add constant term
    X_with_const = np.column_stack([X, np.ones(n_scans)])

    # Fit GLM: beta = (X'X)^-1 X'y for each voxel
    try:
        XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
        betas = XtX_inv @ X_with_const.T @ roi_timeseries
        # betas shape: (n_colors + 1, n_voxels)
        amplitudes = betas[:len(color_cols), :]  # Exclude constant term
    except np.linalg.LinAlgError as e:
        return {
            'glm_n_roi_voxels': n_roi_voxels,
            'glm_n_valid_voxels': 0,
            'glm_valid_ratio': 0.0,
            'glm_mean_amplitude': 0.0,
            'glm_std_amplitude': 0.0,
            'glm_mean_variance_across_colors': 0.0,
            'glm_error': f'GLM convergence failed: {str(e)}'
        }

    # Filter valid voxels
    valid_mask = ~(np.isnan(amplitudes).any(axis=0) |
                   (np.abs(amplitudes).sum(axis=0) < 1e-8))
    n_valid_voxels = int(np.sum(valid_mask))

    if n_valid_voxels == 0:
        return {
            'glm_n_roi_voxels': n_roi_voxels,
            'glm_n_valid_voxels': 0,
            'glm_valid_ratio': 0.0,
            'glm_mean_amplitude': 0.0,
            'glm_std_amplitude': 0.0,
            'glm_mean_variance_across_colors': 0.0
        }

    # Compute statistics on valid voxels only
    valid_amplitudes = amplitudes[:, valid_mask]

    # Mean absolute amplitude (overall signal strength)
    mean_amp = float(np.mean(np.abs(valid_amplitudes)))
    std_amp = float(np.std(np.abs(valid_amplitudes)))

    # Variance across colors per voxel (signal differentiation)
    var_across_colors = np.var(valid_amplitudes, axis=0)
    mean_var = float(np.mean(var_across_colors))

    return {
        'glm_n_roi_voxels': n_roi_voxels,
        'glm_n_valid_voxels': n_valid_voxels,
        'glm_valid_ratio': float(n_valid_voxels / n_roi_voxels),
        'glm_mean_amplitude': mean_amp,
        'glm_std_amplitude': std_amp,
        'glm_mean_variance_across_colors': mean_var
    }


def create_overlay_comparison(roi_mask, bold_refs, method_names, roi_name, output_dir, run_id):
    """
    Create individual overlay for each method

    Parameters:
    -----------
    roi_mask : nib.Nifti1Image
        ROI mask to overlay
    bold_refs : list of nib.Nifti1Image
        BOLD reference images for each method
    method_names : list of str
        Method names
    roi_name : str
        ROI name for title
    output_dir : Path
        Output directory
    run_id : int
        Run number
    """
    saved_files = []

    for bold_ref, method_name in zip(bold_refs, method_names):
        # Create individual figure for each method
        fig = plt.figure(figsize=(15, 5))

        # Create overlay with ortho view (matching original ROI pipeline style)
        display = plotting.plot_roi(
            roi_mask,
            bg_img=bold_ref,
            display_mode='ortho',
            cut_coords=None,
            title=f'{roi_name} on {method_name}',
            cmap='autumn',
            alpha=0.7,
            draw_cross=True,
            annotate=True,
            black_bg=False,
            figure=fig
        )

        # Save individual file
        # Clean method name for filename
        method_clean = method_name.replace('_', '-').replace(' ', '-')
        output_path = output_dir / f'{roi_name}_{method_clean}_run{run_id}.png'

        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        plt.clf()

        saved_files.append(output_path.name)
        print(f"  ✅ Saved: {output_path.name}")

    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Diagnose registration quality with ROI overlay')
    parser.add_argument('--subject', type=str, required=True, help='Subject ID (e.g., 01)')
    parser.add_argument('--run', type=int, default=1, help='Run number (default: 1)')
    parser.add_argument('--method', type=str, required=True,
                        help='Method name (e.g., original_v3, method2_header_bbr, method3_header_mi)')
    parser.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'hV4'],
                        help='ROIs to visualize (default: all)')
    parser.add_argument('--threshold', type=float, default=50,
                        help='ROI probability threshold 0-100 (default: 50)')

    args = parser.parse_args()

    # Paths
    atlas_dir = Path('/scratch/connectome/haba6030/colorBlind/ProbAtlas_v4_2mm/subj_vol_all')
    prep_trials_dir = Path('/scratch/connectome/haba6030/colorBlind/analysis/prep_trials')
    original_dir = Path('/storage/connectome/haba6030')

    # Output directory structure: diagnose/Method_{method_name}/sub-{subject}/
    output_base_dir = prep_trials_dir / 'diagnose' / f'Method_{args.method}'
    output_dir = output_base_dir / f'sub-{args.subject}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print(f"Registration Quality Diagnosis - Sub-{args.subject} Run {args.run}")
    print(f"Method: {args.method}")
    print("="*80)
    print()

    # Map method names to data paths
    method_paths = {
        'original_v3': original_dir / 'fmriprep_out_original_v3',
        'method2_header_bbr': prep_trials_dir / 'method2_header_bbr',
        'method3_header_mi': original_dir / 'fmriprep_out_method3_header_mi'
    }

    if args.method not in method_paths:
        print(f"❌ ERROR: Unknown method '{args.method}'")
        print(f"Available methods: {list(method_paths.keys())}")
        return

    # Get BOLD file path for the specified method
    method_data_dir = method_paths[args.method]
    bold_file = method_data_dir / f'sub-{args.subject}/func' / \
                f'sub-{args.subject}_task-rsvp_run-{args.run}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz'

    bold_files = {args.method: bold_file}

    # Load BOLD data - keep 4D for GLM, compute mean for overlay
    print("Loading BOLD data...")
    bold_4d_imgs = {}  # Store 4D for GLM
    bold_refs = {}     # Store mean for overlay
    for method_name, bold_file in bold_files.items():
        if bold_file.exists():
            print(f"  ⚙️  {method_name}: loading 4D BOLD...")
            bold_4d = nib.load(bold_file)
            bold_4d_imgs[method_name] = bold_4d
            bold_mean = image.mean_img(bold_4d)
            bold_refs[method_name] = bold_mean
            print(f"  ✅ {method_name}: loaded (shape: {bold_4d.shape})")
        else:
            print(f"  ❌ {method_name}: File not found")
            print(f"     {bold_file}")

    if len(bold_refs) == 0:
        print("\n❌ ERROR: No BOLD files found")
        return

    print(f"\n✅ Loaded {len(bold_refs)} method(s)")

    # Load brain mask for the specified method
    print("\nLoading brain masks...")
    brain_masks = {}
    brain_mask_file = method_data_dir / f'sub-{args.subject}/func' / \
                      f'sub-{args.subject}_task-rsvp_run-{args.run}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz'
    brain_mask_files = {args.method: brain_mask_file}

    for method_name, mask_file in brain_mask_files.items():
        if method_name in bold_refs:  # Only load if BOLD was loaded
            if mask_file.exists():
                brain_masks[method_name] = nib.load(mask_file)
                print(f"  ✅ {method_name}: brain mask loaded")
            else:
                print(f"  ⚠️  {method_name}: brain mask not found, will skip metrics")

    # Load event file for GLM
    print("\nLoading event file for GLM...")
    event_dir = Path('/storage/connectome/haba6030/bids_editted')
    event_file = event_dir / f'sub-{args.subject}/func' / \
                f'sub-{args.subject}_task-rsvp_run-{args.run}_events.tsv'

    if event_file.exists():
        events_df = pd.read_csv(event_file, sep='\t')
        print(f"  ✅ Loaded event file: {len(events_df)} events")
    else:
        print(f"  ⚠️  WARNING: Event file not found - GLM metrics will be skipped")
        print(f"     {event_file}")
        events_df = None

    print()

    # Storage for alignment metrics
    all_metrics = []

    # Process each ROI
    for roi_name in args.rois:
        print(f"Processing {roi_name}...")

        # Load and combine ROI
        try:
            roi_mask, roi_prob_map = load_and_combine_roi(atlas_dir, roi_name, args.threshold)
            n_voxels = np.sum(roi_mask.get_fdata() > 0)
            print(f"  ROI voxels: {n_voxels}")
        except Exception as e:
            print(f"  ❌ ERROR loading ROI: {e}")
            continue

        # Compute alignment metrics for each method
        print(f"  Computing alignment metrics...")
        for method_name in bold_refs.keys():
            if method_name in brain_masks:
                try:
                    # Compute basic alignment metrics
                    metrics = compute_roi_alignment_metrics(
                        roi_mask,
                        roi_prob_map,
                        bold_refs[method_name],
                        brain_masks[method_name],
                        method_name,
                        roi_name
                    )

                    if metrics is not None:
                        metrics['subject'] = args.subject
                        metrics['run'] = args.run

                        # Add GLM metrics if event file available
                        if events_df is not None and method_name in bold_4d_imgs:
                            print(f"    {method_name}: Running GLM...")
                            glm_metrics = run_lightweight_glm(
                                bold_4d_imgs[method_name],
                                events_df,
                                roi_mask,
                                brain_masks[method_name],
                                tr=1.5
                            )
                            metrics.update(glm_metrics)

                            if 'glm_error' in glm_metrics:
                                print(f"    {method_name}: GLM failed - {glm_metrics['glm_error']}")
                            else:
                                print(f"    {method_name}: GLM done - "
                                      f"{glm_metrics['glm_n_valid_voxels']}/{glm_metrics['glm_n_roi_voxels']} "
                                      f"valid voxels ({glm_metrics['glm_valid_ratio']:.2%})")

                        all_metrics.append(metrics)
                        print(f"    {method_name}: intersection={metrics['intersection_voxels']}, "
                              f"ROI_coverage={metrics['roi_coverage_ratio']:.2%}")
                except Exception as e:
                    print(f"    ❌ {method_name}: Error computing metrics - {e}")

        # Create overlay comparison (individual files per method)
        print(f"  Creating visualizations...")
        try:
            saved_files = create_overlay_comparison(
                roi_mask,
                list(bold_refs.values()),
                list(bold_refs.keys()),
                roi_name,
                output_dir,
                args.run
            )
        except Exception as e:
            print(f"  ❌ ERROR creating overlay: {e}")
            continue

        print()

    # Save alignment metrics to CSV (per subject-run)
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        metrics_csv = output_dir / f'roi_alignment_metrics_run{args.run}.csv'
        df_metrics.to_csv(metrics_csv, index=False)
        print(f"\n✅ Alignment metrics saved to: {metrics_csv}")

        # Print summary
        print(f"\n{'='*80}")
        print("Alignment Metrics Summary")
        print(f"{'='*80}")
        summary_cols = ['method', 'roi', 'roi_voxels', 'intersection_voxels',
                       'roi_coverage_ratio', 'bold_in_roi_ratio', 'signal_ratio_roi_vs_brain',
                       'glm_n_valid_voxels', 'glm_valid_ratio']
        # Only include GLM columns if they exist
        available_cols = [col for col in summary_cols if col in df_metrics.columns]
        print(df_metrics[available_cols].to_string(index=False))
        print()

        # Update consolidated results.json at method level
        method_results_path = output_base_dir / 'results.json'

        # Load existing results or create new
        if method_results_path.exists():
            with open(method_results_path, 'r') as f:
                method_results = json.load(f)
        else:
            method_results = {
                'method': args.method,
                'timestamp_created': datetime.now().isoformat(),
                'subjects': {}
            }

        # Update timestamp
        method_results['timestamp_updated'] = datetime.now().isoformat()

        # Initialize subject if not exists
        subject_key = f'sub-{args.subject}'
        if subject_key not in method_results['subjects']:
            method_results['subjects'][subject_key] = {
                'runs': {}
            }

        # Build run results
        run_key = f'run-{args.run}'
        run_results = {
            'metadata': {
                'roi_threshold': args.threshold,
                'rois_analyzed': args.rois,
                'timestamp': datetime.now().isoformat()
            },
            'summary': {
                'total_rois': len(df_metrics),
                'mean_roi_coverage_ratio': float(df_metrics['roi_coverage_ratio'].mean()),
                'mean_signal_ratio_roi_vs_brain': float(df_metrics['signal_ratio_roi_vs_brain'].mean())
            },
            'per_roi_metrics': {}
        }

        # Add GLM summary if available
        if 'glm_n_valid_voxels' in df_metrics.columns:
            run_results['summary']['total_glm_roi_voxels'] = int(df_metrics['glm_n_roi_voxels'].sum())
            run_results['summary']['total_glm_valid_voxels'] = int(df_metrics['glm_n_valid_voxels'].sum())
            run_results['summary']['mean_glm_valid_ratio'] = float(df_metrics['glm_valid_ratio'].mean())
            run_results['summary']['mean_glm_amplitude'] = float(df_metrics['glm_mean_amplitude'].mean())
            run_results['summary']['mean_glm_variance_across_colors'] = float(df_metrics['glm_mean_variance_across_colors'].mean())

        # Per-ROI detailed metrics
        for _, row in df_metrics.iterrows():
            roi_name = row['roi']
            run_results['per_roi_metrics'][roi_name] = {
                'roi_voxels': int(row['roi_voxels']),
                'intersection_voxels': int(row['intersection_voxels']),
                'roi_coverage_ratio': float(row['roi_coverage_ratio']),
                'bold_in_roi_ratio': float(row['bold_in_roi_ratio']),
                'mean_signal_in_roi': float(row['mean_signal_in_roi']),
                'signal_ratio_roi_vs_brain': float(row['signal_ratio_roi_vs_brain'])
            }

            # Add GLM metrics if available
            if 'glm_n_valid_voxels' in row:
                run_results['per_roi_metrics'][roi_name]['glm'] = {
                    'n_roi_voxels': int(row['glm_n_roi_voxels']),
                    'n_valid_voxels': int(row['glm_n_valid_voxels']),
                    'valid_ratio': float(row['glm_valid_ratio']),
                    'mean_amplitude': float(row['glm_mean_amplitude']),
                    'std_amplitude': float(row['glm_std_amplitude']),
                    'mean_variance_across_colors': float(row['glm_mean_variance_across_colors'])
                }

        # Update the consolidated results
        method_results['subjects'][subject_key]['runs'][run_key] = run_results

        # Save consolidated results.json
        with open(method_results_path, 'w') as f:
            json.dump(method_results, f, indent=2)
        print(f"✅ Consolidated results updated: {method_results_path}")

    print("="*80)
    print("Diagnosis Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Method results: {output_base_dir / 'results.json'}")
    print()


if __name__ == '__main__':
    main()
