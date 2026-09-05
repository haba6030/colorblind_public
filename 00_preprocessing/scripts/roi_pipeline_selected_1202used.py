#!/usr/bin/env python3
"""
Comprehensive ROI Pipeline - Wang Atlas to Functional Space
Tests all parameter combinations and generates validation reports
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase1_procrustes_decoding/roi_pipeline_selected_1202used.py  (development repository, commit 53c81c2)
# Role      : ROI extraction (Wang atlas V1, V2, V3, hV4 in MNI space).
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


import os
import sys
import argparse
import json
import itertools
from pathlib import Path
import numpy as np
import nibabel as nib
from nilearn import image, plotting
from nilearn.maskers import NiftiMasker
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0  # Disable max figure warning
import pandas as pd
from datetime import datetime

# Parse command-line arguments (must be before global variables that use dataset)
parser = argparse.ArgumentParser(description='Comprehensive ROI Pipeline')
parser.add_argument('subject_id', type=str, help='Subject ID (e.g., 01, 02, ...)')
parser.add_argument('--run', type=int, default=1, help='Run ID (default: 1)')
parser.add_argument('--dataset', type=str, default='method3_header_mi',
                    choices=['original_v3', 'method3_header_mi', 'deoblique_v2'],
                    help='Dataset to use (default: method3_header_mi)')
args = parser.parse_args()

# Dataset configuration
DATASET_PATHS = {
    'original_v3': '/storage/connectome/haba6030/fmriprep_out_original_v3',
    'method3_header_mi': '/storage/connectome/haba6030/fmriprep_out_method3_header_mi',
    'deoblique_v2': '/storage/connectome/haba6030/fmriprep_out_deoblique_v2'
}

# Configuration
BASE_DIR = Path('/scratch/connectome/haba6030/colorBlind')

# UPDATED 2025-12-12: Use 2mm atlas (pre-converted to match fMRIPrep MNI space)
# Original 1mm atlas caused alignment issues due to resolution/affine mismatch
# See: ROI_ALIGNMENT_ISSUES_ANALYSIS.md and convert_wang_atlas_to_2mm.py
ATLAS_DIR = BASE_DIR / 'ProbAtlas_v4_2mm' / 'subj_vol_all'  # Wang atlas 2mm (aligned)
# ATLAS_DIR = BASE_DIR / 'ProbAtlas_v4' / 'subj_vol_all'  # Old 1mm atlas (DO NOT USE)
FMRIPREP_DIR = Path(DATASET_PATHS[args.dataset])  # Dataset-specific fMRIPrep directory
DATA_DIR = Path('/storage/connectome/haba6030/bids_editted')  # Event/stimulus files

print(f"Dataset: {args.dataset}")
print(f"fMRIPrep directory: {FMRIPREP_DIR}")

# Output structure: analysis/roi_masks/{dataset}/sub-XX/  (SHARED RESOURCE)
# This replaces old: derivatives/V3_Comprehensive/ROI_mask/sub-XX/roi_pipeline/
DERIVATIVES_DIR = BASE_DIR / 'analysis' / 'roi_masks' / args.dataset
LOGS_DIR = BASE_DIR / 'analysis' / 'phase1_preprocess_decoding' / args.dataset / 'logs'
DERIVATIVES_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

print(f"ROI masks output: {DERIVATIVES_DIR}/")
print(f"Logs output: {LOGS_DIR}/")

# Target space (matching pilot preprocessing)
TARGET_SPACE = 'MNI152NLin2009cAsym'
TARGET_RES = 2  # mm
EXPECTED_SHAPE = (97, 115, 97)  # res-2 MNI space

# Wang atlas ROI definitions (based on Wang et al. 2015)
# ROI mapping from labels.txt:
# 01 - V1v, 02 - V1d, 03 - V2v, 04 - V2d, 05 - V3v, 06 - V3d, 07 - hV4
ROI_DEFINITIONS = {
    'V1': {
        'files': [
            'perc_VTPM_vol_roi1_lh.nii.gz',  # V1v left
            'perc_VTPM_vol_roi1_rh.nii.gz',  # V1v right
            'perc_VTPM_vol_roi2_lh.nii.gz',  # V1d left
            'perc_VTPM_vol_roi2_rh.nii.gz'   # V1d right
        ],
        'description': 'V1v + V1d (bilateral)'
    },
    'V2': {
        'files': [
            'perc_VTPM_vol_roi3_lh.nii.gz',  # V2v left
            'perc_VTPM_vol_roi3_rh.nii.gz',  # V2v right
            'perc_VTPM_vol_roi4_lh.nii.gz',  # V2d left
            'perc_VTPM_vol_roi4_rh.nii.gz'   # V2d right
        ],
        'description': 'V2v + V2d (bilateral)'
    },
    'V3': {
        'files': [
            'perc_VTPM_vol_roi5_lh.nii.gz',  # V3v left
            'perc_VTPM_vol_roi5_rh.nii.gz',  # V3v right
            'perc_VTPM_vol_roi6_lh.nii.gz',  # V3d left
            'perc_VTPM_vol_roi6_rh.nii.gz'   # V3d right
        ],
        'description': 'V3v + V3d (bilateral)'
    },
    'hV4': {
        'files': [
            'perc_VTPM_vol_roi7_lh.nii.gz',  # hV4 left
            'perc_VTPM_vol_roi7_rh.nii.gz'   # hV4 right
        ],
        'description': 'hV4 (bilateral)'
    }
}

# Parameter grid for testing
# NOTE: Wang atlas stores probabilities as 0-100 (percentage), not 0-1!
# UPDATED for deoblique data: Testing multiple brain_mask configurations
PARAM_GRID = {
    'threshold': [50],  # Percentage values (0-100 scale)
    # 5 = 5%, 10 = 10%, 20 = 20%, 35 = 35% (past default), 50 = 50%
    'interpolation': ['nearest'],
    'binarize_after_resample': [True],
    'brain_mask_type': ['none', 'func'],  # Compare no mask vs functional mask
    # 'none': No masking (baseline)
    # 'func': Use functional brain mask (recommended for deoblique)
    # 'epi_intersect': Use EPI brain mask intersection (most conservative)
    'use_gm_probseg': [True],  # GM probseg intersection (threshold=0.35)
    'use_subject_roi': [True, False]  # Compare with/without subject-specific ROI
}

# Constants for GM probseg
DEFAULT_GM_PROB_THRESHOLD = 0.35  # 35% GM probability (from past code)


class ROIPipelineComprehensive:
    """Comprehensive ROI construction pipeline with parameter search"""

    def __init__(self, subject_id, run_id=1):
        self.subject_id = subject_id
        self.run_id = run_id

        # Setup paths
        self.setup_paths()

        # Results storage
        self.results = []

    def setup_paths(self):
        """Setup all necessary paths"""
        # Subject directories - all subjects now use standard structure
        # Subject IDs: 01-07 (Non-CVD), 08-10 (CVD)
        self.fmriprep_subj_dir = FMRIPREP_DIR / f'sub-{self.subject_id}'
        self.deriv_subj_dir = DERIVATIVES_DIR / f'sub-{self.subject_id}'

        # Create output directory for this analysis
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.output_dir = self.deriv_subj_dir / f'roi_pipeline'
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Reference images
        self.func_ref = self._get_func_ref()
        self.anat_ref = self._get_anat_ref()
        self.func_brain_mask = self._get_func_brain_mask()
        self.epi_brain_mask = self._get_epi_brain_mask()

        print(f"Subject: {self.subject_id}")
        print(f"Output directory: {self.output_dir}")
        print(f"Functional reference: {self.func_ref}")
        print(f"Anatomical reference: {self.anat_ref}")
        print(f"Functional brain mask: {self.func_brain_mask}")
        print(f"EPI brain mask: {self.epi_brain_mask}")

    def _get_func_ref(self):
        """Get functional reference image (boldref)

        Uses robust glob pattern to handle variable naming conventions:
        - sub-01_task-rsvp_run-1_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz
        - sub-01_task-rsvp_run-1_space-MNI152NLin2009cAsym_res-2_desc-preproc_boldref.nii.gz
        """
        func_dir = self.fmriprep_subj_dir / 'func'
        if not func_dir.exists():
            raise FileNotFoundError(f"Functional directory not found: {func_dir}")

        # Try exact pattern first
        pattern = f'sub-{self.subject_id}_task-rsvp_run-{self.run_id}_space-{TARGET_SPACE}_res-{TARGET_RES}_boldref.nii.gz'
        func_file = func_dir / pattern

        if func_file.exists():
            return func_file

        # Try glob pattern to handle different naming conventions
        import glob
        glob_pattern = str(func_dir / f'sub-{self.subject_id}_task-rsvp_run-{self.run_id}_space-{TARGET_SPACE}_res-{TARGET_RES}_*boldref.nii.gz')
        matches = glob.glob(glob_pattern)

        if matches:
            print(f"  ✓ Found boldref with glob pattern: {matches[0]}")
            return Path(matches[0])

        raise FileNotFoundError(
            f"Functional reference not found.\n"
            f"  Tried exact: {func_file}\n"
            f"  Tried glob: {glob_pattern}\n"
            f"  Available files in {func_dir}:\n" +
            "\n".join([f"    - {f.name}" for f in func_dir.glob('*boldref.nii.gz')])
        )

    def _get_anat_ref(self):
        """Get anatomical reference image (T1w in MNI space)

        Uses robust glob pattern to handle variable naming conventions:
        - sub-01_acq-mprage_desc-preproc_T1w.nii.gz
        - sub-06_desc-preproc_T1w.nii.gz
        - sub-XX_space-MNI152NLin2009cAsym_res-2_desc-preproc_T1w.nii.gz
        """
        # Pilot is stored as sub-01 in pilot folder
        subj_label = '01' if self.subject_id == 'P01' else self.subject_id

        # Look for T1w in anat directory
        anat_dir = self.fmriprep_subj_dir / 'anat'
        if not anat_dir.exists():
            print(f"Warning: Anat directory not found: {anat_dir}")
            return None

        # Robust pattern: sub-{ID}_*_space-{SPACE}_res-{RES}_*_desc-preproc_T1w.nii.gz
        # Catches any intermediate fields (ses-1, acq-mprage, etc.)
        pattern = f'sub-{subj_label}_*space-{TARGET_SPACE}_res-{TARGET_RES}_*desc-preproc_T1w.nii.gz'
        anat_files = list(anat_dir.glob(pattern))

        if not anat_files:
            print(f"Warning: Anatomical reference not found with pattern: {pattern}")
            return None

        if len(anat_files) > 1:
            print(f"Warning: Multiple anatomical files found, using first: {anat_files[0].name}")

        return anat_files[0]

    def _get_func_brain_mask(self):
        """Get functional brain mask"""
        # Pilot is stored as sub-01 in pilot folder
        subj_label = '01' if self.subject_id == 'P01' else self.subject_id
        pattern = f'sub-{subj_label}_task-rsvp_run-{self.run_id}_space-{TARGET_SPACE}_res-{TARGET_RES}_desc-brain_mask.nii.gz'

        mask_file = self.fmriprep_subj_dir / 'func' / pattern
        if not mask_file.exists():
            print(f"Warning: Functional brain mask not found: {mask_file}")
            return None
        return mask_file

    def _get_epi_brain_mask(self):
        """Get EPI-based brain mask from actual signal coverage

        Creates a mask from boldref image showing where EPI actually has signal.
        This is more conservative than anatomical brain mask because it excludes
        areas with signal dropout (e.g., orbitofrontal cortex, temporal poles).
        """
        # We'll compute this from boldref (functional reference image)
        # Instead of using a pre-existing file, we create it on-the-fly
        return self.func_ref  # Return boldref path; mask will be computed in apply_brain_mask

    def _get_gm_probseg(self):
        """Get GM (gray matter) probability segmentation

        Returns path to GM probseg file from fMRIPrep anatomical outputs.
        """
        # Pilot is stored as sub-01 in pilot folder
        subj_label = '01' if self.subject_id == 'P01' else self.subject_id

        # Look for GM probseg in anat directory
        anat_dir = self.fmriprep_subj_dir / 'anat'
        if not anat_dir.exists():
            print(f"Warning: Anat directory not found: {anat_dir}")
            return None

        # Search for probseg file with 'GM' or 'gm' in name
        pattern = f'sub-{subj_label}_*probseg*'
        probseg_files = list(anat_dir.glob(pattern))

        gm_files = [f for f in probseg_files if 'gm' in f.name.lower() or 'label-GM' in f.name]

        if not gm_files:
            print(f"Warning: GM probseg not found in {anat_dir}")
            return None

        # Prefer MNI space if available
        mni_files = [f for f in gm_files if 'MNI' in f.name]
        selected = mni_files[0] if mni_files else gm_files[0]

        print(f"Found GM probseg: {selected.name}")
        return selected

    def _get_subject_roi(self, roi_name):
        """Get subject-specific ROI mask in MNI space

        Returns path to subject's individual ROI if available.
        """
        # Pilot is stored as sub-01 in pilot folder
        subj_label = '01' if self.subject_id == 'P01' else self.subject_id

        # Look for subject ROI in anat directory
        anat_dir = self.fmriprep_subj_dir / 'anat'
        if not anat_dir.exists():
            return None

        # Search for files matching ROI name
        roi_lower = roi_name.lower()
        candidates = []

        for f in anat_dir.glob('*.nii.gz'):
            fname_lower = f.name.lower()
            # Look for ROI name in filename and MNI space
            if roi_lower in fname_lower and 'space-mni' in fname_lower:
                # Exclude brain_mask and probseg files
                if 'brain_mask' not in fname_lower and 'probseg' not in fname_lower:
                    candidates.append(f)

        if not candidates:
            return None

        # Return first match
        selected = sorted(candidates)[0]
        print(f"Found subject ROI for {roi_name}: {selected.name}")
        return selected

    def apply_brain_mask(self, roi_img, brain_mask_type):
        """Apply brain mask to ROI

        Parameters
        ----------
        roi_img : nibabel image
            ROI mask image
        brain_mask_type : str
            Type of brain mask ('none', 'func', 'epi_intersect')

        Returns
        -------
        masked_img : nibabel image
            ROI masked by brain mask
        overlap_metrics : dict
            Metrics about the overlap
        """
        roi_data = roi_img.get_fdata()
        original_voxels = np.sum(roi_data > 0)

        if brain_mask_type == 'none':
            return roi_img, {
                'brain_mask_type': 'none',
                'original_voxels': int(original_voxels),
                'masked_voxels': int(original_voxels),
                'overlap_ratio': 1.0
            }

        # Load appropriate brain mask
        if brain_mask_type == 'func':
            mask_file = self.func_brain_mask
        elif brain_mask_type == 'epi_intersect':
            mask_file = self.epi_brain_mask
        else:
            raise ValueError(f"Unknown brain_mask_type: {brain_mask_type}")

        if mask_file is None:
            print(f"Warning: Brain mask not available, skipping masking")
            return roi_img, {
                'brain_mask_type': brain_mask_type,
                'original_voxels': int(original_voxels),
                'masked_voxels': int(original_voxels),
                'overlap_ratio': 1.0,
                'warning': 'Brain mask not found'
            }

        # Load and apply mask
        brain_mask_img = nib.load(mask_file)
        brain_mask_data = brain_mask_img.get_fdata()

        # For EPI intersect, create mask from actual signal intensity
        if brain_mask_type == 'epi_intersect':
            # boldref intensity > 0 means valid EPI signal
            # Use a small threshold to exclude near-zero noise
            intensity_threshold = np.percentile(brain_mask_data[brain_mask_data > 0], 1) if np.any(brain_mask_data > 0) else 0
            brain_mask_data = (brain_mask_data > intensity_threshold).astype(np.float32)
            print(f"  EPI coverage: intensity threshold = {intensity_threshold:.2f}")

        # Ensure same shape
        if brain_mask_data.shape != roi_data.shape:
            print(f"Warning: Brain mask shape {brain_mask_data.shape} != ROI shape {roi_data.shape}")
            # Resample brain mask to match ROI
            brain_mask_img = image.resample_to_img(
                brain_mask_img,
                roi_img,
                interpolation='nearest',
                force_resample=True,
                copy_header=True
            )
            brain_mask_data = brain_mask_img.get_fdata()

        # Apply intersection
        masked_data = roi_data * (brain_mask_data > 0)
        masked_voxels = np.sum(masked_data > 0)
        overlap_ratio = masked_voxels / original_voxels if original_voxels > 0 else 0.0

        # Create masked image
        masked_img = nib.Nifti1Image(masked_data, roi_img.affine, roi_img.header)

        metrics = {
            'brain_mask_type': brain_mask_type,
            'original_voxels': int(original_voxels),
            'masked_voxels': int(masked_voxels),
            'removed_voxels': int(original_voxels - masked_voxels),
            'overlap_ratio': float(overlap_ratio)
        }

        return masked_img, metrics

    def apply_gm_probseg(self, roi_img):
        """Apply GM probability segmentation intersection

        Parameters
        ----------
        roi_img : nibabel image
            ROI mask image

        Returns
        -------
        masked_img : nibabel image
            ROI intersected with GM mask
        metrics : dict
            Metrics about the intersection
        """
        roi_data = roi_img.get_fdata()
        original_voxels = np.sum(roi_data > 0)

        # Load GM probseg
        gm_probseg_path = self._get_gm_probseg()
        if gm_probseg_path is None:
            print("  Skipping GM probseg intersection (not found)")
            return roi_img, {
                'gm_original_voxels': int(original_voxels),
                'gm_masked_voxels': int(original_voxels),
                'gm_overlap_ratio': 1.0,
                'gm_applied': False
            }

        # Load and resample GM probseg
        gm_img = nib.load(gm_probseg_path)

        # Resample to match ROI
        gm_resampled = image.resample_img(
            gm_img,
            target_affine=roi_img.affine,
            target_shape=roi_img.shape[:3],
            interpolation='linear',  # Linear for probability values
            force_resample=True,
            copy_header=True
        )

        gm_data = gm_resampled.get_fdata()

        # Apply threshold (35% GM probability)
        gm_mask = gm_data >= DEFAULT_GM_PROB_THRESHOLD

        # Apply intersection
        masked_data = roi_data * gm_mask
        masked_voxels = np.sum(masked_data > 0)

        if masked_voxels == 0:
            print("  Warning: GM intersection empty, retaining original ROI")
            return roi_img, {
                'gm_original_voxels': int(original_voxels),
                'gm_masked_voxels': int(original_voxels),
                'gm_overlap_ratio': 1.0,
                'gm_applied': False,
                'gm_warning': 'Empty intersection'
            }

        overlap_ratio = masked_voxels / original_voxels if original_voxels > 0 else 0.0

        # Create masked image
        masked_img = nib.Nifti1Image(masked_data, roi_img.affine, roi_img.header)

        print(f"  GM probseg: {original_voxels} → {masked_voxels} voxels ({overlap_ratio:.1%})")

        return masked_img, {
            'gm_original_voxels': int(original_voxels),
            'gm_masked_voxels': int(masked_voxels),
            'gm_removed_voxels': int(original_voxels - masked_voxels),
            'gm_overlap_ratio': float(overlap_ratio),
            'gm_applied': True
        }

    def apply_subject_roi(self, roi_img, roi_name):
        """Apply subject-specific ROI intersection

        Parameters
        ----------
        roi_img : nibabel image
            ROI mask image
        roi_name : str
            Name of ROI (V1, V2, etc.)

        Returns
        -------
        masked_img : nibabel image
            ROI intersected with subject-specific ROI
        metrics : dict
            Metrics about the intersection
        """
        roi_data = roi_img.get_fdata()
        original_voxels = np.sum(roi_data > 0)

        # Load subject-specific ROI
        subj_roi_path = self._get_subject_roi(roi_name)
        if subj_roi_path is None:
            print(f"  Skipping subject ROI intersection for {roi_name} (not found)")
            return roi_img, {
                'subj_roi_original_voxels': int(original_voxels),
                'subj_roi_masked_voxels': int(original_voxels),
                'subj_roi_overlap_ratio': 1.0,
                'subj_roi_applied': False
            }

        # Load and resample subject ROI
        subj_roi_img = nib.load(subj_roi_path)

        # Resample to match atlas ROI
        subj_resampled = image.resample_img(
            subj_roi_img,
            target_affine=roi_img.affine,
            target_shape=roi_img.shape[:3],
            interpolation='nearest',  # Nearest for binary mask
            force_resample=True,
            copy_header=True
        )

        subj_data = subj_resampled.get_fdata()
        subj_mask = subj_data > 0

        # Apply intersection
        masked_data = roi_data * subj_mask
        masked_voxels = np.sum(masked_data > 0)

        if masked_voxels == 0:
            print(f"  Warning: Subject ROI intersection empty for {roi_name}, retaining atlas ROI")
            return roi_img, {
                'subj_roi_original_voxels': int(original_voxels),
                'subj_roi_masked_voxels': int(original_voxels),
                'subj_roi_overlap_ratio': 1.0,
                'subj_roi_applied': False,
                'subj_roi_warning': 'Empty intersection'
            }

        overlap_ratio = masked_voxels / original_voxels if original_voxels > 0 else 0.0

        # Create masked image
        masked_img = nib.Nifti1Image(masked_data, roi_img.affine, roi_img.header)

        print(f"  Subject ROI: {original_voxels} → {masked_voxels} voxels ({overlap_ratio:.1%})")

        return masked_img, {
            'subj_roi_original_voxels': int(original_voxels),
            'subj_roi_masked_voxels': int(masked_voxels),
            'subj_roi_removed_voxels': int(original_voxels - masked_voxels),
            'subj_roi_overlap_ratio': float(overlap_ratio),
            'subj_roi_applied': True
        }

    def load_and_combine_roi(self, roi_name, threshold):
        """Load and combine all atlas files for a given ROI"""
        roi_def = ROI_DEFINITIONS[roi_name]
        combined_prob = None

        for atlas_file in roi_def['files']:
            atlas_path = ATLAS_DIR / atlas_file
            if not atlas_path.exists():
                print(f"Warning: Atlas file not found: {atlas_path}")
                continue

            # Load probability map
            prob_img = nib.load(atlas_path)
            prob_data = prob_img.get_fdata()

            # Accumulate probabilities (union)
            if combined_prob is None:
                combined_prob = prob_data.copy()
                affine = prob_img.affine
                header = prob_img.header
            else:
                # Take maximum probability at each voxel (union)
                combined_prob = np.maximum(combined_prob, prob_data)

        if combined_prob is None:
            raise ValueError(f"No atlas files found for {roi_name}")

        # Apply threshold
        mask_data = (combined_prob >= threshold).astype(np.float32)

        # Create NIfTI image
        mask_img = nib.Nifti1Image(mask_data, affine, header)

        return mask_img, combined_prob

    def resample_to_target(self, mask_img, interpolation='nearest', binarize=True):
        """Resample mask to target functional space"""
        # Load functional reference
        func_ref_img = nib.load(self.func_ref)

        # Resample
        resampled = image.resample_img(
            mask_img,
            target_affine=func_ref_img.affine,
            target_shape=func_ref_img.shape[:3],
            interpolation=interpolation,
            force_resample=True,  # Force resampling even if affines match
            copy_header=True      # Preserve header information
        )

        # Binarize after resampling if requested
        if binarize:
            resampled_data = resampled.get_fdata()
            resampled_data = (resampled_data > 0.5).astype(np.float32)
            resampled = nib.Nifti1Image(
                resampled_data,
                resampled.affine,
                resampled.header
            )

        return resampled

    def validate_mask(self, mask_img, roi_name, params):
        """Validate mask and compute metrics"""
        mask_data = mask_img.get_fdata()

        # Basic metrics
        n_voxels = int(np.sum(mask_data > 0))
        total_voxels = int(np.prod(mask_data.shape))
        coverage_pct = 100 * n_voxels / total_voxels

        # Check shape
        shape_match = mask_data.shape == EXPECTED_SHAPE

        # Check affine match with functional reference
        func_ref_img = nib.load(self.func_ref)
        affine_match = np.allclose(mask_img.affine, func_ref_img.affine)

        # Compute center of mass
        if n_voxels > 0:
            coords = np.array(np.where(mask_data > 0))
            com = coords.mean(axis=1)
        else:
            com = [0, 0, 0]

        metrics = {
            'roi_name': roi_name,
            'subject_id': self.subject_id,
            'n_voxels': n_voxels,
            'total_voxels': total_voxels,
            'coverage_pct': coverage_pct,
            'shape': mask_data.shape,
            'shape_match': shape_match,
            'affine_match': affine_match,
            'center_of_mass': np.asarray(com).tolist(),  # Fixed: ensure numpy array before .tolist()
            **params  # Include all parameters
        }

        return metrics

    def create_visualizations(self, mask_img, prob_map, roi_name, params, metrics):
        """Create comprehensive visualizations"""
        param_str = f"thr{params['threshold']}_int{params['interpolation']}_bin{params['binarize_after_resample']}_mask{params['brain_mask_type']}_gm{params['use_gm_probseg']}_subj{params['use_subject_roi']}"

        # Create figure directory
        fig_dir = self.output_dir / 'figures' / roi_name
        fig_dir.mkdir(parents=True, exist_ok=True)

        # 1. Glass brain plot
        fig = plt.figure(figsize=(15, 5))

        display = plotting.plot_glass_brain(
            mask_img,
            title=f"{roi_name} - {param_str}\nVoxels: {metrics['n_voxels']}",
            display_mode='lyrz',
            colorbar=True,
            figure=fig
        )

        glass_path = fig_dir / f'{roi_name}_glass_{param_str}.png'
        plt.savefig(glass_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the specific figure
        plt.clf()       # Clear current figure

        # 2. Overlay on functional reference
        func_ref_img = nib.load(self.func_ref)

        fig, axes = plt.subplots(3, 1, figsize=(15, 12))

        for i, cut_coords in enumerate([None, None, None]):
            display = plotting.plot_roi(
                mask_img,
                bg_img=func_ref_img,
                cut_coords=cut_coords,
                display_mode='ortho',
                title=f"{roi_name} on Functional Reference - {param_str}",
                axes=axes[i] if i == 0 else None,
                draw_cross=True
            )

        overlay_func_path = fig_dir / f'{roi_name}_overlay_func_{param_str}.png'
        plt.savefig(overlay_func_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the specific figure
        plt.clf()       # Clear current figure

        # 3. Overlay on anatomical reference (if available)
        if self.anat_ref is not None:
            anat_ref_img = nib.load(self.anat_ref)

            fig = plt.figure(figsize=(15, 5))
            display = plotting.plot_roi(
                mask_img,
                bg_img=anat_ref_img,
                display_mode='ortho',
                title=f"{roi_name} on Anatomical Reference - {param_str}",
                draw_cross=True,
                figure=fig
            )

            overlay_anat_path = fig_dir / f'{roi_name}_overlay_anat_{param_str}.png'
            plt.savefig(overlay_anat_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Close the specific figure
            plt.clf()       # Clear current figure

        # 4. Probability map distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        prob_values = prob_map.flatten()
        prob_values = prob_values[prob_values > 0]

        ax.hist(prob_values, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(params['threshold'], color='red', linestyle='--',
                   linewidth=2, label=f"Threshold = {params['threshold']}")
        ax.set_xlabel('Probability Value')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{roi_name} - Probability Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        prob_hist_path = fig_dir / f'{roi_name}_prob_hist_{param_str}.png'
        plt.savefig(prob_hist_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the specific figure
        plt.clf()       # Clear current figure

        # Close all remaining figures to free memory
        plt.close('all')

        return {
            'glass_brain': str(glass_path),
            'overlay_func': str(overlay_func_path),
            'overlay_anat': str(overlay_anat_path) if self.anat_ref else None,
            'prob_histogram': str(prob_hist_path)
        }

    def run_single_combination(self, roi_name, threshold, interpolation, binarize_after_resample,
                               brain_mask_type, use_gm_probseg, use_subject_roi):
        """Run pipeline for a single parameter combination"""
        params = {
            'threshold': threshold,
            'interpolation': interpolation,
            'binarize_after_resample': binarize_after_resample,
            'brain_mask_type': brain_mask_type,
            'use_gm_probseg': use_gm_probseg,
            'use_subject_roi': use_subject_roi
        }

        print(f"\n{'='*60}")
        print(f"Processing {roi_name} with parameters:")
        print(f"  Threshold: {threshold}")
        print(f"  Interpolation: {interpolation}")
        print(f"  Binarize after resample: {binarize_after_resample}")
        print(f"  Brain mask type: {brain_mask_type}")
        print(f"  Use GM probseg: {use_gm_probseg}")
        print(f"  Use subject ROI: {use_subject_roi}")
        print(f"{'='*60}")

        # 1. Load and combine ROI
        print("Step 1: Loading and combining atlas files...")
        mask_img, prob_map = self.load_and_combine_roi(roi_name, threshold)

        # 2. Resample to target space
        print("Step 2: Resampling to target functional space...")
        resampled_mask = self.resample_to_target(
            mask_img,
            interpolation=interpolation,
            binarize=binarize_after_resample
        )

        # 3. Apply brain mask
        print(f"Step 3: Applying brain mask ({brain_mask_type})...")
        masked_roi, mask_metrics = self.apply_brain_mask(resampled_mask, brain_mask_type)

        # 4. Apply GM probseg (if enabled)
        if use_gm_probseg:
            print("Step 4: Applying GM probseg intersection...")
            masked_roi, gm_metrics = self.apply_gm_probseg(masked_roi)
        else:
            gm_metrics = {'gm_applied': False}

        # 5. Apply subject-specific ROI (if enabled)
        if use_subject_roi:
            print("Step 5: Applying subject-specific ROI intersection...")
            masked_roi, subj_roi_metrics = self.apply_subject_roi(masked_roi, roi_name)
        else:
            subj_roi_metrics = {'subj_roi_applied': False}

        # 6. Validate
        print("Step 6: Validating mask...")
        metrics = self.validate_mask(masked_roi, roi_name, params)
        # Add all mask metrics
        metrics.update(mask_metrics)
        metrics.update(gm_metrics)
        metrics.update(subj_roi_metrics)

        # 7. Visualize
        print("Step 7: Creating visualizations...")
        viz_paths = self.create_visualizations(
            masked_roi,
            prob_map,
            roi_name,
            params,
            metrics
        )

        # 8. Save mask
        param_str = f"thr{threshold}_int{interpolation}_bin{binarize_after_resample}_mask{brain_mask_type}_gm{use_gm_probseg}_subj{use_subject_roi}"
        mask_path = self.output_dir / f'{roi_name}_mask_{param_str}.nii.gz'
        nib.save(masked_roi, mask_path)

        print(f"  ✓ Original voxels: {metrics.get('original_voxels', 'N/A')}")
        print(f"  ✓ After masking: {metrics['n_voxels']} (overlap: {metrics.get('overlap_ratio', 1.0):.2%})")
        print(f"  ✓ Coverage: {metrics['coverage_pct']:.4f}%")
        print(f"  ✓ Shape match: {metrics['shape_match']}")
        print(f"  ✓ Affine match: {metrics['affine_match']}")
        print(f"  ✓ Mask saved to: {mask_path}")

        # Store results
        result = {
            **metrics,
            'mask_path': str(mask_path),
            'visualizations': viz_paths
        }

        return result

    def run_all_combinations(self):
        """Run pipeline for all ROIs and parameter combinations"""
        print(f"\n{'#'*70}")
        print(f"# Starting Comprehensive ROI Pipeline")
        print(f"# Subject: {self.subject_id}")
        print(f"# Output: {self.output_dir}")
        print(f"{'#'*70}\n")

        # Generate all parameter combinations
        param_combinations = list(itertools.product(
            PARAM_GRID['threshold'],
            PARAM_GRID['interpolation'],
            PARAM_GRID['binarize_after_resample'],
            PARAM_GRID['brain_mask_type'],
            PARAM_GRID['use_gm_probseg'],
            PARAM_GRID['use_subject_roi']
        ))

        total_combinations = len(ROI_DEFINITIONS) * len(param_combinations)
        print(f"Total combinations to test: {total_combinations}")
        print(f"  ROIs: {list(ROI_DEFINITIONS.keys())}")
        print(f"  Thresholds: {PARAM_GRID['threshold']}")
        print(f"  Interpolations: {PARAM_GRID['interpolation']}")
        print(f"  Binarize options: {PARAM_GRID['binarize_after_resample']}")
        print(f"  Brain mask types: {PARAM_GRID['brain_mask_type']}")
        print(f"  Use GM probseg: {PARAM_GRID['use_gm_probseg']}")
        print(f"  Use subject ROI: {PARAM_GRID['use_subject_roi']}\n")

        # Run all combinations
        for roi_name in ROI_DEFINITIONS.keys():
            for threshold, interpolation, binarize, brain_mask_type, use_gm, use_subj in param_combinations:
                try:
                    result = self.run_single_combination(
                        roi_name,
                        threshold,
                        interpolation,
                        binarize,
                        brain_mask_type,
                        use_gm,
                        use_subj
                    )
                    self.results.append(result)
                except Exception as e:
                    print(f"ERROR processing {roi_name} with params "
                          f"(thr={threshold}, int={interpolation}, bin={binarize}, "
                          f"mask={brain_mask_type}, gm={use_gm}, subj={use_subj}): {e}")
                    import traceback
                    traceback.print_exc()

        # Save results
        self.save_results()

        # Generate comparison report
        self.generate_comparison_report()

        print(f"\n{'#'*70}")
        print(f"# Pipeline Complete!")
        print(f"# Results saved to: {self.output_dir}")
        print(f"{'#'*70}\n")

    def save_results(self):
        """Save results to JSON and CSV"""
        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Save CSV
        csv_path = self.output_dir / 'results_summary.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

        # Save JSON (with full details)
        json_path = self.output_dir / 'results_full.json'
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"Full results saved to: {json_path}")

    def generate_comparison_report(self):
        """Generate comparison report and visualizations"""
        df = pd.DataFrame(self.results)

        report_path = self.output_dir / 'COMPARISON_REPORT.md'

        with open(report_path, 'w') as f:
            f.write(f"# ROI Pipeline Comparison Report\n\n")
            f.write(f"**Subject:** {self.subject_id}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Output Directory:** `{self.output_dir}`\n\n")

            f.write(f"## Summary Statistics\n\n")
            f.write(f"- Total combinations tested: {len(self.results)}\n")
            f.write(f"- ROIs: {', '.join(ROI_DEFINITIONS.keys())}\n\n")

            # Voxel count comparison by ROI
            f.write(f"## Voxel Counts by ROI and Parameters\n\n")
            for roi_name in ROI_DEFINITIONS.keys():
                roi_df = df[df['roi_name'] == roi_name]
                f.write(f"### {roi_name}\n\n")
                f.write(f"| Threshold | Interpolation | Binarize | Voxels | Coverage % |\n")
                f.write(f"|-----------|---------------|----------|--------|------------|\n")
                for _, row in roi_df.iterrows():
                    f.write(f"| {row['threshold']:.2f} | {row['interpolation']} | "
                           f"{row['binarize_after_resample']} | {row['n_voxels']} | "
                           f"{row['coverage_pct']:.4f} |\n")
                f.write(f"\n")

            # Best configurations
            f.write(f"## Recommended Configurations\n\n")
            f.write(f"Based on voxel counts and coverage:\n\n")
            for roi_name in ROI_DEFINITIONS.keys():
                roi_df = df[df['roi_name'] == roi_name]
                best_row = roi_df.loc[roi_df['n_voxels'].idxmax()]
                f.write(f"### {roi_name}\n")
                f.write(f"- **Threshold:** {best_row['threshold']}\n")
                f.write(f"- **Interpolation:** {best_row['interpolation']}\n")
                f.write(f"- **Binarize after resample:** {best_row['binarize_after_resample']}\n")
                f.write(f"- **Voxels:** {best_row['n_voxels']}\n")
                f.write(f"- **Coverage:** {best_row['coverage_pct']:.4f}%\n")
                f.write(f"- **Mask file:** `{best_row['mask_path']}`\n\n")

        print(f"\nComparison report saved to: {report_path}")

        # Create comparison plots
        self.create_comparison_plots(df)

    def create_comparison_plots(self, df):
        """Create comparison plots across parameters"""
        plot_dir = self.output_dir / 'comparison_plots'
        plot_dir.mkdir(exist_ok=True)

        # Plot 1: Voxel count vs threshold for each ROI
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        for idx, roi_name in enumerate(ROI_DEFINITIONS.keys()):
            ax = axes[idx]
            roi_df = df[df['roi_name'] == roi_name]

            for interp in PARAM_GRID['interpolation']:
                for binarize in PARAM_GRID['binarize_after_resample']:
                    subset = roi_df[
                        (roi_df['interpolation'] == interp) &
                        (roi_df['binarize_after_resample'] == binarize)
                    ]
                    label = f"{interp}, bin={binarize}"
                    ax.plot(subset['threshold'], subset['n_voxels'],
                           marker='o', label=label)

            ax.set_xlabel('Threshold')
            ax.set_ylabel('Number of Voxels')
            ax.set_title(f'{roi_name} - Voxel Count vs Threshold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_dir / 'voxel_count_comparison.png', dpi=150)
        plt.close()

        print(f"Comparison plots saved to: {plot_dir}")


def main():
    """Main entry point"""
    # Arguments already parsed at top of file (args)
    # Run pipeline with dataset specification
    pipeline = ROIPipelineComprehensive(args.subject_id, args.run)
    pipeline.run_all_combinations()


if __name__ == '__main__':
    main()
