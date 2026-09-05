#!/usr/bin/env python3
"""
exp2 (2nd MRI, filter validation) — per-CONDITION C010 amplitude extraction.

Adapts the FROZEN canonical C010 recipe
(analysis/phase1_procrustes_decoding/run_full_dataset_C010.py)
to the 2-condition exp2 design WITHOUT touching the frozen exp1 script.

Recipe held bit-for-bit identical to exp1:
  - FIR 1st-level (8 delays x TR) -> ROI-mean HRF + np.gradient deriv
  - 2nd-level PER-RUN fit: 8 HRF + 8 deriv + per-run drift (linear+const)
  - betas[:8] = color amplitudes
  - onset correction events.onset -= 3*TR  (3 dropped vols)
  - numpy-index masking (NOT nilearn apply_mask)

exp2-specific changes (only these):
  1. 8 runs. FIR HRF estimated across ALL 8 runs (one HRF; vasculature is
     condition-invariant). Per-run amplitudes -> (8, 8, V).
  2. ROI mask = masknone_gmTrue (atlas n GM, session-independent) intersected
     with the exp2 functional coverage (intersection of all 8 run brain masks)
     -> replicates the exp1 'maskfunc' recipe for exp2 coverage.
  3. Condition split (ABBA WOOWWOOW): Window = runs {1,4,5,8},
     Optimal = runs {2,3,6,7}.
  4. PER-CONDITION Procrustes: each condition's 4 runs aligned to its OWN
     run-0 (mirrors paper within-subject alignment). -> (4, 8, V) per condition.

Output layout (each condition dir mirrors a canonical subject dir):
  derivatives/full_dataset_C010_exp2/sub-{ID}/{window|optimal}/{ROI}/
      amplitudes_procrustes.npy   (4, 8, V)
      amplitudes_raw.npy          (4, 8, V)
      procrustes_disparities.npy  (4,)
      roi_hrf.npy / roi_hrf_deriv.npy / config.json
  plus sub-{ID}/_all_runs/{ROI}/amplitudes_raw_8run.npy  (8, 8, V) for reference
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp2_C010_conditions.py  (development repository, commit 53c81c2)
# Role      : Condition-wise (window / optimal) amplitude estimation for session 2 (ABBA run map).
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
from scipy.linalg import orthogonal_procrustes
import json
import sys
import argparse

# ============================================================================
# Configuration (exp2 paths)
# ============================================================================
BASE_DIR = Path("/scratch/connectome/haba6030/colorBlind")
FMRIPREP_DIR = Path("/storage/connectome/haba6030/fmriprep_out_method3_2nd")
EVENT_DIR = Path("/storage/connectome/haba6030/bids_2nd")
ROI_MASKS_DIR = BASE_DIR / "analysis" / "roi_masks" / "method3_header_mi"
# OUTPUT_DIR set in main() per mask variant:
#   native  -> full_dataset_C010_exp2          (masknone n exp2-coverage, exp2's full coverage)
#   matched -> full_dataset_C010_exp2_matched  (exp1 maskfunc voxels -> voxel-matched to no-filter anchor)
OUTPUT_DIR = None
MASK_VARIANT = 'native'

TR = 1.5
N_RUNS = 8
N_COLORS = 8
FIR_DELAYS = np.arange(8) * TR

# Condition map (1-indexed run numbers), ABBA counterbalancing — MIRRORED across subjects.
#   sub-08 (deutan): W O O W W O O W  -> window  {1,4,5,8}, optimal {2,3,6,7}
#   sub-09 (protan): O W W O O W W O  -> optimal {1,4,5,8}, window  {2,3,6,7}  (mirror)
# sub-09 map verified per-run from ses-1_colorDetect_run-N_info.json "Filter condition".
CONDITION_RUNS_BY_SUBJECT = {
    "08": {"window": [1, 4, 5, 8], "optimal": [2, 3, 6, 7]},
    "09": {"window": [2, 3, 6, 7], "optimal": [1, 4, 5, 8]},
}

# C010: drift only, NO confounds (identical to exp1)
MOTION_TISSUE = False
WM_ACOMPCOR = False


# ============================================================================
# ROI mask (exp2): masknone_gmTrue (atlas n GM) n exp2 coverage
# ============================================================================
def get_atlas_gm_mask_path(subject_id, roi_name):
    roi_map = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}
    roi_prefix = roi_map.get(roi_name)
    if roi_prefix is None:
        raise ValueError(f"Unknown ROI: {roi_name}")
    # session-INDEPENDENT atlas n GM mask (no functional intersection baked in)
    fname = f"{roi_prefix}_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjFalse.nii.gz"
    return ROI_MASKS_DIR / f"sub-{subject_id}" / "roi_pipeline" / fname


def bold_path(subject_id, run_idx):
    return FMRIPREP_DIR / f"sub-{subject_id}" / "func" / \
        f"sub-{subject_id}_task-rsvp_run-{run_idx}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"


def brainmask_path(subject_id, run_idx):
    return FMRIPREP_DIR / f"sub-{subject_id}" / "func" / \
        f"sub-{subject_id}_task-rsvp_run-{run_idx}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"


def get_maskfunc_path(subject_id, roi_name):
    """exp1 functional-masked ROI (atlas n GM n exp1-func-coverage). Used by
    'matched' variant to voxel-match exp2 conditions to the exp1 no-filter anchor."""
    roi_map = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}
    pref = roi_map[roi_name]
    fname = f"{pref}_mask_thr50_intnearest_binTrue_maskfunc_gmTrue_subjFalse.nii.gz"
    return ROI_MASKS_DIR / f"sub-{subject_id}" / "roi_pipeline" / fname


def build_exp2_roi_mask(subject_id, roi_name):
    """native: atlas n GM  intersect  (all-8-run exp2 brain-mask coverage).
    matched: exp1 maskfunc directly (group-MNI grid; applied to exp2 BOLD) so
             Window/Optimal use the SAME voxels as the exp1 no-filter anchor."""
    if MASK_VARIANT == 'matched':
        mp = get_maskfunc_path(subject_id, roi_name)
        if not mp.exists():
            raise FileNotFoundError(f"exp1 maskfunc not found: {mp}")
        mi = nib.load(mp)
        # verify grid vs an exp2 BOLD/brainmask
        bm = nib.load(brainmask_path(subject_id, 1))
        if mi.shape[:3] != bm.shape[:3] or not np.allclose(mi.affine, bm.affine):
            raise ValueError("GRID MISMATCH exp1 maskfunc vs exp2 BOLD")
        mask = mi.get_fdata() > 0
        print(f"    ROI {roi_name} [matched]: exp1 maskfunc voxels={int(mask.sum())}")
        return mask

    atlas_path = get_atlas_gm_mask_path(subject_id, roi_name)
    if not atlas_path.exists():
        raise FileNotFoundError(f"atlas/GM mask not found: {atlas_path}")
    atlas_img = nib.load(atlas_path)
    atlas = atlas_img.get_fdata() > 0

    # exp2 coverage = voxels present in ALL 8 run brain masks
    coverage = None
    for r in range(1, N_RUNS + 1):
        bm_p = brainmask_path(subject_id, r)
        if not bm_p.exists():
            raise FileNotFoundError(f"brain mask not found: {bm_p}")
        bm_img = nib.load(bm_p)
        if bm_img.shape[:3] != atlas_img.shape[:3] or not np.allclose(bm_img.affine, atlas_img.affine):
            raise ValueError(
                f"GRID MISMATCH atlas vs brain mask run-{r}: "
                f"{atlas_img.shape} aff?={np.allclose(bm_img.affine, atlas_img.affine)}")
        bm = bm_img.get_fdata() > 0
        coverage = bm if coverage is None else (coverage & bm)

    mask = atlas & coverage
    n_atlas, n_final = int(atlas.sum()), int(mask.sum())
    print(f"    ROI {roi_name}: atlasnGM={n_atlas}  ->after exp2 coverage={n_final} "
          f"(dropped {n_atlas - n_final})")
    return mask  # 3D bool


def load_bold_data(subject_id, run_idx, roi_mask):
    func_file = bold_path(subject_id, run_idx)
    if not func_file.exists():
        raise FileNotFoundError(f"BOLD not found: {func_file}")
    func_img = nib.load(func_file)
    # CRITICAL: numpy indexing, NOT nilearn.masking.apply_mask
    func_data_4d = func_img.get_fdata()
    func_data = func_data_4d[roi_mask].T  # (n_scans, n_voxels)

    events_file = EVENT_DIR / f"sub-{subject_id}" / "func" / \
        f"sub-{subject_id}_task-rsvp_run-{run_idx}_events.tsv"
    if not events_file.exists():
        raise FileNotFoundError(f"events not found: {events_file}")
    events = pd.read_csv(events_file, sep='\t')
    events['onset'] = events['onset'] - (3 * TR)  # 3 dropped volumes
    return func_data, events


# ============================================================================
# GLM functions — VERBATIM from canonical run_full_dataset_C010.py
# ============================================================================
def create_drift_regressors(n_scans, run_idx, n_runs):
    drift_cols = np.zeros((n_scans, n_runs * 2))
    drift_cols[:, run_idx - 1] = np.linspace(-0.5, 0.5, n_scans)
    drift_cols[:, n_runs + run_idx - 1] = 1.0
    return drift_cols


def convolve_hrf_with_events(events, n_scans, tr, hrf, hrf_deriv):
    frame_times = np.arange(n_scans) * tr
    X_hrf = np.zeros((n_scans, N_COLORS * 2))
    for color_idx in range(N_COLORS):
        color_name = f'color_{color_idx + 1}'
        color_events = events[events['trial_type'] == color_name]
        if len(color_events) == 0:
            continue
        for _, event in color_events.iterrows():
            onset = event['onset']
            hrf_signal = np.interp(frame_times - onset, np.arange(len(hrf)) * tr, hrf, left=0, right=0)
            X_hrf[:, color_idx] += hrf_signal
            deriv_signal = np.interp(frame_times - onset, np.arange(len(hrf_deriv)) * tr, hrf_deriv, left=0, right=0)
            X_hrf[:, N_COLORS + color_idx] += deriv_signal
    return X_hrf


def build_2nd_level_design_matrix(events, n_scans, tr, hrf, hrf_deriv, run_idx, n_runs):
    X_components = []
    X_hrf = convolve_hrf_with_events(events, n_scans, tr, hrf, hrf_deriv)
    X_components.append(X_hrf[:, :N_COLORS])   # HRF
    X_components.append(X_hrf[:, N_COLORS:])   # Derivative
    X_components.append(create_drift_regressors(n_scans, run_idx, n_runs))
    return np.hstack(X_components)


def apply_procrustes_alignment(amplitudes_raw):
    """EXACT Phase-2 code. Aligns runs 1..k to run-0 of the given block."""
    n_runs, n_colors, n_voxels = amplitudes_raw.shape
    reference = amplitudes_raw[0]
    aligned = np.zeros_like(amplitudes_raw)
    aligned[0] = reference
    disparities = np.zeros(n_runs)
    for run_idx in range(1, n_runs):
        target = amplitudes_raw[run_idx]
        R, scale = orthogonal_procrustes(target.T, reference.T)
        aligned_run = (target.T @ R).T
        disparity = np.sum((aligned_run - reference) ** 2) / n_voxels
        disparities[run_idx] = disparity
        aligned[run_idx] = aligned_run
    return aligned, disparities


# ============================================================================
# Main: per-ROI -> 8-run amplitudes -> condition split -> per-cond Procrustes
# ============================================================================
def estimate_roi_hrf(subject_id, roi_mask):
    """FIR 1st-level across ALL 8 runs -> single ROI HRF (+ gradient deriv)."""
    all_func, all_X = [], []
    for run_idx in range(1, N_RUNS + 1):
        func_data, events = load_bold_data(subject_id, run_idx, roi_mask)
        n_scans = func_data.shape[0]
        X_fir = np.zeros((n_scans, len(FIR_DELAYS) * N_COLORS))
        for color_idx in range(N_COLORS):
            color_events = events[events['trial_type'] == f'color_{color_idx + 1}']
            for delay_idx in range(len(FIR_DELAYS)):
                reg_idx = color_idx * len(FIR_DELAYS) + delay_idx
                regressor = np.zeros(n_scans)
                for _, event in color_events.iterrows():
                    onset_tr = int(np.round(event['onset'] / TR))
                    target_tr = onset_tr + delay_idx
                    if 0 <= target_tr < n_scans:
                        regressor[target_tr] = 1.0
                X_fir[:, reg_idx] = regressor
        all_func.append(func_data)
        all_X.append(X_fir)
    Y = np.vstack(all_func)
    X = np.vstack(all_X)
    betas_fir = np.linalg.lstsq(X, Y, rcond=None)[0]
    roi_hrf = betas_fir.mean(axis=1).reshape(N_COLORS, len(FIR_DELAYS)).mean(axis=0)
    roi_hrf_deriv = np.gradient(roi_hrf)
    return roi_hrf, roi_hrf_deriv


def estimate_run_amplitudes(subject_id, roi_mask, roi_hrf, roi_hrf_deriv):
    """Per-run 2nd-level amplitudes for all 8 runs -> (8, 8, V)."""
    amps = []
    for run_idx in range(1, N_RUNS + 1):
        func_data, events = load_bold_data(subject_id, run_idx, roi_mask)
        n_scans = func_data.shape[0]
        X = build_2nd_level_design_matrix(events, n_scans, TR, roi_hrf, roi_hrf_deriv, run_idx, N_RUNS)
        betas, _, _, _ = np.linalg.lstsq(X, func_data, rcond=None)
        amps.append(betas[:N_COLORS, :])
    return np.array(amps)  # (8, 8, V)


def run_subject_roi(subject_id, roi_name):
    print(f"\n{'='*72}\nexp2 C010  sub-{subject_id}  {roi_name}\n{'='*72}")
    roi_mask = build_exp2_roi_mask(subject_id, roi_name)

    print("Step 1: FIR HRF (across 8 runs)...")
    roi_hrf, roi_hrf_deriv = estimate_roi_hrf(subject_id, roi_mask)

    print("Step 2: per-run amplitudes (8 runs)...")
    amps8 = estimate_run_amplitudes(subject_id, roi_mask, roi_hrf, roi_hrf_deriv)
    print(f"  -> 8-run amplitudes {amps8.shape}")

    # reference dump of all 8 raw runs
    allraw_dir = OUTPUT_DIR / f"sub-{subject_id}" / "_all_runs" / roi_name
    allraw_dir.mkdir(parents=True, exist_ok=True)
    np.save(allraw_dir / "amplitudes_raw_8run.npy", amps8)

    print("Step 3: condition split + per-condition Procrustes...")
    cond_runs = CONDITION_RUNS_BY_SUBJECT[subject_id]
    for cond, runs1 in cond_runs.items():
        idx = [r - 1 for r in runs1]  # 0-indexed
        block_raw = amps8[idx]        # (4, 8, V)
        block_proc, disp = apply_procrustes_alignment(block_raw)
        out = OUTPUT_DIR / f"sub-{subject_id}" / cond / roi_name
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "amplitudes_raw.npy", block_raw)
        np.save(out / "amplitudes_procrustes.npy", block_proc)
        np.save(out / "procrustes_disparities.npy", disp)
        np.save(out / "roi_hrf.npy", roi_hrf)
        np.save(out / "roi_hrf_deriv.npy", roi_hrf_deriv)
        with open(out / "config.json", "w") as f:
            json.dump({
                "subject": subject_id, "roi": roi_name, "condition": cond,
                "runs_1indexed": runs1, "n_voxels": int(block_raw.shape[2]),
                "pipeline": "C010_exp2", "mask": "masknone_gmTrue n exp2-coverage",
                "procrustes": "per-condition, aligned to block run-0",
                "mean_disparity": float(disp.mean()),
            }, f, indent=2)
        print(f"  [{cond}] runs={runs1} shape={block_proc.shape} "
              f"mean_disp={disp.mean():.6f} -> {out}")
    print(f"DONE sub-{subject_id} {roi_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', required=True, help='e.g. 08')
    ap.add_argument('--roi', required=True, choices=['V1', 'V2', 'V3', 'V4'])
    ap.add_argument('--mask-variant', default='native', choices=['native', 'matched'])
    args = ap.parse_args()
    global MASK_VARIANT, OUTPUT_DIR
    MASK_VARIANT = args.mask_variant
    OUTPUT_DIR = BASE_DIR / "derivatives" / (
        "full_dataset_C010_exp2" if MASK_VARIANT == 'native'
        else "full_dataset_C010_exp2_matched")
    try:
        run_subject_roi(args.subject, args.roi)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
