#!/usr/bin/env python3
"""
LOO-Consistent SRM Analysis with Crawford & Howell Individual Testing
+ LOSO-Based Color-Dependency Test

Fixes all identified biases:
  1. HC-only SRM training (RT-2: no circularity)
  2. LOO for HC disparity (no group-mean leakage)
  3. Same LOO references for CVD (no reference asymmetry)
  4. Crawford & Howell (1998) individual CVD testing (RT-3: n=3 heterogeneity)
  5. LOSO color-dependency: HC tested in space they did NOT train (RT-4)

Main Framework (LOO-consistent):
  Train single SRM on 7 HC → transform HC via w_, project CVD via SVD
  For each LOO fold i (leaving out HC sub-i):
    ref_i = mean(other 6 HC aligned patterns)
    hc_disp[i]    = disparity(HC sub-i, ref_i)
    cvd_disp[j,i] = disparity(CVD sub-j, ref_i)

LOSO Color-Dependency Framework:
  For each LOSO fold i (leaving out HC sub-i):
    Train SRM on 6 HC (excluding sub-i)
    Project held-out HC sub-i via SVD (same method as CVD)
    Project 3 CVD subjects via SVD
    ref = mean(6 trained HC in SRM space)
    hc_held_out_disp[i] = disparity(sub-i_projected, ref)
    cvd_disp[j][i]      = disparity(CVD_j_projected, ref)

  Both HC and CVD are tested in spaces they did NOT train →
  eliminates the structural floor confound in HC color-label permutation.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase2_SRM_across_between/rerun_loo_consistent.py  (development repository, commit 53c81c2)
# Role      : Canonical SRM + disparity pipeline (common space and symmetric LOSO). Output: loo_consistent_results.json. Requires BrainIAK (mpirun -np 1).
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
import time
import socket
import sys
from pathlib import Path
from datetime import datetime
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import squareform, pdist
from scipy.stats import t as t_dist, spearmanr
from itertools import combinations

try:
    from brainiak.funcalign.srm import SRM
except ImportError:
    print("ERROR: brainiak not available")
    sys.exit(1)

# ============================================================================
# CONFIG
# ============================================================================

HC_SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
CVD_SUBJECTS = ['sub-08', 'sub-09', 'sub-10']
ROIS = ['V1', 'V2', 'V3', 'V4']
ROI_DIR_MAP = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'V4'}
ROI_DISPLAY = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}
K_VALUES = {'V1': 4, 'V2': 4, 'V3': 3, 'V4': 3}

N_PERM_GROUP = 10000
N_PERM_COLOR = 1000
SEED = 42

if socket.gethostname().startswith('node'):
    DATA_DIR = Path("/scratch/connectome/haba6030/colorBlind/derivatives/full_dataset_C010")
else:
    DATA_DIR = Path("/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/colorBlind_analysis"
                    "/analysis/phase1_preprocess_decoding/results/full_dataset_C010")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "loo_consistent"


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_subject_data(subject, roi):
    roi_dir = ROI_DIR_MAP.get(roi, roi)
    path = DATA_DIR / subject / roi_dir / "amplitudes_procrustes.npy"
    return np.load(path)  # (6, 8, n_voxels)


def project_new_subject(srm_model, new_data):
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def compute_procrustes_disparity(X, Y):
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    X_n = X_c / np.linalg.norm(X_c, 'fro')
    Y_n = Y_c / np.linalg.norm(Y_c, 'fro')
    R, _ = orthogonal_procrustes(X_n, Y_n)
    return float(np.linalg.norm(X_n @ R - Y_n, 'fro'))


def hedges_g(group1, group2):
    n1, n2 = len(group1), len(group2)
    s_pooled = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    if s_pooled == 0:
        return 0.0
    d = (np.mean(group2) - np.mean(group1)) / s_pooled
    return d * (1 - 3 / (4*(n1+n2-2) - 1))


def crawford_howell_test(patient_score, control_scores):
    """
    Crawford & Howell (1998) modified t-test for single case.

    Tests whether patient_score is significantly above the control distribution.

    Returns: t-statistic, p-value (one-tailed), df
    """
    n = len(control_scores)
    control_mean = np.mean(control_scores)
    control_sd = np.std(control_scores, ddof=1)

    if control_sd == 0:
        return float('inf') if patient_score > control_mean else 0.0, 0.0, n - 1

    t_stat = (patient_score - control_mean) / (control_sd * np.sqrt((n + 1) / n))
    df = n - 1
    p_value = 1 - t_dist.cdf(t_stat, df)  # one-tailed (patient > control)

    return float(t_stat), float(p_value), int(df)


def compute_loo_consistent_disparities(hc_aligned_list, cvd_aligned_list):
    """
    Compute LOO-consistent disparities where HC and CVD use the SAME references.

    For each fold i (leaving out HC sub-i):
      ref_i = mean(other 6 HC)
      hc_disp[i] = disparity(HC sub-i, ref_i)
      cvd_disp[j][i] = disparity(CVD sub-j, ref_i)

    Returns
    -------
    hc_loo_disps : np.ndarray, shape (n_hc,)
    cvd_loo_disps : np.ndarray, shape (n_cvd, n_hc) — [j, i] = CVD j at fold i
    cvd_scores : np.ndarray, shape (n_cvd,) — mean across folds
    """
    hc_arr = np.array(hc_aligned_list)  # (n_hc, 8, k)
    n_hc = len(hc_aligned_list)
    n_cvd = len(cvd_aligned_list)

    hc_loo_disps = np.zeros(n_hc)
    cvd_loo_disps = np.zeros((n_cvd, n_hc))

    for i in range(n_hc):
        # LOO reference: mean of other 6 HC
        others = np.delete(hc_arr, i, axis=0)
        ref_i = others.mean(axis=0)  # (8, k)

        # HC sub-i disparity
        hc_loo_disps[i] = compute_procrustes_disparity(hc_aligned_list[i], ref_i)

        # CVD disparities to same reference
        for j in range(n_cvd):
            cvd_loo_disps[j, i] = compute_procrustes_disparity(cvd_aligned_list[j], ref_i)

    cvd_scores = cvd_loo_disps.mean(axis=1)  # (n_cvd,) — average across 7 folds

    return hc_loo_disps, cvd_loo_disps, cvd_scores


def compute_rdm_correlations(hc_aligned_list, cvd_aligned_list):
    """Compute within-group and between-group RDM correlations."""
    mask = np.triu(np.ones((8, 8), dtype=bool), k=1)
    all_patterns = hc_aligned_list + cvd_aligned_list
    all_labels = ['hc'] * len(hc_aligned_list) + ['cvd'] * len(cvd_aligned_list)
    rdms = [squareform(pdist(p, metric='correlation')) for p in all_patterns]

    hc_hc, hc_cvd, cvd_cvd = [], [], []
    for i in range(len(rdms)):
        for j in range(i + 1, len(rdms)):
            r, _ = spearmanr(rdms[i][mask], rdms[j][mask])
            if not np.isfinite(r):
                continue
            li, lj = all_labels[i], all_labels[j]
            if li == 'hc' and lj == 'hc':
                hc_hc.append(r)
            elif li == 'cvd' and lj == 'cvd':
                cvd_cvd.append(r)
            else:
                hc_cvd.append(r)

    return {
        'hc_hc': {'mean': float(np.mean(hc_hc)), 'n': len(hc_hc)},
        'hc_cvd': {'mean': float(np.mean(hc_cvd)), 'n': len(hc_cvd)},
        'cvd_cvd': {'mean': float(np.mean(cvd_cvd)) if cvd_cvd else float('nan'), 'n': len(cvd_cvd)},
    }


def run_loso_srm_analysis(hc_data, cvd_data, k):
    """
    LOSO-based SRM: each HC is tested in a space they did NOT train.

    For each fold i (leaving out HC sub-i):
      1. Train SRM on 6 HC (excluding sub-i)
      2. Transform 6 training HC using srm.w_
      3. Project held-out HC sub-i via SVD (identical to CVD treatment)
      4. Project 3 CVD subjects via SVD
      5. Reference = mean of 6 trained HC (in SRM space)
      6. hc_held_out_disp[i] = disparity(held-out projected, reference)
      7. cvd_disp[j][i] = disparity(CVD projected, reference)

    Returns
    -------
    hc_held_out_disps : (n_hc,)
    cvd_fold_disps : (n_cvd, n_hc)  — [j, i] = CVD j at fold i
    cvd_scores : (n_cvd,) — mean across folds
    """
    n_hc = len(hc_data)
    n_cvd = len(cvd_data)

    hc_betas = [d.mean(axis=0) for d in hc_data]      # list of (8, n_voxels)
    cvd_betas = [d.mean(axis=0) for d in cvd_data]

    hc_held_out_disps = np.zeros(n_hc)
    cvd_fold_disps = np.zeros((n_cvd, n_hc))

    for i in range(n_hc):
        # Train SRM on 6 HC (excluding sub-i)
        train_indices = [j for j in range(n_hc) if j != i]
        train_betas = [hc_betas[j].T for j in train_indices]  # list of (n_voxels, 8)
        srm = SRM(n_iter=10, features=k)
        srm.fit(train_betas)

        # Transform 6 training HC → aligned in SRM space
        train_aligned = [(srm.w_[t].T @ train_betas[t]).T for t in range(len(train_indices))]
        ref = np.mean(train_aligned, axis=0)  # (8, k)

        # Project held-out HC sub-i via SVD (same method as CVD)
        W_ho = project_new_subject(srm, hc_betas[i].T)
        ho_aligned = hc_betas[i] @ W_ho  # (8, k)
        hc_held_out_disps[i] = compute_procrustes_disparity(ho_aligned, ref)

        # Project CVD subjects via SVD
        for j in range(n_cvd):
            W_cvd = project_new_subject(srm, cvd_betas[j].T)
            cvd_aligned = cvd_betas[j] @ W_cvd  # (8, k)
            cvd_fold_disps[j, i] = compute_procrustes_disparity(cvd_aligned, ref)

    cvd_scores = cvd_fold_disps.mean(axis=1)
    return hc_held_out_disps, cvd_fold_disps, cvd_scores


def run_loso_srm_from_betas(hc_betas, cvd_betas, k):
    """
    Same as run_loso_srm_analysis but takes pre-computed beta arrays.
    Used by the color-label permutation to avoid redundant .mean(axis=0) calls.

    Parameters
    ----------
    hc_betas : list of (8, n_voxels) arrays
    cvd_betas : list of (8, n_voxels) arrays
    """
    n_hc = len(hc_betas)
    n_cvd = len(cvd_betas)

    hc_held_out_disps = np.zeros(n_hc)
    cvd_fold_disps = np.zeros((n_cvd, n_hc))

    for i in range(n_hc):
        train_indices = [j for j in range(n_hc) if j != i]
        train_betas_t = [hc_betas[j].T for j in train_indices]
        srm = SRM(n_iter=10, features=k)
        srm.fit(train_betas_t)

        train_aligned = [(srm.w_[t].T @ train_betas_t[t]).T for t in range(len(train_indices))]
        ref = np.mean(train_aligned, axis=0)

        W_ho = project_new_subject(srm, hc_betas[i].T)
        ho_aligned = hc_betas[i] @ W_ho
        hc_held_out_disps[i] = compute_procrustes_disparity(ho_aligned, ref)

        for j in range(n_cvd):
            W_cvd = project_new_subject(srm, cvd_betas[j].T)
            cvd_aligned = cvd_betas[j] @ W_cvd
            cvd_fold_disps[j, i] = compute_procrustes_disparity(cvd_aligned, ref)

    cvd_scores = cvd_fold_disps.mean(axis=1)
    return hc_held_out_disps, cvd_fold_disps, cvd_scores


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def run_loo_consistent_analysis(roi, k):
    roi_display = ROI_DISPLAY.get(roi, roi)
    print(f"\n{'=' * 70}")
    print(f"LOO-Consistent Analysis: {roi_display}, k={k}")
    print(f"{'=' * 70}")

    # --- Load data ---
    hc_data = [load_subject_data(s, roi) for s in HC_SUBJECTS]
    cvd_data = [load_subject_data(s, roi) for s in CVD_SUBJECTS]
    for i, s in enumerate(HC_SUBJECTS):
        print(f"  {s}: {hc_data[i].shape}")
    for i, s in enumerate(CVD_SUBJECTS):
        print(f"  {s}: {cvd_data[i].shape}")

    # --- Train HC-only SRM ---
    print(f"\n  Training SRM on 7 HC (k={k})...")
    hc_betas = [d.mean(axis=0).T for d in hc_data]  # list of (n_voxels, 8)
    srm = SRM(n_iter=10, features=k)
    srm.fit(hc_betas)

    # Transform HC
    hc_aligned = [srm.w_[i].T @ hc_betas[i] for i in range(len(HC_SUBJECTS))]
    hc_aligned = [a.T for a in hc_aligned]  # list of (8, k)

    # Project CVD
    cvd_aligned = []
    for i, s in enumerate(CVD_SUBJECTS):
        beta = cvd_data[i].mean(axis=0)  # (8, n_voxels)
        W_cvd = project_new_subject(srm, beta.T)
        cvd_aligned.append((beta @ W_cvd))  # (8, k)

    # --- LOO-consistent disparities ---
    print("\n  Computing LOO-consistent disparities...")
    hc_loo_disps, cvd_loo_disps, cvd_scores = compute_loo_consistent_disparities(
        hc_aligned, cvd_aligned)

    print(f"\n  HC LOO disparities (same 6-subj references):")
    for i, s in enumerate(HC_SUBJECTS):
        print(f"    {s}: {hc_loo_disps[i]:.4f}")
    print(f"    Mean: {hc_loo_disps.mean():.4f} ± {hc_loo_disps.std(ddof=1):.4f}")

    print(f"\n  CVD disparities (averaged across 7 LOO references):")
    for j, s in enumerate(CVD_SUBJECTS):
        fold_vals = ', '.join(f'{v:.3f}' for v in cvd_loo_disps[j])
        print(f"    {s}: {cvd_scores[j]:.4f}  (folds: [{fold_vals}])")

    separation = cvd_scores.mean() - hc_loo_disps.mean()
    g = hedges_g(hc_loo_disps, cvd_scores)
    print(f"\n  Separation: {separation:.4f}")
    print(f"  Hedges' g: {g:.3f}")

    # --- Crawford & Howell individual tests ---
    print(f"\n  Crawford & Howell individual CVD tests:")
    individual_results = {}
    for j, s in enumerate(CVD_SUBJECTS):
        t_stat, p_val, df = crawford_howell_test(cvd_scores[j], hc_loo_disps)
        sig = '*' if p_val < 0.05 else ''
        print(f"    {s}: score={cvd_scores[j]:.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")
        individual_results[s] = {
            'cvd_score': float(cvd_scores[j]),
            'fold_disparities': cvd_loo_disps[j].tolist(),
            't_stat': t_stat,
            'p_value': p_val,
            'df': df,
            'significant': p_val < 0.05,
            'pct_above_hc': float((cvd_scores[j] - hc_loo_disps.mean()) / hc_loo_disps.mean() * 100),
        }

    # --- Group-level permutation test (LOO-consistent) ---
    print(f"\n  Group permutation test ({N_PERM_GROUP} iterations, LOO-consistent)...")
    rng = np.random.RandomState(SEED)
    observed_diff = cvd_scores.mean() - hc_loo_disps.mean()

    all_aligned = hc_aligned + cvd_aligned  # list of 10 (8, k) arrays
    n_hc = len(HC_SUBJECTS)

    null_diffs = np.zeros(N_PERM_GROUP)
    for p_i in range(N_PERM_GROUP):
        perm = rng.permutation(len(all_aligned))
        pseudo_hc = [all_aligned[idx] for idx in perm[:n_hc]]
        pseudo_cvd = [all_aligned[idx] for idx in perm[n_hc:]]

        ph_loo, _, pc_scores = compute_loo_consistent_disparities(pseudo_hc, pseudo_cvd)
        null_diffs[p_i] = pc_scores.mean() - ph_loo.mean()

    p_group = float(np.mean(null_diffs >= observed_diff))
    print(f"  Observed diff: {observed_diff:.4f}")
    print(f"  Null mean: {null_diffs.mean():.4f}")
    print(f"  p = {p_group:.4f}")

    # --- Bootstrap CIs ---
    print(f"  Bootstrap CIs (10000 iterations)...")
    n_boot = 10000
    boot_hc, boot_cvd, boot_sep, boot_g = [], [], [], []
    for _ in range(n_boot):
        hc_b = hc_loo_disps[rng.choice(len(hc_loo_disps), len(hc_loo_disps), replace=True)]
        cvd_b = cvd_scores[rng.choice(len(cvd_scores), len(cvd_scores), replace=True)]
        boot_hc.append(hc_b.mean())
        boot_cvd.append(cvd_b.mean())
        boot_sep.append(cvd_b.mean() - hc_b.mean())
        boot_g.append(hedges_g(hc_b, cvd_b))

    boot_hc, boot_cvd, boot_sep, boot_g = map(np.array, [boot_hc, boot_cvd, boot_sep, boot_g])

    # --- RDM correlations ---
    rdm_corrs = compute_rdm_correlations(hc_aligned, cvd_aligned)

    # --- Color-label permutation (1D + 1D-ext, LOO-consistent) ---
    print(f"\n  Color-label permutation ({N_PERM_COLOR} iterations)...")
    rng_color = np.random.default_rng(SEED)

    null_color = {
        'disparity_diff': [], 'hc_rdm': [], 'cvd_rdm': [],
        'hc_loo_mean': [], 'cvd_score_mean': [],
        'cvd_pairwise_mean': [],
    }

    for c_i in range(N_PERM_COLOR):
        if (c_i + 1) % 100 == 0:
            print(f"    Color perm {c_i + 1}/{N_PERM_COLOR}")

        # Shuffle color labels in original voxel data
        hc_shuffled = [d.mean(axis=0)[rng_color.permutation(8)] for d in hc_data]
        cvd_shuffled = [d.mean(axis=0)[rng_color.permutation(8)] for d in cvd_data]

        # Retrain HC-only SRM on shuffled data
        sh_betas = [s.T for s in hc_shuffled]
        sh_srm = SRM(n_iter=10, features=k)
        sh_srm.fit(sh_betas)

        sh_hc_al = [(sh_srm.w_[i].T @ sh_betas[i]).T for i in range(n_hc)]
        sh_cvd_al = []
        for cs in cvd_shuffled:
            W = project_new_subject(sh_srm, cs.T)
            sh_cvd_al.append(cs @ W)

        # LOO-consistent disparities
        sh_hc_loo, _, sh_cvd_sc = compute_loo_consistent_disparities(sh_hc_al, sh_cvd_al)
        null_color['disparity_diff'].append(sh_cvd_sc.mean() - sh_hc_loo.mean())
        null_color['hc_loo_mean'].append(sh_hc_loo.mean())
        null_color['cvd_score_mean'].append(sh_cvd_sc.mean())

        # CVD pairwise
        cvd_pw = []
        for a in range(len(sh_cvd_al)):
            for b in range(a + 1, len(sh_cvd_al)):
                cvd_pw.append(compute_procrustes_disparity(sh_cvd_al[a], sh_cvd_al[b]))
        null_color['cvd_pairwise_mean'].append(np.mean(cvd_pw))

        # RDM correlations
        sh_rdm = compute_rdm_correlations(sh_hc_al, sh_cvd_al)
        null_color['hc_rdm'].append(sh_rdm['hc_hc']['mean'])
        null_color['cvd_rdm'].append(sh_rdm['cvd_cvd']['mean'])

    for key in null_color:
        null_color[key] = np.array(null_color[key])

    # Observed color-label metrics
    obs_cvd_pw = []
    for a in range(len(cvd_aligned)):
        for b in range(a + 1, len(cvd_aligned)):
            obs_cvd_pw.append(compute_procrustes_disparity(cvd_aligned[a], cvd_aligned[b]))
    obs_cvd_pw_mean = np.mean(obs_cvd_pw)

    p_color = {
        'disparity_diff': float(np.mean(null_color['disparity_diff'] >= observed_diff)),
        'hc_rdm': float(np.mean(null_color['hc_rdm'] >= rdm_corrs['hc_hc']['mean'])),
        'cvd_rdm': float(np.mean(null_color['cvd_rdm'] >= rdm_corrs['cvd_cvd']['mean'])),
        'hc_loo_disp': float(np.mean(null_color['hc_loo_mean'] <= hc_loo_disps.mean())),
        'cvd_score_disp': float(np.mean(null_color['cvd_score_mean'] <= cvd_scores.mean())),
        'cvd_pairwise_disp': float(np.mean(null_color['cvd_pairwise_mean'] <= obs_cvd_pw_mean)),
    }

    print(f"\n  Color-label permutation results (single-SRM):")
    print(f"    Disp diff:      obs={observed_diff:.4f}  null={null_color['disparity_diff'].mean():.4f}  p={p_color['disparity_diff']:.4f}")
    print(f"    HC RDM:         obs={rdm_corrs['hc_hc']['mean']:.3f}  null={null_color['hc_rdm'].mean():.3f}  p={p_color['hc_rdm']:.4f}")
    print(f"    CVD RDM:        obs={rdm_corrs['cvd_cvd']['mean']:.3f}  null={null_color['cvd_rdm'].mean():.3f}  p={p_color['cvd_rdm']:.4f}")
    print(f"    HC LOO disp:    obs={hc_loo_disps.mean():.4f}  null={null_color['hc_loo_mean'].mean():.4f}  p={p_color['hc_loo_disp']:.4f}")
    print(f"    CVD score disp: obs={cvd_scores.mean():.4f}  null={null_color['cvd_score_mean'].mean():.4f}  p={p_color['cvd_score_disp']:.4f}")
    print(f"    CVD pairwise:   obs={obs_cvd_pw_mean:.4f}  null={null_color['cvd_pairwise_mean'].mean():.4f}  p={p_color['cvd_pairwise_disp']:.4f}")

    # === LOSO-BASED COLOR-DEPENDENCY TEST (RT-4 fix) ===
    # Both HC and CVD tested in spaces they did NOT train
    print(f"\n  LOSO-based SRM analysis (7 folds)...")
    loso_hc_disps, loso_cvd_fold_disps, loso_cvd_scores = run_loso_srm_analysis(
        hc_data, cvd_data, k)

    print(f"\n  LOSO HC held-out disparities (projected via SVD, NOT trained):")
    for i, s in enumerate(HC_SUBJECTS):
        print(f"    {s}: {loso_hc_disps[i]:.4f}")
    print(f"    Mean: {loso_hc_disps.mean():.4f} +/- {loso_hc_disps.std(ddof=1):.4f}")

    print(f"\n  LOSO CVD disparities (projected via SVD, same as HC):")
    for j, s in enumerate(CVD_SUBJECTS):
        fold_vals = ', '.join(f'{v:.3f}' for v in loso_cvd_fold_disps[j])
        print(f"    {s}: {loso_cvd_scores[j]:.4f}  (folds: [{fold_vals}])")

    loso_separation = loso_cvd_scores.mean() - loso_hc_disps.mean()
    loso_g = hedges_g(loso_hc_disps, loso_cvd_scores)
    print(f"    Separation: {loso_separation:.4f}, Hedges' g: {loso_g:.3f}")

    # LOSO Crawford & Howell
    print(f"\n  LOSO Crawford & Howell individual CVD tests:")
    loso_individual = {}
    for j, s in enumerate(CVD_SUBJECTS):
        t_stat, p_val, df = crawford_howell_test(loso_cvd_scores[j], loso_hc_disps)
        sig = '*' if p_val < 0.05 else ''
        print(f"    {s}: score={loso_cvd_scores[j]:.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")
        loso_individual[s] = {
            'cvd_score': float(loso_cvd_scores[j]),
            'fold_disparities': loso_cvd_fold_disps[j].tolist(),
            't_stat': float(t_stat), 'p_value': float(p_val), 'df': int(df),
            'significant': p_val < 0.05,
        }

    # LOSO group permutation (shuffle 10 precomputed LOSO scores)
    print(f"\n  LOSO group permutation ({N_PERM_GROUP} iterations)...")
    rng_loso = np.random.RandomState(SEED + 1)
    loso_observed_diff = loso_cvd_scores.mean() - loso_hc_disps.mean()
    all_loso_scores = np.concatenate([loso_hc_disps, loso_cvd_scores])
    loso_null_diffs = np.zeros(N_PERM_GROUP)
    for p_i in range(N_PERM_GROUP):
        perm = rng_loso.permutation(len(all_loso_scores))
        pseudo_hc = all_loso_scores[perm[:n_hc]]
        pseudo_cvd = all_loso_scores[perm[n_hc:]]
        loso_null_diffs[p_i] = pseudo_cvd.mean() - pseudo_hc.mean()
    loso_p_group = float(np.mean(loso_null_diffs >= loso_observed_diff))
    print(f"    Observed diff: {loso_observed_diff:.4f}, p = {loso_p_group:.4f}")

    # LOSO color-label permutation
    print(f"\n  LOSO color-label permutation ({N_PERM_COLOR} iterations, 7 folds each)...")
    rng_loso_color = np.random.default_rng(SEED + 2)

    null_loso = {
        'hc_held_out_mean': np.zeros(N_PERM_COLOR),
        'cvd_score_mean': np.zeros(N_PERM_COLOR),
        'disparity_diff': np.zeros(N_PERM_COLOR),
    }

    for c_i in range(N_PERM_COLOR):
        if (c_i + 1) % 100 == 0:
            print(f"    LOSO color perm {c_i + 1}/{N_PERM_COLOR}")

        # Shuffle color labels independently per subject
        hc_shuf_betas = [d.mean(axis=0)[rng_loso_color.permutation(8)] for d in hc_data]
        cvd_shuf_betas = [d.mean(axis=0)[rng_loso_color.permutation(8)] for d in cvd_data]

        # Full LOSO with shuffled data
        sh_hc, _, sh_cvd = run_loso_srm_from_betas(hc_shuf_betas, cvd_shuf_betas, k)
        null_loso['hc_held_out_mean'][c_i] = sh_hc.mean()
        null_loso['cvd_score_mean'][c_i] = sh_cvd.mean()
        null_loso['disparity_diff'][c_i] = sh_cvd.mean() - sh_hc.mean()

    # LOSO color-dependency p-values
    # Lower observed disparity = better fit to color structure
    # P(null <= observed) tests whether shuffled data produces LOWER disparity than true labels
    p_loso_color = {
        'hc_held_out_disp': float(np.mean(null_loso['hc_held_out_mean'] <= loso_hc_disps.mean())),
        'cvd_score_disp': float(np.mean(null_loso['cvd_score_mean'] <= loso_cvd_scores.mean())),
        'disparity_diff': float(np.mean(null_loso['disparity_diff'] >= loso_observed_diff)),
    }

    print(f"\n  LOSO color-label permutation results:")
    print(f"    HC held-out:  obs={loso_hc_disps.mean():.4f}  null={null_loso['hc_held_out_mean'].mean():.4f}  p={p_loso_color['hc_held_out_disp']:.4f}")
    print(f"    CVD score:    obs={loso_cvd_scores.mean():.4f}  null={null_loso['cvd_score_mean'].mean():.4f}  p={p_loso_color['cvd_score_disp']:.4f}")
    print(f"    Disp diff:    obs={loso_observed_diff:.4f}  null={null_loso['disparity_diff'].mean():.4f}  p={p_loso_color['disparity_diff']:.4f}")

    # --- Compile results ---
    return {
        'roi': roi_display,
        'k': k,
        'method': 'hc_only_srm_loo_consistent',
        'hc_loo_disparities': {s: float(hc_loo_disps[i]) for i, s in enumerate(HC_SUBJECTS)},
        'summary': {
            'hc_mean': float(hc_loo_disps.mean()),
            'hc_std': float(hc_loo_disps.std(ddof=1)),
            'hc_ci': [float(np.percentile(boot_hc, 2.5)), float(np.percentile(boot_hc, 97.5))],
            'cvd_mean': float(cvd_scores.mean()),
            'cvd_std': float(cvd_scores.std(ddof=1)),
            'cvd_ci': [float(np.percentile(boot_cvd, 2.5)), float(np.percentile(boot_cvd, 97.5))],
            'separation': float(separation),
            'separation_ci': [float(np.percentile(boot_sep, 2.5)), float(np.percentile(boot_sep, 97.5))],
            'hedges_g': float(g),
            'hedges_g_ci': [float(np.percentile(boot_g, 2.5)), float(np.percentile(boot_g, 97.5))],
            'group_perm_p': p_group,
            'n_perm_group': N_PERM_GROUP,
        },
        'individual_cvd': individual_results,
        'rdm_correlations': rdm_corrs,
        'color_label_permutation': {
            'n_permutations': N_PERM_COLOR,
            'p_values': p_color,
            'null_means': {kk: float(v.mean()) for kk, v in null_color.items()},
            'observed': {
                'disparity_diff': float(observed_diff),
                'hc_loo_mean': float(hc_loo_disps.mean()),
                'cvd_score_mean': float(cvd_scores.mean()),
                'cvd_pairwise_mean': float(obs_cvd_pw_mean),
                'hc_rdm': rdm_corrs['hc_hc']['mean'],
                'cvd_rdm': rdm_corrs['cvd_cvd']['mean'],
            },
        },
        'loso_analysis': {
            'description': 'LOSO-based SRM: HC tested in space they did NOT train (same as CVD)',
            'hc_held_out_disparities': {s: float(loso_hc_disps[i]) for i, s in enumerate(HC_SUBJECTS)},
            'cvd_fold_disparities': {
                s: loso_cvd_fold_disps[j].tolist() for j, s in enumerate(CVD_SUBJECTS)
            },
            'summary': {
                'hc_mean': float(loso_hc_disps.mean()),
                'hc_std': float(loso_hc_disps.std(ddof=1)),
                'cvd_mean': float(loso_cvd_scores.mean()),
                'cvd_std': float(loso_cvd_scores.std(ddof=1)),
                'separation': float(loso_separation),
                'hedges_g': float(loso_g),
                'group_perm_p': loso_p_group,
            },
            'individual_cvd': loso_individual,
            'color_label_permutation': {
                'n_permutations': N_PERM_COLOR,
                'p_values': p_loso_color,
                'null_means': {kk: float(v.mean()) for kk, v in null_loso.items()},
                'observed': {
                    'hc_held_out_mean': float(loso_hc_disps.mean()),
                    'cvd_score_mean': float(loso_cvd_scores.mean()),
                    'disparity_diff': float(loso_observed_diff),
                },
            },
        },
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = OUTPUT_DIR / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LOO-Consistent SRM Analysis + Crawford & Howell + LOSO Color-Dependency")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {out_dir}")
    print(f"k values: {K_VALUES}")
    print(f"Group perm: {N_PERM_GROUP}, Color perm: {N_PERM_COLOR}")
    print("=" * 70)

    all_results = {}
    start = time.time()

    for roi in ROIS:
        k = K_VALUES[roi]
        result = run_loo_consistent_analysis(roi, k)
        all_results[ROI_DISPLAY.get(roi, roi)] = result

    # Save
    out_file = out_dir / "loo_consistent_results.json"
    output = {
        'config': {
            'method': 'hc_only_srm_loo_consistent',
            'description': 'HC-only SRM + LOO-consistent + Crawford & Howell + LOSO color-dependency test',
            'k_values': K_VALUES,
            'n_perm_group': N_PERM_GROUP,
            'n_perm_color': N_PERM_COLOR,
            'seed': SEED,
            'timestamp': timestamp,
            'elapsed_sec': time.time() - start,
        },
        'results': all_results,
    }
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # === SUMMARY TABLES ===
    print(f"\n\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")

    # Table 1: Group comparison
    print("\n--- Table 1: Group Disparity (LOO-consistent) ---")
    print(f"{'ROI':<5} | {'HC LOO [95% CI]':<26} | {'CVD [95% CI]':<26} | {'Sep [95% CI]':<26} | {'g [95% CI]':<22} | {'p':>6}")
    print("-" * 120)
    for roi_d in ['V1', 'V2', 'V3', 'hV4']:
        if roi_d not in all_results:
            continue
        s = all_results[roi_d]['summary']
        print(f"{roi_d:<5} | "
              f"{s['hc_mean']:.3f} [{s['hc_ci'][0]:.3f}, {s['hc_ci'][1]:.3f}] | "
              f"{s['cvd_mean']:.3f} [{s['cvd_ci'][0]:.3f}, {s['cvd_ci'][1]:.3f}] | "
              f"{s['separation']:.3f} [{s['separation_ci'][0]:.3f}, {s['separation_ci'][1]:.3f}] | "
              f"{s['hedges_g']:.2f} [{s['hedges_g_ci'][0]:.2f}, {s['hedges_g_ci'][1]:.2f}] | "
              f"{s['group_perm_p']:>6.4f}")

    # Table 2: Individual CVD (Crawford & Howell)
    print("\n--- Table 2: Individual CVD Tests (Crawford & Howell 1998) ---")
    print(f"{'Subject':<8} | {'V1 score (t, p)':<22} | {'V2 score (t, p)':<22} | {'V3 score (t, p)':<22} | {'hV4 score (t, p)':<22}")
    print("-" * 100)
    for s in CVD_SUBJECTS:
        parts = []
        for roi_d in ['V1', 'V2', 'V3', 'hV4']:
            if roi_d in all_results and s in all_results[roi_d]['individual_cvd']:
                iv = all_results[roi_d]['individual_cvd'][s]
                sig = '*' if iv['significant'] else ''
                parts.append(f"{iv['cvd_score']:.3f} ({iv['t_stat']:.1f}, {iv['p_value']:.3f}){sig}")
            else:
                parts.append("N/A")
        print(f"{s:<8} | " + " | ".join(f"{p:<22}" for p in parts))

    # Table 3: Color-label permutation (single-SRM based)
    print("\n--- Table 3: Color-Label Permutation (single-SRM, 1D + 1D-ext) ---")
    print(f"{'ROI':<5} | {'Disp diff p':<12} | {'HC RDM p':<10} | {'CVD RDM p':<10} | {'HC LOO p':<10} | {'CVD score p':<12} | {'CVD pair p':<12}")
    print("-" * 85)
    for roi_d in ['V1', 'V2', 'V3', 'hV4']:
        if roi_d not in all_results:
            continue
        cp = all_results[roi_d]['color_label_permutation']['p_values']
        print(f"{roi_d:<5} | "
              f"{cp['disparity_diff']:<12.4f} | "
              f"{cp['hc_rdm']:<10.4f} | "
              f"{cp['cvd_rdm']:<10.4f} | "
              f"{cp['hc_loo_disp']:<10.4f} | "
              f"{cp['cvd_score_disp']:<12.4f} | "
              f"{cp['cvd_pairwise_disp']:<12.4f}")

    # Table 4: LOSO color-dependency (RT-4 fix)
    print("\n--- Table 4: LOSO Color-Dependency (HC + CVD both projected, NOT trained) ---")
    print(f"{'ROI':<5} | {'HC held-out':<14} | {'CVD score':<14} | {'Sep':<8} | {'g':<8} | {'p_group':<8} | {'p_hc_color':<11} | {'p_cvd_color':<11} | {'p_diff_color':<12}")
    print("-" * 115)
    for roi_d in ['V1', 'V2', 'V3', 'hV4']:
        if roi_d not in all_results or 'loso_analysis' not in all_results[roi_d]:
            continue
        ls = all_results[roi_d]['loso_analysis']['summary']
        lp = all_results[roi_d]['loso_analysis']['color_label_permutation']['p_values']
        print(f"{roi_d:<5} | "
              f"{ls['hc_mean']:<14.4f} | "
              f"{ls['cvd_mean']:<14.4f} | "
              f"{ls['separation']:<8.4f} | "
              f"{ls['hedges_g']:<8.2f} | "
              f"{ls['group_perm_p']:<8.4f} | "
              f"{lp['hc_held_out_disp']:<11.4f} | "
              f"{lp['cvd_score_disp']:<11.4f} | "
              f"{lp['disparity_diff']:<12.4f}")

    # Table 5: LOSO individual CVD (Crawford & Howell)
    print("\n--- Table 5: LOSO Individual CVD Tests (Crawford & Howell 1998) ---")
    print(f"{'Subject':<8} | {'V1 score (t, p)':<22} | {'V2 score (t, p)':<22} | {'V3 score (t, p)':<22} | {'hV4 score (t, p)':<22}")
    print("-" * 100)
    for s in CVD_SUBJECTS:
        parts = []
        for roi_d in ['V1', 'V2', 'V3', 'hV4']:
            if roi_d in all_results and 'loso_analysis' in all_results[roi_d]:
                loso = all_results[roi_d]['loso_analysis']
                if s in loso['individual_cvd']:
                    iv = loso['individual_cvd'][s]
                    sig = '*' if iv['significant'] else ''
                    parts.append(f"{iv['cvd_score']:.3f} ({iv['t_stat']:.1f}, {iv['p_value']:.3f}){sig}")
                else:
                    parts.append("N/A")
            else:
                parts.append("N/A")
        print(f"{s:<8} | " + " | ".join(f"{p:<22}" for p in parts))

    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
