#!/usr/bin/env python3
"""
A4: Crossnobis RDM — SRM-independent pairwise color distances.

Computes bias-free (cross-validated) Mahalanobis distances in native voxel space,
completely independent of SRM alignment. If the same HC-CVD group differences
emerge here, SRM is not creating artifacts.

Method (Walther et al. 2016):
  1. For each subject: estimate noise covariance via Ledoit-Wolf shrinkage
  2. Cross-validated Mahalanobis distance over all C(6,2)=15 run pairs
  3. Produces 8×8 unbiased distance matrix per subject
  4. Compare RDM similarity: HC-HC (21) vs HC-CVD (21) vs CVD-CVD (3)
  5. Convergent validity: correlate crossnobis "distance from HC mean" with SRM disparities

No BrainIAK required. ~5 min runtime.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase2_SRM_across_between/validation/compute_crossnobis_rdm.py  (development repository, commit 53c81c2)
# Role      : Cross-validated Mahalanobis RDM in native voxel space (S9). Reads residuals_2nd_level.npy (not distributed).
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


import numpy as np
import json
import time
import socket
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.covariance import LedoitWolf

# ============================================================================
# CONFIG
# ============================================================================

HC_SUBJECTS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
CVD_SUBJECTS = ['sub-08', 'sub-09', 'sub-10']
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS
ROIS = ['V1', 'V2', 'V3', 'V4']
ROI_DIR_MAP = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'V4'}
ROI_DISPLAY = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}
K_VALUES = {'V1': 4, 'V2': 4, 'V3': 3, 'V4': 3}

N_PERM = 10000
N_BOOT = 10000
SEED = 42

if socket.gethostname().startswith('node'):
    DATA_DIR = Path("/scratch/connectome/haba6030/colorBlind/derivatives/full_dataset_C010")
else:
    DATA_DIR = Path("/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/colorBlind_analysis"
                    "/analysis/phase1_preprocess_decoding/results/full_dataset_C010")

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "crossnobis_rdm"

# Path to SRM results for convergent validity
SRM_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "loo_consistent"

COLOR_LABELS = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_subject_data(subject, roi):
    """Load amplitudes_procrustes.npy → (6, 8, n_voxels)"""
    roi_dir = ROI_DIR_MAP.get(roi, roi)
    path = DATA_DIR / subject / roi_dir / "amplitudes_procrustes.npy"
    return np.load(path)


def compute_crossnobis_rdm(data):
    """
    Compute cross-validated Mahalanobis (crossnobis) distance matrix.

    Parameters
    ----------
    data : ndarray, shape (n_runs, n_conditions, n_voxels)
        e.g. (6, 8, n_voxels)

    Returns
    -------
    rdm : ndarray, shape (n_conditions, n_conditions)
        Unbiased crossnobis distance matrix (symmetric, zero diagonal).
    """
    n_runs, n_cond, n_vox = data.shape

    if n_vox < 20:
        print(f"    WARNING: n_voxels={n_vox} < 20, Ledoit-Wolf may be imprecise")

    # Compute residuals for noise covariance estimation
    # Residuals = each run's data minus the run-specific mean across conditions
    residuals = []
    for r in range(n_runs):
        run_mean = data[r].mean(axis=0, keepdims=True)  # (1, n_vox)
        residuals.append(data[r] - run_mean)  # (n_cond, n_vox)
    residuals = np.vstack(residuals)  # (n_runs * n_cond, n_vox)

    # Estimate noise covariance with Ledoit-Wolf shrinkage (handles p > n)
    lw = LedoitWolf()
    lw.fit(residuals)
    sigma_inv = np.linalg.inv(lw.covariance_)  # (n_vox, n_vox)

    # Cross-validated Mahalanobis distance over all C(6,2)=15 run pairs
    rdm = np.zeros((n_cond, n_cond))
    n_pairs = 0

    for r1, r2 in combinations(range(n_runs), 2):
        for i in range(n_cond):
            for j in range(i + 1, n_cond):
                # Cross-validated: use difference from run1 × difference from run2
                diff_r1 = data[r1, i] - data[r1, j]  # (n_vox,)
                diff_r2 = data[r2, i] - data[r2, j]  # (n_vox,)
                # Crossnobis = diff1 @ sigma_inv @ diff2 (unbiased estimator)
                d = diff_r1 @ sigma_inv @ diff_r2
                rdm[i, j] += d
                rdm[j, i] += d
        n_pairs += 1

    rdm /= n_pairs  # Average over run pairs
    return rdm


def rdm_to_vector(rdm):
    """Extract upper triangle of RDM as vector."""
    mask = np.triu(np.ones(rdm.shape, dtype=bool), k=1)
    return rdm[mask]


def load_srm_disparities():
    """Load SRM LOO-consistent disparities for convergent validity."""
    # Find latest results
    result_dirs = sorted(SRM_RESULTS_DIR.glob("*/loo_consistent_results.json"))
    if not result_dirs:
        print("  WARNING: No SRM results found for convergent validity")
        return None

    latest = result_dirs[-1]
    print(f"  Loading SRM results from: {latest.parent.name}")
    with open(latest) as f:
        srm_data = json.load(f)

    # Extract per-subject disparities for each ROI
    srm_disps = {}
    for roi_display, roi_result in srm_data['results'].items():
        disps = {}
        # HC LOO disparities
        for s, v in roi_result['hc_loo_disparities'].items():
            disps[s] = v
        # CVD mean disparities
        for s, iv in roi_result['individual_cvd'].items():
            disps[s] = iv['cvd_score']
        srm_disps[roi_display] = disps

    return srm_disps


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_roi(roi):
    """Run full crossnobis analysis for one ROI."""
    roi_display = ROI_DISPLAY.get(roi, roi)
    print(f"\n{'=' * 60}")
    print(f"Crossnobis RDM Analysis: {roi_display}")
    print(f"{'=' * 60}")

    # --- Compute crossnobis RDMs for all subjects ---
    rdms = {}
    rdm_vectors = {}

    for s in ALL_SUBJECTS:
        data = load_subject_data(s, roi)
        n_vox = data.shape[2]
        group = 'HC' if s in HC_SUBJECTS else 'CVD'
        print(f"  {s} ({group}): {data.shape}, n_voxels={n_vox}")
        rdm = compute_crossnobis_rdm(data)
        rdms[s] = rdm
        rdm_vectors[s] = rdm_to_vector(rdm)

    # --- RDM similarity analysis ---
    # Pairwise Spearman correlations between subjects' RDMs
    hc_hc_sims, hc_cvd_sims, cvd_cvd_sims = [], [], []
    pair_sims = {}

    for s1, s2 in combinations(ALL_SUBJECTS, 2):
        r, _ = spearmanr(rdm_vectors[s1], rdm_vectors[s2])
        if not np.isfinite(r):
            continue
        pair_sims[(s1, s2)] = r

        g1 = 'hc' if s1 in HC_SUBJECTS else 'cvd'
        g2 = 'hc' if s2 in HC_SUBJECTS else 'cvd'
        if g1 == 'hc' and g2 == 'hc':
            hc_hc_sims.append(r)
        elif g1 == 'cvd' and g2 == 'cvd':
            cvd_cvd_sims.append(r)
        else:
            hc_cvd_sims.append(r)

    print(f"\n  RDM Similarities (Spearman):")
    print(f"    HC-HC:   mean={np.mean(hc_hc_sims):.3f} (n={len(hc_hc_sims)})")
    print(f"    HC-CVD:  mean={np.mean(hc_cvd_sims):.3f} (n={len(hc_cvd_sims)})")
    if cvd_cvd_sims:
        print(f"    CVD-CVD: mean={np.mean(cvd_cvd_sims):.3f} (n={len(cvd_cvd_sims)})")

    # --- Mann-Whitney U on HC-HC vs HC-CVD similarities ---
    if len(hc_hc_sims) >= 2 and len(hc_cvd_sims) >= 2:
        u_stat, u_p = mannwhitneyu(hc_hc_sims, hc_cvd_sims, alternative='greater')
        print(f"\n  Mann-Whitney U (HC-HC > HC-CVD): U={u_stat:.1f}, p={u_p:.4f}")
    else:
        u_stat, u_p = float('nan'), float('nan')

    # --- Permutation test on similarity difference ---
    print(f"  Permutation test ({N_PERM} iterations)...")
    rng = np.random.RandomState(SEED)
    observed_diff = np.mean(hc_hc_sims) - np.mean(hc_cvd_sims)

    # Pool all pairwise similarities with group labels
    all_pairs = []
    all_labels = []  # 'within' or 'between'
    for (s1, s2), r in pair_sims.items():
        g1 = 'hc' if s1 in HC_SUBJECTS else 'cvd'
        g2 = 'hc' if s2 in HC_SUBJECTS else 'cvd'
        if g1 == 'hc' and g2 == 'hc':
            all_pairs.append(r)
            all_labels.append('within')
        elif (g1 == 'hc' and g2 == 'cvd') or (g1 == 'cvd' and g2 == 'hc'):
            all_pairs.append(r)
            all_labels.append('between')
    all_pairs = np.array(all_pairs)
    all_labels = np.array(all_labels)
    n_within = np.sum(all_labels == 'within')

    null_diffs = np.zeros(N_PERM)
    for pi in range(N_PERM):
        perm_labels = all_labels[rng.permutation(len(all_labels))]
        null_within = all_pairs[perm_labels == 'within'].mean()
        null_between = all_pairs[perm_labels == 'between'].mean()
        null_diffs[pi] = null_within - null_between

    perm_p = float(np.mean(null_diffs >= observed_diff))
    print(f"    Observed HC-HC − HC-CVD: {observed_diff:.4f}")
    print(f"    Null mean: {null_diffs.mean():.4f}")
    print(f"    p = {perm_p:.4f}")

    # --- Bootstrap CIs ---
    print(f"  Bootstrap CIs ({N_BOOT} iterations)...")
    boot_hchc, boot_hccvd, boot_diff = [], [], []
    for _ in range(N_BOOT):
        b_hc = np.array(hc_hc_sims)[rng.choice(len(hc_hc_sims), len(hc_hc_sims), replace=True)]
        b_cvd = np.array(hc_cvd_sims)[rng.choice(len(hc_cvd_sims), len(hc_cvd_sims), replace=True)]
        boot_hchc.append(b_hc.mean())
        boot_hccvd.append(b_cvd.mean())
        boot_diff.append(b_hc.mean() - b_cvd.mean())

    boot_hchc = np.array(boot_hchc)
    boot_hccvd = np.array(boot_hccvd)
    boot_diff = np.array(boot_diff)

    # --- Distance from HC mean RDM ---
    # For convergent validity: compute each subject's distance from the HC mean RDM
    hc_rdm_vectors = [rdm_vectors[s] for s in HC_SUBJECTS]
    hc_mean_rdm_vec = np.mean(hc_rdm_vectors, axis=0)

    subject_distances = {}
    for s in ALL_SUBJECTS:
        # 1 - Spearman r with HC mean RDM
        r, _ = spearmanr(rdm_vectors[s], hc_mean_rdm_vec)
        subject_distances[s] = float(1.0 - r) if np.isfinite(r) else float('nan')

    print(f"\n  Distance from HC mean RDM (1 - Spearman r):")
    for s in HC_SUBJECTS:
        print(f"    {s}: {subject_distances[s]:.4f}")
    for s in CVD_SUBJECTS:
        print(f"    {s}: {subject_distances[s]:.4f}  (CVD)")

    return {
        'roi': roi_display,
        'rdm_similarities': {
            'hc_hc': {'mean': float(np.mean(hc_hc_sims)), 'values': hc_hc_sims,
                       'ci': [float(np.percentile(boot_hchc, 2.5)), float(np.percentile(boot_hchc, 97.5))]},
            'hc_cvd': {'mean': float(np.mean(hc_cvd_sims)), 'values': hc_cvd_sims,
                        'ci': [float(np.percentile(boot_hccvd, 2.5)), float(np.percentile(boot_hccvd, 97.5))]},
            'cvd_cvd': {'mean': float(np.mean(cvd_cvd_sims)) if cvd_cvd_sims else float('nan'),
                         'values': cvd_cvd_sims},
            'diff': {'observed': float(observed_diff),
                      'ci': [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))]},
        },
        'mann_whitney': {'U': float(u_stat), 'p': float(u_p)},
        'permutation': {'observed_diff': float(observed_diff),
                         'null_mean': float(null_diffs.mean()),
                         'p': perm_p, 'n_perm': N_PERM},
        'distance_from_hc_mean': subject_distances,
    }


# ============================================================================
# CONVERGENT VALIDITY
# ============================================================================

def compute_convergent_validity(results, srm_disps):
    """Correlate crossnobis distances with SRM disparities."""
    print(f"\n{'=' * 60}")
    print("Convergent Validity: Crossnobis vs SRM")
    print(f"{'=' * 60}")

    convergent = {}
    all_crossnobis, all_srm = [], []

    for roi_display in ['V1', 'V2', 'V3', 'hV4']:
        if roi_display not in results or roi_display not in srm_disps:
            continue

        xnobis_dists = results[roi_display]['distance_from_hc_mean']
        srm_d = srm_disps[roi_display]

        # Match subjects
        subjects = [s for s in ALL_SUBJECTS if s in xnobis_dists and s in srm_d]
        x_vals = [xnobis_dists[s] for s in subjects]
        s_vals = [srm_d[s] for s in subjects]

        # Filter NaN
        valid = [(x, s) for x, s in zip(x_vals, s_vals) if np.isfinite(x) and np.isfinite(s)]
        if len(valid) < 4:
            convergent[roi_display] = {'r': float('nan'), 'p': float('nan'), 'n': len(valid)}
            continue

        x_arr = np.array([v[0] for v in valid])
        s_arr = np.array([v[1] for v in valid])
        r, p = spearmanr(x_arr, s_arr)

        convergent[roi_display] = {'r': float(r), 'p': float(p), 'n': len(valid)}
        print(f"  {roi_display}: Spearman r={r:.3f}, p={p:.4f} (n={len(valid)})")

        all_crossnobis.extend(x_arr.tolist())
        all_srm.extend(s_arr.tolist())

    # Pooled across ROIs
    if len(all_crossnobis) >= 4:
        r_pool, p_pool = spearmanr(all_crossnobis, all_srm)
        convergent['pooled'] = {'r': float(r_pool), 'p': float(p_pool), 'n': len(all_crossnobis)}
        print(f"  Pooled: Spearman r={r_pool:.3f}, p={p_pool:.4f} (n={len(all_crossnobis)})")

    return convergent


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("A4: Crossnobis RDM Analysis (SRM-independent)")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)

    start = time.time()
    all_results = {}

    for roi in ROIS:
        result = analyze_roi(roi)
        all_results[ROI_DISPLAY.get(roi, roi)] = result

    # Convergent validity with SRM
    srm_disps = load_srm_disparities()
    convergent = {}
    if srm_disps:
        convergent = compute_convergent_validity(all_results, srm_disps)

    # --- Summary table ---
    print(f"\n\n{'=' * 60}")
    print("SUMMARY TABLE")
    print(f"{'=' * 60}")
    print(f"{'ROI':<5} | {'HC-HC [95% CI]':<24} | {'HC-CVD [95% CI]':<24} | {'Diff [95% CI]':<24} | {'U':>5} | {'p_U':>7} | {'p_perm':>7} | {'r_conv':>7}")
    print("-" * 120)

    for roi_d in ['V1', 'V2', 'V3', 'hV4']:
        if roi_d not in all_results:
            continue
        r = all_results[roi_d]
        sim = r['rdm_similarities']
        mw = r['mann_whitney']
        perm = r['permutation']
        r_conv = convergent.get(roi_d, {}).get('r', float('nan'))
        print(f"{roi_d:<5} | "
              f"{sim['hc_hc']['mean']:.3f} [{sim['hc_hc']['ci'][0]:.3f}, {sim['hc_hc']['ci'][1]:.3f}] | "
              f"{sim['hc_cvd']['mean']:.3f} [{sim['hc_cvd']['ci'][0]:.3f}, {sim['hc_cvd']['ci'][1]:.3f}] | "
              f"{sim['diff']['observed']:.3f} [{sim['diff']['ci'][0]:.3f}, {sim['diff']['ci'][1]:.3f}] | "
              f"{mw['U']:>5.0f} | {mw['p']:>7.4f} | {perm['p']:>7.4f} | {r_conv:>7.3f}")

    # --- Save ---
    elapsed = time.time() - start
    output = {
        'config': {
            'method': 'crossnobis_rdm',
            'description': 'Cross-validated Mahalanobis distances in native voxel space (SRM-independent)',
            'n_perm': N_PERM,
            'n_boot': N_BOOT,
            'seed': SEED,
            'elapsed_sec': elapsed,
        },
        'results': all_results,
        'convergent_validity': convergent,
    }

    out_file = OUTPUT_DIR / "crossnobis_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
