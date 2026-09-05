#!/usr/bin/env python3
"""
A5: PCA→CCA Replication — Replicate SRM group comparison with a different alignment.

Uses PCA for dimensionality reduction and CCA for alignment (completely independent
of BrainIAK/SRM). If the same HC-CVD disparity pattern emerges, the group difference
is alignment-algorithm-agnostic.

Method:
  1. For all C(10,2) = 45 subject pairs:
     - PCA each to k dims → CCA alignment → Procrustes disparity in CCA space
     - Also: PCA-only Procrustes disparity (no CCA, more conservative)
  2. Categorize: HC-HC (21), HC-CVD (21), CVD-CVD (3)
  3. Compare between-group vs within-group disparity

No BrainIAK required. ~10 min runtime.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase2_SRM_across_between/validation/compute_pca_cca_replication.py  (development repository, commit 53c81c2)
# Role      : PCA and PCA-CCA pairwise alignment replication (S9).
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
import warnings
from pathlib import Path
from itertools import combinations
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA

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

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "pca_cca_replication"

# Path to SRM results for convergent validity
SRM_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "loo_consistent"


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def load_subject_data(subject, roi):
    """Load amplitudes_procrustes.npy → (6, 8, n_voxels)"""
    roi_dir = ROI_DIR_MAP.get(roi, roi)
    path = DATA_DIR / subject / roi_dir / "amplitudes_procrustes.npy"
    return np.load(path)


def compute_procrustes_disparity(X, Y):
    """Procrustes disparity between two (8, k) pattern matrices."""
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    norm_X = np.linalg.norm(X_c, 'fro')
    norm_Y = np.linalg.norm(Y_c, 'fro')
    if norm_X == 0 or norm_Y == 0:
        return float('nan')
    X_n = X_c / norm_X
    Y_n = Y_c / norm_Y
    R, _ = orthogonal_procrustes(X_n, Y_n)
    return float(np.linalg.norm(X_n @ R - Y_n, 'fro'))


def pca_reduce(betas, k):
    """PCA-reduce (8, n_voxels) → (8, k)."""
    pca = PCA(n_components=k)
    return pca.fit_transform(betas)  # (8, k)


def cca_align(X, Y, k):
    """
    CCA-align two (8, k) matrices. Returns aligned versions.

    With only 8 observations and k=3-4 components, canonical correlations
    are inflated, but we use Procrustes disparity in CCA space (not the
    correlations themselves).
    """
    n_components = min(k, X.shape[0] - 1)  # max n_obs - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # CCA convergence warnings
        cca = CCA(n_components=n_components, max_iter=1000)
        X_c, Y_c = cca.fit_transform(X, Y)
    return X_c, Y_c


def compute_pairwise_disparities(betas_dict, k, method='pca_cca'):
    """
    Compute pairwise Procrustes disparities for all C(n,2) subject pairs.

    Parameters
    ----------
    betas_dict : dict
        {subject: (8, n_voxels) array}
    k : int
        Number of PCA components
    method : str
        'pca_only' or 'pca_cca'

    Returns
    -------
    pair_disps : dict
        {(s1, s2): float}
    """
    subjects = list(betas_dict.keys())
    pair_disps = {}

    # PCA-reduce all subjects
    pca_reduced = {}
    for s in subjects:
        pca_reduced[s] = pca_reduce(betas_dict[s], k)

    for s1, s2 in combinations(subjects, 2):
        X = pca_reduced[s1]  # (8, k)
        Y = pca_reduced[s2]  # (8, k)

        if method == 'pca_cca':
            X_a, Y_a = cca_align(X, Y, k)
            d = compute_procrustes_disparity(X_a, Y_a)
        else:  # pca_only
            d = compute_procrustes_disparity(X, Y)

        pair_disps[(s1, s2)] = d

    return pair_disps


def categorize_pairs(pair_disps):
    """Split pairwise disparities into HC-HC, HC-CVD, CVD-CVD."""
    hc_hc, hc_cvd, cvd_cvd = [], [], []

    for (s1, s2), d in pair_disps.items():
        if not np.isfinite(d):
            continue
        g1 = 'hc' if s1 in HC_SUBJECTS else 'cvd'
        g2 = 'hc' if s2 in HC_SUBJECTS else 'cvd'
        if g1 == 'hc' and g2 == 'hc':
            hc_hc.append(d)
        elif g1 == 'cvd' and g2 == 'cvd':
            cvd_cvd.append(d)
        else:
            hc_cvd.append(d)

    return hc_hc, hc_cvd, cvd_cvd


def hedges_g(group1, group2):
    """Hedges' g effect size (group2 - group1)."""
    g1, g2 = np.array(group1), np.array(group2)
    n1, n2 = len(g1), len(g2)
    s_pooled = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    if s_pooled == 0:
        return 0.0
    d = (g2.mean() - g1.mean()) / s_pooled
    return d * (1 - 3 / (4*(n1+n2-2) - 1))


def load_srm_disparities():
    """Load SRM LOO-consistent disparities for convergent validity."""
    result_files = sorted(SRM_RESULTS_DIR.glob("*/loo_consistent_results.json"))
    if not result_files:
        print("  WARNING: No SRM results found for convergent validity")
        return None

    latest = result_files[-1]
    print(f"  Loading SRM results from: {latest.parent.name}")
    with open(latest) as f:
        srm_data = json.load(f)

    srm_disps = {}
    for roi_display, roi_result in srm_data['results'].items():
        disps = {}
        for s, v in roi_result['hc_loo_disparities'].items():
            disps[s] = v
        for s, iv in roi_result['individual_cvd'].items():
            disps[s] = iv['cvd_score']
        srm_disps[roi_display] = disps

    return srm_disps


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_roi(roi, k):
    """Run full PCA→CCA analysis for one ROI."""
    roi_display = ROI_DISPLAY.get(roi, roi)
    print(f"\n{'=' * 60}")
    print(f"PCA→CCA Replication: {roi_display}, k={k}")
    print(f"{'=' * 60}")

    # Load mean betas for all subjects
    betas_dict = {}
    for s in ALL_SUBJECTS:
        data = load_subject_data(s, roi)  # (6, 8, n_voxels)
        betas = data.mean(axis=0)  # (8, n_voxels)
        group = 'HC' if s in HC_SUBJECTS else 'CVD'
        print(f"  {s} ({group}): {data.shape}")
        betas_dict[s] = betas

    rng = np.random.RandomState(SEED)
    results = {}

    for method in ['pca_only', 'pca_cca']:
        print(f"\n  --- Method: {method} ---")

        # Compute pairwise disparities
        pair_disps = compute_pairwise_disparities(betas_dict, k, method=method)
        hc_hc, hc_cvd, cvd_cvd = categorize_pairs(pair_disps)

        print(f"    HC-HC:   mean={np.mean(hc_hc):.4f} (n={len(hc_hc)})")
        print(f"    HC-CVD:  mean={np.mean(hc_cvd):.4f} (n={len(hc_cvd)})")
        if cvd_cvd:
            print(f"    CVD-CVD: mean={np.mean(cvd_cvd):.4f} (n={len(cvd_cvd)})")

        # Test: HC-CVD > HC-HC (CVD subjects are more distant from each other's HC)
        observed_diff = np.mean(hc_cvd) - np.mean(hc_hc)
        g = hedges_g(hc_hc, hc_cvd)
        print(f"    Diff (HC-CVD − HC-HC): {observed_diff:.4f}")
        print(f"    Hedges' g: {g:.3f}")

        # Mann-Whitney U
        if len(hc_hc) >= 2 and len(hc_cvd) >= 2:
            u_stat, u_p = mannwhitneyu(hc_cvd, hc_hc, alternative='greater')
            print(f"    Mann-Whitney U (HC-CVD > HC-HC): U={u_stat:.1f}, p={u_p:.4f}")
        else:
            u_stat, u_p = float('nan'), float('nan')

        # Permutation test: shuffle group labels
        print(f"    Permutation test ({N_PERM} iterations)...")
        all_disps = np.array(hc_hc + hc_cvd)
        all_labels = np.array(['within'] * len(hc_hc) + ['between'] * len(hc_cvd))

        null_diffs = np.zeros(N_PERM)
        for pi in range(N_PERM):
            perm_labels = all_labels[rng.permutation(len(all_labels))]
            null_within = all_disps[perm_labels == 'within'].mean()
            null_between = all_disps[perm_labels == 'between'].mean()
            null_diffs[pi] = null_between - null_within

        perm_p = float(np.mean(null_diffs >= observed_diff))
        print(f"    p_perm = {perm_p:.4f}")

        # Bootstrap CIs
        boot_diff, boot_g = [], []
        for _ in range(N_BOOT):
            b_hc = np.array(hc_hc)[rng.choice(len(hc_hc), len(hc_hc), replace=True)]
            b_cvd = np.array(hc_cvd)[rng.choice(len(hc_cvd), len(hc_cvd), replace=True)]
            boot_diff.append(b_cvd.mean() - b_hc.mean())
            boot_g.append(hedges_g(b_hc, b_cvd))
        boot_diff = np.array(boot_diff)
        boot_g_arr = np.array(boot_g)

        # Per-subject mean distance to HC group
        subject_mean_dist = {}
        for s in ALL_SUBJECTS:
            dists = []
            for (s1, s2), d in pair_disps.items():
                if s == s1 and s2 in HC_SUBJECTS:
                    dists.append(d)
                elif s == s2 and s1 in HC_SUBJECTS:
                    dists.append(d)
            if dists:
                subject_mean_dist[s] = float(np.mean(dists))

        results[method] = {
            'hc_hc': {'mean': float(np.mean(hc_hc)), 'std': float(np.std(hc_hc, ddof=1)),
                       'values': hc_hc},
            'hc_cvd': {'mean': float(np.mean(hc_cvd)), 'std': float(np.std(hc_cvd, ddof=1)),
                        'values': hc_cvd},
            'cvd_cvd': {'mean': float(np.mean(cvd_cvd)) if cvd_cvd else float('nan'),
                         'values': cvd_cvd},
            'diff': {
                'observed': float(observed_diff),
                'ci': [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))],
            },
            'hedges_g': {
                'value': float(g),
                'ci': [float(np.percentile(boot_g_arr, 2.5)), float(np.percentile(boot_g_arr, 97.5))],
            },
            'mann_whitney': {'U': float(u_stat), 'p': float(u_p)},
            'permutation': {'observed_diff': float(observed_diff),
                             'null_mean': float(null_diffs.mean()),
                             'p': perm_p, 'n_perm': N_PERM},
            'subject_mean_dist_to_hc': subject_mean_dist,
        }

    return {'roi': roi_display, 'k': k, **results}


# ============================================================================
# CONVERGENT VALIDITY
# ============================================================================

def compute_convergent_validity(results, srm_disps):
    """Correlate PCA-CCA per-subject distances with SRM disparities."""
    print(f"\n{'=' * 60}")
    print("Convergent Validity: PCA-CCA vs SRM")
    print(f"{'=' * 60}")

    convergent = {}
    all_pca_cca, all_srm = [], []

    for method in ['pca_only', 'pca_cca']:
        convergent[method] = {}
        m_pca, m_srm = [], []

        for roi_d in ['V1', 'V2', 'V3', 'hV4']:
            if roi_d not in results or roi_d not in srm_disps:
                continue
            pca_dists = results[roi_d][method]['subject_mean_dist_to_hc']
            srm_d = srm_disps[roi_d]

            subjects = [s for s in ALL_SUBJECTS if s in pca_dists and s in srm_d]
            x_vals = [pca_dists[s] for s in subjects]
            s_vals = [srm_d[s] for s in subjects]

            valid = [(x, s) for x, s in zip(x_vals, s_vals) if np.isfinite(x) and np.isfinite(s)]
            if len(valid) < 4:
                convergent[method][roi_d] = {'r': float('nan'), 'p': float('nan'), 'n': len(valid)}
                continue

            x_arr = np.array([v[0] for v in valid])
            s_arr = np.array([v[1] for v in valid])
            r, p = spearmanr(x_arr, s_arr)

            convergent[method][roi_d] = {'r': float(r), 'p': float(p), 'n': len(valid)}
            print(f"  [{method}] {roi_d}: Spearman r={r:.3f}, p={p:.4f} (n={len(valid)})")

            m_pca.extend(x_arr.tolist())
            m_srm.extend(s_arr.tolist())

        if len(m_pca) >= 4:
            r_pool, p_pool = spearmanr(m_pca, m_srm)
            convergent[method]['pooled'] = {'r': float(r_pool), 'p': float(p_pool), 'n': len(m_pca)}
            print(f"  [{method}] Pooled: Spearman r={r_pool:.3f}, p={p_pool:.4f} (n={len(m_pca)})")

        if method == 'pca_cca':
            all_pca_cca = m_pca
            all_srm = m_srm

    return convergent


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("A5: PCA→CCA Replication (SRM-independent)")
    print(f"Data: {DATA_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"K values: {K_VALUES}")
    print("=" * 60)

    start = time.time()
    all_results = {}

    for roi in ROIS:
        k = K_VALUES[roi]
        result = analyze_roi(roi, k)
        all_results[ROI_DISPLAY.get(roi, roi)] = result

    # Convergent validity
    srm_disps = load_srm_disparities()
    convergent = {}
    if srm_disps:
        convergent = compute_convergent_validity(all_results, srm_disps)

    # --- Summary table ---
    print(f"\n\n{'=' * 60}")
    print("SUMMARY TABLE")
    print(f"{'=' * 60}")

    for method in ['pca_only', 'pca_cca']:
        print(f"\n  --- {method} ---")
        print(f"  {'ROI':<5} | {'HC-HC':<10} | {'HC-CVD':<10} | {'Diff [95% CI]':<24} | {'g [95% CI]':<22} | {'p_U':>7} | {'p_perm':>7} | {'r_conv':>7}")
        print("  " + "-" * 110)

        for roi_d in ['V1', 'V2', 'V3', 'hV4']:
            if roi_d not in all_results:
                continue
            r = all_results[roi_d][method]
            r_conv = convergent.get(method, {}).get(roi_d, {}).get('r', float('nan'))
            print(f"  {roi_d:<5} | "
                  f"{r['hc_hc']['mean']:.4f}     | "
                  f"{r['hc_cvd']['mean']:.4f}     | "
                  f"{r['diff']['observed']:.4f} [{r['diff']['ci'][0]:.4f}, {r['diff']['ci'][1]:.4f}] | "
                  f"{r['hedges_g']['value']:.2f} [{r['hedges_g']['ci'][0]:.2f}, {r['hedges_g']['ci'][1]:.2f}] | "
                  f"{r['mann_whitney']['p']:>7.4f} | "
                  f"{r['permutation']['p']:>7.4f} | "
                  f"{r_conv:>7.3f}")

    # --- Save ---
    elapsed = time.time() - start
    output = {
        'config': {
            'method': 'pca_cca_replication',
            'description': 'PCA→CCA pairwise alignment replication (SRM-independent)',
            'k_values': K_VALUES,
            'n_perm': N_PERM,
            'n_boot': N_BOOT,
            'seed': SEED,
            'elapsed_sec': elapsed,
        },
        'results': all_results,
        'convergent_validity': convergent,
    }

    out_file = OUTPUT_DIR / "pca_cca_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
