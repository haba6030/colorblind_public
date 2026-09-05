#!/usr/bin/env python3
"""
A3: Variance Explained — How much of each subject's data does SRM capture?

Quantifies reconstruction quality: VE = 1 - ||X - W @ S||^2 / ||X||^2
If HC variance explained > CVD, this independently confirms SRM fits HC better.

Two frameworks:
  Framework A (single-SRM):
    - HC: use trained W from SRM
    - CVD: use SVD-projected W (same as main analysis)
    → Baseline comparison, but confounded by training-vs-projection asymmetry

  Framework B (LOSO):
    - Each HC left out, projected via SVD (same treatment as CVD)
    - Unbiased: both HC and CVD use SVD projection
    → Fair comparison, though VE will be lower overall

Requires BrainIAK. Run with: mpirun -np 1 python compute_variance_explained.py
~20 min runtime.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase2_SRM_across_between/validation/compute_variance_explained.py  (development repository, commit 53c81c2)
# Role      : SRM variance explained under the symmetric LOSO projection (S9, tab:variance_explained).
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
import sys
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr
from scipy.stats import t as t_dist

try:
    from brainiak.funcalign.srm import SRM
except ImportError:
    print("ERROR: brainiak not available. Run with: mpirun -np 1 python compute_variance_explained.py")
    sys.exit(1)

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

OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "variance_explained"

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


def project_new_subject(srm_model, new_data):
    """SVD-based orthogonal projection for held-out/CVD subjects."""
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def compute_variance_explained(X, W, S):
    """
    Compute variance explained by the SRM reconstruction.

    VE = 1 - ||X - W @ S||^2 / ||X||^2

    Parameters
    ----------
    X : ndarray, shape (n_voxels, 8)
        Original voxel data
    W : ndarray, shape (n_voxels, k)
        Weight matrix (trained or SVD-projected)
    S : ndarray, shape (k, 8)
        Shared response

    Returns
    -------
    ve : float
        Variance explained (0 to 1, can be negative if reconstruction is worse than zero)
    """
    X_hat = W @ S
    ss_res = np.sum((X - X_hat) ** 2)
    ss_tot = np.sum(X ** 2)
    if ss_tot == 0:
        return float('nan')
    return float(1.0 - ss_res / ss_tot)


def hedges_g(group1, group2):
    """Hedges' g effect size (group2 - group1)."""
    g1, g2 = np.array(group1), np.array(group2)
    n1, n2 = len(g1), len(g2)
    s_pooled = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    if s_pooled == 0:
        return 0.0
    d = (g2.mean() - g1.mean()) / s_pooled
    return d * (1 - 3 / (4*(n1+n2-2) - 1))


def crawford_howell_test(patient_score, control_scores):
    """Crawford & Howell (1998) modified t-test for single case."""
    n = len(control_scores)
    control_mean = np.mean(control_scores)
    control_sd = np.std(control_scores, ddof=1)
    if control_sd == 0:
        return float('inf') if patient_score < control_mean else 0.0, 0.0, n - 1
    # For VE, lower is worse → one-tailed: patient < control
    t_stat = (patient_score - control_mean) / (control_sd * np.sqrt((n + 1) / n))
    df = n - 1
    p_value = t_dist.cdf(t_stat, df)  # one-tailed (patient < control)
    return float(t_stat), float(p_value), int(df)


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
    """Run variance explained analysis for one ROI."""
    roi_display = ROI_DISPLAY.get(roi, roi)
    print(f"\n{'=' * 60}")
    print(f"Variance Explained: {roi_display}, k={k}")
    print(f"{'=' * 60}")

    # Load data
    hc_data = [load_subject_data(s, roi) for s in HC_SUBJECTS]
    cvd_data = [load_subject_data(s, roi) for s in CVD_SUBJECTS]
    for i, s in enumerate(HC_SUBJECTS):
        print(f"  {s}: {hc_data[i].shape}")
    for i, s in enumerate(CVD_SUBJECTS):
        print(f"  {s}: {cvd_data[i].shape}")

    # Mean betas
    hc_betas = [d.mean(axis=0) for d in hc_data]      # list of (8, n_voxels)
    cvd_betas = [d.mean(axis=0) for d in cvd_data]

    rng = np.random.RandomState(SEED)

    # ================================================================
    # Framework A: Single SRM (trained on all 7 HC)
    # ================================================================
    print(f"\n  --- Framework A: Single SRM ---")
    hc_betas_t = [b.T for b in hc_betas]  # list of (n_voxels, 8)
    srm = SRM(n_iter=10, features=k)
    srm.fit(hc_betas_t)

    # HC: use trained W
    hc_ve_a = []
    for i, s in enumerate(HC_SUBJECTS):
        ve = compute_variance_explained(hc_betas_t[i], srm.w_[i], srm.s_)
        hc_ve_a.append(ve)
        print(f"    {s}: VE={ve:.4f}")

    # CVD: use SVD-projected W
    cvd_ve_a = []
    for i, s in enumerate(CVD_SUBJECTS):
        W_cvd = project_new_subject(srm, cvd_betas[i].T)
        ve = compute_variance_explained(cvd_betas[i].T, W_cvd, srm.s_)
        cvd_ve_a.append(ve)
        print(f"    {s} (CVD): VE={ve:.4f}")

    hc_ve_a = np.array(hc_ve_a)
    cvd_ve_a = np.array(cvd_ve_a)
    print(f"    HC mean: {hc_ve_a.mean():.4f} ± {hc_ve_a.std(ddof=1):.4f}")
    print(f"    CVD mean: {cvd_ve_a.mean():.4f} ± {cvd_ve_a.std(ddof=1):.4f}")
    print(f"    Diff: {hc_ve_a.mean() - cvd_ve_a.mean():.4f}")

    # ================================================================
    # Framework B: LOSO (unbiased — both HC and CVD use SVD)
    # ================================================================
    print(f"\n  --- Framework B: LOSO ---")
    n_hc = len(HC_SUBJECTS)

    hc_ve_b = np.zeros(n_hc)
    # CVD VE per fold: (n_cvd, n_hc)
    cvd_ve_b_folds = np.zeros((len(CVD_SUBJECTS), n_hc))

    for fold_i in range(n_hc):
        # Train SRM on 6 HC (excluding fold_i)
        train_idx = [j for j in range(n_hc) if j != fold_i]
        train_betas = [hc_betas_t[j] for j in train_idx]
        fold_srm = SRM(n_iter=10, features=k)
        fold_srm.fit(train_betas)

        # Project held-out HC via SVD
        W_ho = project_new_subject(fold_srm, hc_betas[fold_i].T)
        hc_ve_b[fold_i] = compute_variance_explained(hc_betas[fold_i].T, W_ho, fold_srm.s_)

        # Project CVD via SVD (same method)
        for j in range(len(CVD_SUBJECTS)):
            W_cvd = project_new_subject(fold_srm, cvd_betas[j].T)
            cvd_ve_b_folds[j, fold_i] = compute_variance_explained(cvd_betas[j].T, W_cvd, fold_srm.s_)

    cvd_ve_b = cvd_ve_b_folds.mean(axis=1)  # Average across folds

    print(f"    LOSO HC held-out VE:")
    for i, s in enumerate(HC_SUBJECTS):
        print(f"      {s}: {hc_ve_b[i]:.4f}")
    print(f"      Mean: {hc_ve_b.mean():.4f} ± {hc_ve_b.std(ddof=1):.4f}")

    print(f"    LOSO CVD VE (mean across folds):")
    for j, s in enumerate(CVD_SUBJECTS):
        fold_vals = ', '.join(f'{v:.3f}' for v in cvd_ve_b_folds[j])
        print(f"      {s}: {cvd_ve_b[j]:.4f}  (folds: [{fold_vals}])")
    print(f"      Mean: {cvd_ve_b.mean():.4f} ± {cvd_ve_b.std(ddof=1):.4f}")

    sep_b = hc_ve_b.mean() - cvd_ve_b.mean()
    g_b = hedges_g(cvd_ve_b, hc_ve_b)  # HC higher = positive g
    print(f"    Diff (HC − CVD): {sep_b:.4f}")
    print(f"    Hedges' g: {g_b:.3f}")

    # --- Statistical tests (on LOSO framework B) ---

    # Crawford & Howell individual tests
    print(f"\n  Crawford & Howell (VE: CVD < HC):")
    individual_b = {}
    for j, s in enumerate(CVD_SUBJECTS):
        t_stat, p_val, df = crawford_howell_test(cvd_ve_b[j], hc_ve_b)
        sig = '*' if p_val < 0.05 else ''
        print(f"    {s}: VE={cvd_ve_b[j]:.4f}, t={t_stat:.3f}, p={p_val:.4f} {sig}")
        individual_b[s] = {
            've': float(cvd_ve_b[j]),
            'fold_ve': cvd_ve_b_folds[j].tolist(),
            't_stat': t_stat, 'p_value': p_val, 'df': df,
            'significant': p_val < 0.05,
        }

    # Mann-Whitney U (HC VE > CVD VE)
    if len(hc_ve_b) >= 2 and len(cvd_ve_b) >= 2:
        u_stat, u_p = mannwhitneyu(hc_ve_b, cvd_ve_b, alternative='greater')
        print(f"\n  Mann-Whitney U (HC VE > CVD VE): U={u_stat:.1f}, p={u_p:.4f}")
    else:
        u_stat, u_p = float('nan'), float('nan')

    # Permutation test
    print(f"  Permutation test ({N_PERM} iterations)...")
    observed_diff_b = hc_ve_b.mean() - cvd_ve_b.mean()
    all_ve = np.concatenate([hc_ve_b, cvd_ve_b])
    null_diffs = np.zeros(N_PERM)
    for pi in range(N_PERM):
        perm = rng.permutation(len(all_ve))
        pseudo_hc = all_ve[perm[:n_hc]]
        pseudo_cvd = all_ve[perm[n_hc:]]
        null_diffs[pi] = pseudo_hc.mean() - pseudo_cvd.mean()
    perm_p = float(np.mean(null_diffs >= observed_diff_b))
    print(f"    Observed diff: {observed_diff_b:.4f}, p = {perm_p:.4f}")

    # Bootstrap CIs
    print(f"  Bootstrap CIs ({N_BOOT} iterations)...")
    boot_hc, boot_cvd, boot_diff, boot_g = [], [], [], []
    for _ in range(N_BOOT):
        b_hc = hc_ve_b[rng.choice(n_hc, n_hc, replace=True)]
        b_cvd = cvd_ve_b[rng.choice(len(cvd_ve_b), len(cvd_ve_b), replace=True)]
        boot_hc.append(b_hc.mean())
        boot_cvd.append(b_cvd.mean())
        boot_diff.append(b_hc.mean() - b_cvd.mean())
        boot_g.append(hedges_g(b_cvd, b_hc))
    boot_hc = np.array(boot_hc)
    boot_cvd = np.array(boot_cvd)
    boot_diff = np.array(boot_diff)
    boot_g = np.array(boot_g)

    return {
        'roi': roi_display,
        'k': k,
        'framework_a': {
            'description': 'Single SRM: HC trained W, CVD SVD-projected W',
            'hc_ve': {s: float(hc_ve_a[i]) for i, s in enumerate(HC_SUBJECTS)},
            'cvd_ve': {s: float(cvd_ve_a[j]) for j, s in enumerate(CVD_SUBJECTS)},
            'hc_mean': float(hc_ve_a.mean()),
            'hc_std': float(hc_ve_a.std(ddof=1)),
            'cvd_mean': float(cvd_ve_a.mean()),
            'cvd_std': float(cvd_ve_a.std(ddof=1)),
            'diff': float(hc_ve_a.mean() - cvd_ve_a.mean()),
        },
        'framework_b': {
            'description': 'LOSO: both HC and CVD use SVD projection (unbiased)',
            'hc_ve': {s: float(hc_ve_b[i]) for i, s in enumerate(HC_SUBJECTS)},
            'cvd_ve': {s: float(cvd_ve_b[j]) for j, s in enumerate(CVD_SUBJECTS)},
            'summary': {
                'hc_mean': float(hc_ve_b.mean()),
                'hc_std': float(hc_ve_b.std(ddof=1)),
                'hc_ci': [float(np.percentile(boot_hc, 2.5)), float(np.percentile(boot_hc, 97.5))],
                'cvd_mean': float(cvd_ve_b.mean()),
                'cvd_std': float(cvd_ve_b.std(ddof=1)),
                'cvd_ci': [float(np.percentile(boot_cvd, 2.5)), float(np.percentile(boot_cvd, 97.5))],
                'diff': float(sep_b),
                'diff_ci': [float(np.percentile(boot_diff, 2.5)), float(np.percentile(boot_diff, 97.5))],
                'hedges_g': float(g_b),
                'hedges_g_ci': [float(np.percentile(boot_g, 2.5)), float(np.percentile(boot_g, 97.5))],
            },
            'mann_whitney': {'U': float(u_stat), 'p': float(u_p)},
            'permutation': {'observed_diff': float(observed_diff_b),
                             'null_mean': float(null_diffs.mean()),
                             'p': perm_p, 'n_perm': N_PERM},
            'individual_cvd': individual_b,
        },
    }


# ============================================================================
# CONVERGENT VALIDITY
# ============================================================================

def compute_convergent_validity(results, srm_disps):
    """Correlate VE with SRM disparities (expect negative r: higher VE = lower disparity)."""
    print(f"\n{'=' * 60}")
    print("Convergent Validity: VE vs SRM Disparity")
    print(f"{'=' * 60}")

    convergent = {}
    all_ve, all_srm = [], []

    for roi_d in ['V1', 'V2', 'V3', 'hV4']:
        if roi_d not in results or roi_d not in srm_disps:
            continue

        # Use LOSO VE for fair comparison
        ve_dict = results[roi_d]['framework_b']
        # Merge HC and CVD VE
        subject_ve = {}
        subject_ve.update(ve_dict['hc_ve'])
        subject_ve.update(ve_dict['cvd_ve'])

        srm_d = srm_disps[roi_d]
        subjects = [s for s in ALL_SUBJECTS if s in subject_ve and s in srm_d]
        ve_vals = [subject_ve[s] for s in subjects]
        srm_vals = [srm_d[s] for s in subjects]

        valid = [(v, s) for v, s in zip(ve_vals, srm_vals) if np.isfinite(v) and np.isfinite(s)]
        if len(valid) < 4:
            convergent[roi_d] = {'r': float('nan'), 'p': float('nan'), 'n': len(valid)}
            continue

        v_arr = np.array([x[0] for x in valid])
        s_arr = np.array([x[1] for x in valid])
        r, p = spearmanr(v_arr, s_arr)

        convergent[roi_d] = {'r': float(r), 'p': float(p), 'n': len(valid)}
        print(f"  {roi_d}: Spearman r={r:.3f}, p={p:.4f} (n={len(valid)}) {'(expected negative)' if r < 0 else ''}")

        all_ve.extend(v_arr.tolist())
        all_srm.extend(s_arr.tolist())

    if len(all_ve) >= 4:
        r_pool, p_pool = spearmanr(all_ve, all_srm)
        convergent['pooled'] = {'r': float(r_pool), 'p': float(p_pool), 'n': len(all_ve)}
        print(f"  Pooled: Spearman r={r_pool:.3f}, p={p_pool:.4f} (n={len(all_ve)})")

    return convergent


# ============================================================================
# MAIN
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("A3: Variance Explained Analysis")
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

    # --- Summary tables ---
    print(f"\n\n{'=' * 60}")
    print("SUMMARY TABLES")
    print(f"{'=' * 60}")

    # Framework A
    print("\n--- Framework A: Single SRM (confounded) ---")
    print(f"{'ROI':<5} | {'HC VE':>8} | {'CVD VE':>8} | {'Diff':>8}")
    print("-" * 40)
    for roi_d in ['V1', 'V2', 'V3', 'hV4']:
        if roi_d not in all_results:
            continue
        a = all_results[roi_d]['framework_a']
        print(f"{roi_d:<5} | {a['hc_mean']:>8.4f} | {a['cvd_mean']:>8.4f} | {a['diff']:>8.4f}")

    # Framework B
    print("\n--- Framework B: LOSO (unbiased) ---")
    print(f"{'ROI':<5} | {'HC VE [95% CI]':<24} | {'CVD VE [95% CI]':<24} | {'Diff [95% CI]':<24} | {'g [95% CI]':<22} | {'p_U':>7} | {'p_perm':>7}")
    print("-" * 140)
    for roi_d in ['V1', 'V2', 'V3', 'hV4']:
        if roi_d not in all_results:
            continue
        b = all_results[roi_d]['framework_b']['summary']
        mw = all_results[roi_d]['framework_b']['mann_whitney']
        perm = all_results[roi_d]['framework_b']['permutation']
        print(f"{roi_d:<5} | "
              f"{b['hc_mean']:.3f} [{b['hc_ci'][0]:.3f}, {b['hc_ci'][1]:.3f}] | "
              f"{b['cvd_mean']:.3f} [{b['cvd_ci'][0]:.3f}, {b['cvd_ci'][1]:.3f}] | "
              f"{b['diff']:.3f} [{b['diff_ci'][0]:.3f}, {b['diff_ci'][1]:.3f}] | "
              f"{b['hedges_g']:.2f} [{b['hedges_g_ci'][0]:.2f}, {b['hedges_g_ci'][1]:.2f}] | "
              f"{mw['p']:>7.4f} | {perm['p']:>7.4f}")

    # Individual CVD (Framework B)
    print("\n--- Individual CVD Tests: LOSO VE (Crawford & Howell) ---")
    print(f"{'Subject':<8} | {'V1 VE (t, p)':<22} | {'V2 VE (t, p)':<22} | {'V3 VE (t, p)':<22} | {'hV4 VE (t, p)':<22}")
    print("-" * 100)
    for s in CVD_SUBJECTS:
        parts = []
        for roi_d in ['V1', 'V2', 'V3', 'hV4']:
            if roi_d in all_results:
                iv = all_results[roi_d]['framework_b']['individual_cvd'].get(s, {})
                if iv:
                    sig = '*' if iv['significant'] else ''
                    parts.append(f"{iv['ve']:.3f} ({iv['t_stat']:.1f}, {iv['p_value']:.3f}){sig}")
                else:
                    parts.append("N/A")
            else:
                parts.append("N/A")
        print(f"{s:<8} | " + " | ".join(f"{p:<22}" for p in parts))

    # --- Save ---
    elapsed = time.time() - start
    output = {
        'config': {
            'method': 'variance_explained',
            'description': 'SRM variance explained: Framework A (single) + Framework B (LOSO, unbiased)',
            'k_values': K_VALUES,
            'n_perm': N_PERM,
            'n_boot': N_BOOT,
            'seed': SEED,
            'elapsed_sec': elapsed,
        },
        'results': all_results,
        'convergent_validity': convergent,
    }

    out_file = OUTPUT_DIR / "ve_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Results saved to: {out_file}")


if __name__ == '__main__':
    main()
