#!/usr/bin/env python3
"""
diagnostic_delta_rdm.py — ΔRDM sanity check before full fitting.

ΔRDM approach:
  Simulation side:
    ΔRDM_sim(δ) = RDM( C(θ+δ) @ W_HC ) - RDM( C(θ) @ W_HC )
    → Pure effect of cone shift on pairwise geometry, no basis-response mismatch.

  Observation side:
    ΔRDM_obs = RDM_CVD - RDM_HC_mean
    → Observed HC-CVD difference in pairwise distances.

  Both use within-subject RDMs → alignment-free.

Distance metrics:
  - correlation distance (1 - Pearson r): standard, scale-free
  - crossnobis (Mahalanobis with noise covariance): noise-normalized

Comparison metrics:
  - Pearson r / cosine similarity: magnitude-sensitive (primary)
  - Spearman ρ: rank-only (secondary)
  - Signed agreement rate: conservative lower bound

Three sanity checks:
  1. Is ΔRDM_obs nonzero and structured? (not pure noise)
  2. Does cone-shift theory predict the right pairs? (deutan → red-green axis)
  3. Does ΔRDM_sim change systematically with δθ? (not flat)

Usage:
    conda activate srm
    python scripts/diagnostic_delta_rdm.py \
        --rois V1 V2 V4 \
        --output_dir results/v2/diagnostic_delta_rdm
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/diagnostic_delta_rdm.py  (development repository, commit 53c81c2)
# Role      : Delta-RDM computation and simulated Delta-RDM under a candidate distortion (imported by neural_loss.py).
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
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import sys

_SCRIPT_DIR = Path(__file__).resolve().parent
_PHASE2_DIR = _SCRIPT_DIR.parent
sys.path.insert(0, str(_SCRIPT_DIR))

for _base in [_PHASE2_DIR.parent, _PHASE2_DIR.parent.parent]:
    _fwd = _base / 'phase4_forward_model' / 'scripts'
    if _fwd.exists() and str(_fwd) not in sys.path:
        sys.path.insert(0, str(_fwd))
        break

from utils_forward_model import (
    HC_SUBJECTS, CVD_SUBJECTS, ROIS, N_CHANNELS, N_RUNS, N_COLORS,
    HUE_ANGLES, load_amplitudes, create_basis_matrix,
    gcv_select_alpha, fit_W_ridge,
)
from utils_distortion_models import get_design_matrix

LOCAL_BASELINE = _PHASE2_DIR.parent / 'phase1_preprocess_decoding' / 'results' / 'full_dataset_C010'

CVD_TYPE = {'08': 'deutan', '09': 'protan', '10': 'deutan'}  # sub-10 is mild deutan per project CLAUDE.md §6

COLOR_NAMES = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']

# 28 pair labels for 8 colors
PAIR_LABELS = []
for i in range(N_COLORS):
    for j in range(i + 1, N_COLORS):
        PAIR_LABELS.append(f'{COLOR_NAMES[i]}-{COLOR_NAMES[j]}')


# ============================================================================
# RDM computation
# ============================================================================

def compute_rdm_correlation(patterns):
    """Correlation distance RDM (1 - Pearson r).

    Args:
        patterns: (n_cond, V_s) mean patterns per condition

    Returns:
        rdm_upper: (28,) upper triangle of 8×8 RDM
    """
    return pdist(patterns, metric='correlation')


def estimate_noise_cov(amplitudes):
    """Estimate noise covariance from run-level residuals.

    For each color, compute residual = run_response - mean_response,
    then pool across colors.

    Args:
        amplitudes: (N_RUNS, N_COLORS, V_s)

    Returns:
        sigma: (V_s, V_s) noise covariance (regularized)
    """
    n_runs, n_colors, V_s = amplitudes.shape
    residuals = []
    for c in range(n_colors):
        mean_c = amplitudes[:, c, :].mean(axis=0, keepdims=True)
        resid_c = amplitudes[:, c, :] - mean_c  # (N_RUNS, V_s)
        residuals.append(resid_c)
    residuals = np.vstack(residuals)  # (N_RUNS * N_COLORS, V_s)

    # Shrinkage regularization (Ledoit-Wolf style, simplified)
    n_samples = residuals.shape[0]
    sigma = (residuals.T @ residuals) / (n_samples - 1)

    # Regularize: sigma_reg = (1-shrink)*sigma + shrink*trace(sigma)/V_s * I
    trace_mean = np.trace(sigma) / V_s
    shrinkage = 0.1
    sigma_reg = (1 - shrinkage) * sigma + shrinkage * trace_mean * np.eye(V_s)

    return sigma_reg


def compute_rdm_crossnobis(amplitudes):
    """Crossnobis (cross-validated Mahalanobis) distance RDM.

    Uses leave-one-run-out cross-validation:
    For each run pair (a, b):
      d(i,j) = (mean_a[i] - mean_a[j])^T @ Σ^{-1}_b @ (mean_b[i] - mean_b[j])

    This is unbiased and noise-normalized.

    Args:
        amplitudes: (N_RUNS, N_COLORS, V_s)

    Returns:
        rdm_upper: (28,) upper triangle, can be negative (unbiased)
    """
    n_runs, n_colors, V_s = amplitudes.shape

    # If V_s is very large, use whitening approach for efficiency
    # Estimate noise precision from all data (shared across folds)
    sigma_reg = estimate_noise_cov(amplitudes)

    # Whitening matrix via Cholesky
    try:
        L = np.linalg.cholesky(sigma_reg)
        L_inv = np.linalg.solve(L, np.eye(V_s))
    except np.linalg.LinAlgError:
        # Fallback: use eigendecomposition with truncation
        eigvals, eigvecs = np.linalg.eigh(sigma_reg)
        keep = eigvals > eigvals.max() * 1e-6
        L_inv = eigvecs[:, keep] @ np.diag(1.0 / np.sqrt(eigvals[keep]))
        L_inv = L_inv  # (V_s, k) — dimensionality reduction

    # Cross-validated distances
    rdm_sum = np.zeros((n_colors, n_colors))
    n_pairs = 0

    for a in range(n_runs):
        for b in range(a + 1, n_runs):
            # Whiten patterns from each run
            pat_a = amplitudes[a] @ L_inv  # (n_colors, V_s or k)
            pat_b = amplitudes[b] @ L_inv

            for i in range(n_colors):
                for j in range(i + 1, n_colors):
                    diff_a = pat_a[i] - pat_a[j]
                    diff_b = pat_b[i] - pat_b[j]
                    rdm_sum[i, j] += np.dot(diff_a, diff_b)
                    rdm_sum[j, i] += np.dot(diff_a, diff_b)
            n_pairs += 1

    rdm_mean = rdm_sum / n_pairs
    return rdm_mean[np.triu_indices(n_colors, k=1)]


def compute_rdm_crossnobis_predicted(predicted_patterns, noise_cov_inv_half):
    """Crossnobis-style distance for predicted patterns (no cross-validation).

    For predicted patterns (single set, no runs), compute noise-normalized
    pairwise distances using the noise covariance from actual data.

    Args:
        predicted_patterns: (n_cond, V_s) predicted mean patterns
        noise_cov_inv_half: (V_s, k) whitening matrix (L_inv from actual data)

    Returns:
        rdm_upper: (28,) upper triangle
    """
    whitened = predicted_patterns @ noise_cov_inv_half  # (n_cond, k)
    return pdist(whitened, metric='sqeuclidean')


# ============================================================================
# ΔRDM computation
# ============================================================================

def compute_delta_rdm_obs(amp_cvd, hc_amps_dict, distance='correlation'):
    """Compute observed ΔRDM = RDM_CVD - RDM_HC_mean.

    Args:
        amp_cvd: (N_RUNS, N_COLORS, V_s) CVD amplitudes
        hc_amps_dict: dict {subj: (N_RUNS, N_COLORS, V_s)}
        distance: 'correlation' or 'crossnobis'

    Returns:
        delta_rdm: (28,) observed distortion vector
        rdm_cvd: (28,) CVD RDM
        rdm_hc_mean: (28,) mean HC RDM
        rdm_hc_individual: dict {subj: (28,)} per-HC RDMs
    """
    # CVD RDM
    if distance == 'correlation':
        rdm_cvd = compute_rdm_correlation(amp_cvd.mean(axis=0))
    else:
        rdm_cvd = compute_rdm_crossnobis(amp_cvd)

    # HC RDMs
    rdm_hc_individual = {}
    for subj, amp in hc_amps_dict.items():
        if distance == 'correlation':
            rdm_hc_individual[subj] = compute_rdm_correlation(amp.mean(axis=0))
        else:
            rdm_hc_individual[subj] = compute_rdm_crossnobis(amp)

    rdm_hc_mean = np.mean(list(rdm_hc_individual.values()), axis=0)

    delta_rdm = rdm_cvd - rdm_hc_mean
    return delta_rdm, rdm_cvd, rdm_hc_mean, rdm_hc_individual


def compute_delta_rdm_sim(hc_W_dict, C_shifted, C_baseline,
                          hc_amps_dict=None, distance='correlation'):
    """Compute simulated ΔRDM = RDM(C(θ+δ)@W) - RDM(C(θ)@W), averaged over HC.

    Args:
        hc_W_dict: dict {subj: (K, V_s)} precomputed weights
        C_shifted: (8, K) shifted design matrix
        C_baseline: (8, K) original design matrix
        hc_amps_dict: needed for crossnobis (noise covariance)
        distance: 'correlation' or 'crossnobis'

    Returns:
        delta_rdm_mean: (28,) mean ΔRDM across HCs
        delta_rdm_per_hc: dict {subj: (28,)}
    """
    delta_rdm_per_hc = {}

    for subj, W in hc_W_dict.items():
        Y_shifted = C_shifted @ W     # (8, V_s)
        Y_baseline = C_baseline @ W   # (8, V_s)

        if distance == 'correlation':
            rdm_shifted = compute_rdm_correlation(Y_shifted)
            rdm_baseline = compute_rdm_correlation(Y_baseline)
        else:
            # For crossnobis on predictions, use noise cov from actual data
            if hc_amps_dict is None:
                raise ValueError('hc_amps_dict needed for crossnobis')
            sigma_reg = estimate_noise_cov(hc_amps_dict[subj])
            try:
                L = np.linalg.cholesky(sigma_reg)
                L_inv = np.linalg.solve(L, np.eye(sigma_reg.shape[0]))
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(sigma_reg)
                keep = eigvals > eigvals.max() * 1e-6
                L_inv = eigvecs[:, keep] @ np.diag(
                    1.0 / np.sqrt(eigvals[keep]))

            rdm_shifted = compute_rdm_crossnobis_predicted(Y_shifted, L_inv)
            rdm_baseline = compute_rdm_crossnobis_predicted(Y_baseline, L_inv)

        delta_rdm_per_hc[subj] = rdm_shifted - rdm_baseline

    delta_rdm_mean = np.mean(list(delta_rdm_per_hc.values()), axis=0)
    return delta_rdm_mean, delta_rdm_per_hc


# ============================================================================
# Comparison metrics
# ============================================================================

def cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def signed_agreement_rate(a, b):
    """Proportion of pairs where ΔRDM_sim and ΔRDM_obs have the same sign.

    Excludes near-zero elements (|x| < threshold) from both vectors.

    Returns:
        agreement: float in [0, 1]
        n_valid: number of pairs compared
    """
    a, b = np.asarray(a), np.asarray(b)
    # Threshold: exclude near-zero in either vector
    threshold = 1e-8
    valid = (np.abs(a) > threshold) & (np.abs(b) > threshold)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return 0.5, 0  # uninformative
    agreement = float(np.mean(np.sign(a[valid]) == np.sign(b[valid])))
    return agreement, n_valid


def compare_delta_rdm(sim, obs):
    """Compute all comparison metrics between ΔRDM_sim and ΔRDM_obs.

    Args:
        sim: (28,) simulated ΔRDM
        obs: (28,) observed ΔRDM

    Returns:
        metrics: dict of all metrics
    """
    # Primary: Pearson r and cosine similarity
    if np.std(sim) == 0 or np.std(obs) == 0:
        pearson_r, pearson_p = 0.0, 1.0
    else:
        pearson_r, pearson_p = pearsonr(sim, obs)
        pearson_r = float(pearson_r) if np.isfinite(pearson_r) else 0.0
        pearson_p = float(pearson_p) if np.isfinite(pearson_p) else 1.0

    cosine = cosine_similarity(sim, obs)

    # Secondary: Spearman ρ
    if np.std(sim) == 0 or np.std(obs) == 0:
        spearman_r, spearman_p = 0.0, 1.0
    else:
        spearman_r, spearman_p = spearmanr(sim, obs)
        spearman_r = float(spearman_r) if np.isfinite(spearman_r) else 0.0
        spearman_p = float(spearman_p) if np.isfinite(spearman_p) else 1.0

    # Conservative: signed agreement
    agreement, n_valid = signed_agreement_rate(sim, obs)

    return {
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        'cosine': cosine,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'signed_agreement': agreement,
        'signed_agreement_n': n_valid,
    }


# ============================================================================
# W precomputation (reused from step1_fit_loco_v2.py)
# ============================================================================

def precompute_hc_W(hc_amps_dict, C_original):
    """Precompute ridge encoding weights for each HC subject."""
    hc_W_dict = {}
    hc_alpha_dict = {}
    C_pooled = np.tile(C_original, (N_RUNS, 1))

    for subj, amp in hc_amps_dict.items():
        V_s = amp.shape[2]
        X_all = amp.reshape(-1, V_s)
        alpha, _ = gcv_select_alpha(C_pooled, X_all)
        W = fit_W_ridge(C_pooled, X_all, alpha)
        hc_W_dict[subj] = W
        hc_alpha_dict[subj] = float(alpha)

    return hc_W_dict, hc_alpha_dict


# ============================================================================
# Sanity checks
# ============================================================================

def sanity_check_1_delta_rdm_obs(delta_rdm_obs, distance_name):
    """Check 1: Is ΔRDM_obs nonzero and structured?

    Reports: mean, std, range, top-5 largest absolute distortions.
    """
    abs_d = np.abs(delta_rdm_obs)
    top5_idx = np.argsort(abs_d)[-5:][::-1]

    result = {
        'mean': float(np.mean(delta_rdm_obs)),
        'std': float(np.std(delta_rdm_obs)),
        'range': [float(delta_rdm_obs.min()), float(delta_rdm_obs.max())],
        'abs_mean': float(abs_d.mean()),
        'top5_pairs': [
            {'pair': PAIR_LABELS[i],
             'delta': float(delta_rdm_obs[i]),
             'direction': 'closer' if delta_rdm_obs[i] < 0 else 'farther'}
            for i in top5_idx
        ],
        'nonzero_fraction': float(np.mean(abs_d > abs_d.mean() * 0.1)),
    }

    print(f'    Sanity 1 ({distance_name}): ΔRDM_obs structure')
    print(f'      mean={result["mean"]:.4f}, std={result["std"]:.4f}, '
          f'range=[{result["range"][0]:.4f}, {result["range"][1]:.4f}]')
    print(f'      Top 5 distorted pairs:')
    for item in result['top5_pairs']:
        print(f'        {item["pair"]}: Δ={item["delta"]:+.4f} ({item["direction"]})')

    return result


def sanity_check_2_theory_prediction(delta_rdm_obs, cvd_type, distance_name):
    """Check 2: Do theory-predicted confusion pairs show reduced distance?

    Deutan confusion: red-green axis (colors 0,3 / 1,4 / 7,5 roughly)
    Protan confusion: similar but shifted
    """
    # Confusion-line pairs (theory-predicted to have REDUCED distance in CVD)
    if cvd_type == 'deutan':
        # Deutan: confuses along red-green axis
        # Most affected: red(0)-green(3), orange(1)-cyan(4), magenta(7)-blue(5)
        confusion_pairs = ['red-green', 'orange-cyan', 'blue-magenta']
    elif cvd_type == 'protan':
        # Protan: similar confusion axis, slightly shifted
        confusion_pairs = ['red-green', 'orange-cyan', 'blue-magenta']
    else:
        confusion_pairs = []

    confusion_indices = [PAIR_LABELS.index(p) for p in confusion_pairs
                         if p in PAIR_LABELS]

    result = {
        'cvd_type': cvd_type,
        'confusion_pairs': confusion_pairs,
        'confusion_deltas': {},
    }

    if confusion_indices:
        for idx in confusion_indices:
            result['confusion_deltas'][PAIR_LABELS[idx]] = float(
                delta_rdm_obs[idx])

        # Are confusion pairs closer in CVD (negative ΔRDM)?
        confusion_values = delta_rdm_obs[confusion_indices]
        n_closer = int(np.sum(confusion_values < 0))
        result['n_closer'] = n_closer
        result['n_total'] = len(confusion_indices)
        result['mean_confusion_delta'] = float(confusion_values.mean())

        print(f'    Sanity 2 ({distance_name}): Theory-predicted confusion pairs')
        print(f'      {cvd_type} confusion pairs:')
        for pair, delta in result['confusion_deltas'].items():
            direction = 'CLOSER (expected)' if delta < 0 else 'farther (unexpected)'
            print(f'        {pair}: Δ={delta:+.4f} → {direction}')
        print(f'      {n_closer}/{len(confusion_indices)} pairs closer in CVD')

    return result


def sanity_check_3_sensitivity(hc_W_dict, hc_amps_dict, C_baseline,
                               delta_rdm_obs, cvd_type, distance_name,
                               delta_range=None):
    """Check 3: Does ΔRDM_sim change systematically with δθ?

    Sweep δθ from 0 to 60nm and compute metrics at each point.
    """
    if delta_range is None:
        delta_range = np.arange(0, 61, 5)

    results = []

    for delta in delta_range:
        C_shifted = get_design_matrix('cone_1way', [delta], cvd_type=cvd_type)

        if distance_name == 'crossnobis':
            delta_sim, _ = compute_delta_rdm_sim(
                hc_W_dict, C_shifted, C_baseline,
                hc_amps_dict=hc_amps_dict, distance='crossnobis')
        else:
            delta_sim, _ = compute_delta_rdm_sim(
                hc_W_dict, C_shifted, C_baseline, distance='correlation')

        metrics = compare_delta_rdm(delta_sim, delta_rdm_obs)
        metrics['delta_nm'] = float(delta)
        metrics['delta_rdm_sim_norm'] = float(np.linalg.norm(delta_sim))
        results.append(metrics)

    # Find best δθ for each metric
    pearson_rs = [r['pearson_r'] for r in results]
    cosines = [r['cosine'] for r in results]
    spearman_rs = [r['spearman_r'] for r in results]

    best_pearson_idx = int(np.argmax(pearson_rs))
    best_cosine_idx = int(np.argmax(cosines))
    best_spearman_idx = int(np.argmax(spearman_rs))

    summary = {
        'delta_range': delta_range.tolist(),
        'sweep': results,
        'best_pearson': {
            'delta_nm': results[best_pearson_idx]['delta_nm'],
            'pearson_r': results[best_pearson_idx]['pearson_r'],
            'pearson_p': results[best_pearson_idx]['pearson_p'],
        },
        'best_cosine': {
            'delta_nm': results[best_cosine_idx]['delta_nm'],
            'cosine': results[best_cosine_idx]['cosine'],
        },
        'best_spearman': {
            'delta_nm': results[best_spearman_idx]['delta_nm'],
            'spearman_r': results[best_spearman_idx]['spearman_r'],
            'spearman_p': results[best_spearman_idx]['spearman_p'],
        },
        'norm_at_0': results[0]['delta_rdm_sim_norm'],
        'norm_at_max': results[-1]['delta_rdm_sim_norm'],
        'monotonic': bool(np.all(np.diff(
            [r['delta_rdm_sim_norm'] for r in results]) >= -1e-10)),
    }

    print(f'    Sanity 3 ({distance_name}): δθ sensitivity')
    print(f'      ||ΔRDM_sim|| at δ=0: {summary["norm_at_0"]:.6f}, '
          f'at δ={delta_range[-1]}: {summary["norm_at_max"]:.4f}')
    print(f'      Monotonic increase: {summary["monotonic"]}')
    print(f'      Best Pearson:  δ={summary["best_pearson"]["delta_nm"]}nm, '
          f'r={summary["best_pearson"]["pearson_r"]:.3f} '
          f'(p={summary["best_pearson"]["pearson_p"]:.4f})')
    print(f'      Best Cosine:   δ={summary["best_cosine"]["delta_nm"]}nm, '
          f'cos={summary["best_cosine"]["cosine"]:.3f}')
    print(f'      Best Spearman: δ={summary["best_spearman"]["delta_nm"]}nm, '
          f'ρ={summary["best_spearman"]["spearman_r"]:.3f} '
          f'(p={summary["best_spearman"]["spearman_p"]:.4f})')

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ΔRDM diagnostic: sanity checks before full fitting')
    parser.add_argument('--output_dir', type=str,
                        default='results/v2/diagnostic_delta_rdm')
    parser.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V4'])
    parser.add_argument('--cvd_subjects', nargs='+', default=CVD_SUBJECTS)
    parser.add_argument('--baseline_dir', type=str,
                        default=str(LOCAL_BASELINE))
    parser.add_argument('--distances', nargs='+',
                        default=['correlation', 'crossnobis'],
                        choices=['correlation', 'crossnobis'],
                        help='Distance metrics to compute (default: both)')
    args = parser.parse_args()

    print('=' * 60)
    print('ΔRDM Diagnostic: Sanity Checks')
    print(f'ROIs: {args.rois}')
    print(f'CVD subjects: {args.cvd_subjects}')
    print(f'Distances: {", ".join(args.distances)}')
    print(f'Metrics: Pearson, cosine, Spearman, signed agreement')
    print('=' * 60)

    C_baseline = create_basis_matrix(HUE_ANGLES, N_CHANNELS)

    for roi in args.rois:
        print(f'\n{"="*50}')
        print(f'ROI: {roi}')
        print(f'{"="*50}')

        # Load all HC amplitudes
        hc_amps = {}
        for subj in HC_SUBJECTS:
            hc_amps[subj] = load_amplitudes(args.baseline_dir, subj, roi)
        print(f'  Loaded {len(hc_amps)} HC subjects')

        # Precompute W_HC
        hc_W, hc_alphas = precompute_hc_W(hc_amps, C_baseline)
        print(f'  Precomputed W (alphas: '
              + ', '.join(f'{a:.1f}' for a in hc_alphas.values()) + ')')

        for cvd_subj in args.cvd_subjects:
            cvd_type = CVD_TYPE[cvd_subj.split('-')[-1]]  # accept 'sub-08' or '08'
            print(f'\n  --- sub-{cvd_subj} ({cvd_type}) ---')

            # Load CVD amplitudes
            amp_cvd = load_amplitudes(args.baseline_dir, cvd_subj, roi)

            subject_results = {
                'subject': cvd_subj,
                'roi': roi,
                'cvd_type': cvd_type,
                'timestamp': datetime.now().isoformat(),
                'hc_alphas': hc_alphas,
                'distances': {},
            }

            for dist_name in args.distances:
                print(f'\n    === Distance: {dist_name} ===')

                # Compute observed ΔRDM
                delta_obs, rdm_cvd, rdm_hc_mean, rdm_hc_ind = \
                    compute_delta_rdm_obs(amp_cvd, hc_amps, distance=dist_name)

                # Sanity check 1: structure
                sc1 = sanity_check_1_delta_rdm_obs(delta_obs, dist_name)

                # Sanity check 2: theory prediction
                sc2 = sanity_check_2_theory_prediction(
                    delta_obs, cvd_type, dist_name)

                # Sanity check 3: δθ sensitivity sweep
                sc3 = sanity_check_3_sensitivity(
                    hc_W, hc_amps, C_baseline, delta_obs,
                    cvd_type, dist_name)

                # HC inter-subject RDM variability (for context)
                hc_rdm_matrix = np.array(list(rdm_hc_ind.values()))
                hc_rdm_std = hc_rdm_matrix.std(axis=0)

                subject_results['distances'][dist_name] = {
                    'delta_rdm_obs': delta_obs.tolist(),
                    'rdm_cvd': rdm_cvd.tolist(),
                    'rdm_hc_mean': rdm_hc_mean.tolist(),
                    'hc_rdm_std_mean': float(hc_rdm_std.mean()),
                    'sanity_1_structure': sc1,
                    'sanity_2_theory': sc2,
                    'sanity_3_sensitivity': sc3,
                    'pair_labels': PAIR_LABELS,
                }

            # Save
            out_dir = Path(args.output_dir) / roi
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'sub-{cvd_subj}_delta_rdm_diagnostic.json'
            with open(out_path, 'w') as f:
                json.dump(subject_results, f, indent=2)
            print(f'\n    Saved: {out_path}')

    # Summary table
    print('\n' + '=' * 70)
    print('SUMMARY: Best δθ per metric (cone_1way sweep)')
    print('=' * 70)
    print(f'{"ROI":<5} {"Subj":<8} {"CVD":<7} {"Dist":<12} '
          f'{"Pearson":<18} {"Cosine":<18} {"Spearman":<18}')
    print('-' * 70)

    # Re-read saved results for summary
    for roi in args.rois:
        for cvd_subj in args.cvd_subjects:
            out_path = (Path(args.output_dir) / roi
                        / f'sub-{cvd_subj}_delta_rdm_diagnostic.json')
            if not out_path.exists():
                continue
            with open(out_path) as f:
                data = json.load(f)

            for dist_name in args.distances:
                if dist_name not in data.get('distances', {}):
                    continue
                d = data['distances'][dist_name]['sanity_3_sensitivity']
                bp = d['best_pearson']
                bc = d['best_cosine']
                bs = d['best_spearman']
                print(
                    f'{roi:<5} sub-{cvd_subj:<4} {data["cvd_type"]:<7} '
                    f'{dist_name:<12} '
                    f'δ={bp["delta_nm"]:>2.0f} r={bp["pearson_r"]:+.3f}  '
                    f'δ={bc["delta_nm"]:>2.0f} c={bc["cosine"]:+.3f}  '
                    f'δ={bs["delta_nm"]:>2.0f} ρ={bs["spearman_r"]:+.3f}'
                )

    print('\nDiagnostic complete.')


if __name__ == '__main__':
    main()
