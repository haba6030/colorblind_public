#!/usr/bin/env python3
"""
permutation_test_loco.py — 10K Permutation Test for LOCO voxel_corr.

Loads raw amplitudes, re-fits ridge_gcv with shuffled color labels.
Generates null distribution and computes permutation p-value.

Algorithm per iteration:
  1. For each HC subject: perm = rng.permutation(8) -> amp_shuffled = amp[:, perm, :]
  2. Standard 8-fold LOCO on shuffled data (hold out 1 color, train on 7, GCV alpha, ridge)
  3. Mean voxel_corr across 8 folds -> subject-level null LOCO_r
  4. Mean across 7 HC subjects -> group-level null statistic
  5. After 10K: p_perm = mean(null >= observed)

Runtime: ~15 min for 3 ROIs (10K perms x 7 subjects x 8 folds = 560K fits per ROI).

Usage:
    python permutation_test_loco.py \
        --baseline_dir ../../phase1_preprocess_decoding/results/full_dataset_C010 \
        --validation_dir ../results/validation \
        --output_dir ../results/loco_reinforcement \
        --n_perms 10000 --seed 42
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/permutation_test_loco.py  (development repository, commit 53c81c2)
# Role      : Group-level subject-label permutation for LOCO (10,000 permutations).
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
import time
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils_forward_model import (
    HC_SUBJECTS, ROIS, N_RUNS, N_COLORS, N_CHANNELS,
    HUE_ANGLES, ALPHA_GRID,
    load_amplitudes, create_basis_matrix,
    fit_W_ridge, gcv_select_alpha,
    predict_patterns, voxel_pattern_correlation,
)

ROI_DISPLAY = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}


# ============================================================================
# LOCO procedure (single subject, single ROI)
# ============================================================================

def loco_ridge_gcv(amp, C_full):
    """Standard 8-fold LOCO with GCV alpha selection.

    Args:
        amp: (6, 8, V_s) amplitudes (possibly shuffled)
        C_full: (8, K) design matrix for all 8 hue angles

    Returns:
        mean_voxel_corr: float, mean across 8 folds
    """
    V_s = amp.shape[2]
    fold_corrs = np.zeros(N_COLORS)

    for test_color in range(N_COLORS):
        train_colors = [c for c in range(N_COLORS) if c != test_color]
        C_train_row = C_full[train_colors]  # (7, K)

        # Pool all runs for training colours
        X_train = amp[:, train_colors, :].reshape(-1, V_s)  # (42, V_s)
        C_train = np.tile(C_train_row, (N_RUNS, 1))          # (42, K)

        # GCV alpha selection + ridge fit
        best_alpha, _ = gcv_select_alpha(C_train, X_train, ALPHA_GRID)
        W = fit_W_ridge(C_train, X_train, best_alpha)  # (K, V_s)

        # Predict test colour pattern
        C_test = C_full[test_color:test_color+1]  # (1, K)
        Y_hat = predict_patterns(W, C_test)  # (1, V_s)
        X_test_mean = amp[:, test_color, :].mean(axis=0, keepdims=True)  # (1, V_s)

        r = voxel_pattern_correlation(Y_hat, X_test_mean)
        fold_corrs[test_color] = r[0]

    return float(np.mean(fold_corrs))


# ============================================================================
# Data loading helpers
# ============================================================================

def find_loco_json(validation_dir, subject):
    """Find LOCO JSON, checking nested validation subdir."""
    p = Path(validation_dir) / f'sub-{subject}_loco.json'
    if p.exists():
        return p
    p2 = Path(validation_dir) / 'validation' / f'sub-{subject}_loco.json'
    if p2.exists():
        return p2
    raise FileNotFoundError(f'LOCO JSON not found for sub-{subject}')


def get_observed_hc_mean(validation_dir, roi):
    """Get observed HC mean LOCO voxel_corr (ridge_gcv) from JSONs."""
    vals = []
    for subj in HC_SUBJECTS:
        try:
            path = find_loco_json(validation_dir, subj)
            data = json.loads(path.read_text())
            if roi in data and 'ridge_gcv' in data[roi]:
                vals.append(data[roi]['ridge_gcv']['mean_voxel_corr'])
        except FileNotFoundError:
            pass
    return float(np.mean(vals)) if vals else np.nan


# ============================================================================
# Main permutation loop
# ============================================================================

def run_permutation_test(baseline_dir, validation_dir, output_dir,
                         n_perms, seed, rois):
    """Run permutation test for all ROIs."""
    rng = np.random.default_rng(seed)
    C_full = create_basis_matrix(HUE_ANGLES, N_CHANNELS, 'fe')  # (8, 6)

    results = {}

    for roi in rois:
        roi_disp = ROI_DISPLAY[roi]
        print(f'\n--- {roi_disp} ---')

        # Get observed value
        observed = get_observed_hc_mean(validation_dir, roi)
        print(f'  Observed HC mean LOCO_r: {observed:.4f}')

        # Load all HC amplitudes
        hc_amps = {}
        for subj in HC_SUBJECTS:
            try:
                amp = load_amplitudes(baseline_dir, subj, roi)
                hc_amps[subj] = amp
            except FileNotFoundError:
                print(f'  Warning: missing amplitudes for sub-{subj} {roi}')

        available_subjects = list(hc_amps.keys())
        n_subj = len(available_subjects)
        print(f'  Loaded {n_subj} HC subjects')

        if n_subj == 0:
            print(f'  Skipping {roi_disp}: no data')
            continue

        # Permutation loop
        null_dist = np.zeros(n_perms)
        t0 = time.time()

        for perm_i in range(n_perms):
            subj_corrs = np.zeros(n_subj)

            for s_idx, subj in enumerate(available_subjects):
                amp = hc_amps[subj]
                # Shuffle color labels
                perm = rng.permutation(N_COLORS)
                amp_shuffled = amp[:, perm, :]
                # LOCO on shuffled data
                subj_corrs[s_idx] = loco_ridge_gcv(amp_shuffled, C_full)

            null_dist[perm_i] = np.mean(subj_corrs)

            # Progress
            if (perm_i + 1) % 500 == 0:
                elapsed = time.time() - t0
                rate = (perm_i + 1) / elapsed
                eta = (n_perms - perm_i - 1) / rate
                print(f'  [{perm_i+1}/{n_perms}] '
                      f'null mean={np.mean(null_dist[:perm_i+1]):.4f}, '
                      f'{rate:.0f} perm/s, ETA {eta:.0f}s')

        elapsed = time.time() - t0
        p_perm = float(np.mean(null_dist >= observed))
        null_mean = float(np.mean(null_dist))
        null_sd = float(np.std(null_dist))
        null_ci = (float(np.percentile(null_dist, 2.5)),
                   float(np.percentile(null_dist, 97.5)))

        print(f'  p_perm = {p_perm:.4f} ({elapsed:.1f}s)')
        print(f'  Null: mean={null_mean:.4f}, SD={null_sd:.4f}, '
              f'95%CI=[{null_ci[0]:.4f}, {null_ci[1]:.4f}]')

        # Save null distribution
        np.save(output_dir / f'permutation_null_{roi_disp}.npy', null_dist)

        # Plot
        plot_permutation(null_dist, observed, p_perm, roi_disp, output_dir)

        results[roi_disp] = {
            'observed': observed,
            'p_perm': p_perm,
            'n_perms': n_perms,
            'null_mean': null_mean,
            'null_sd': null_sd,
            'null_ci_95': null_ci,
            'n_subjects': n_subj,
            'elapsed_s': round(elapsed, 1),
        }

    return results


def plot_permutation(null_dist, observed, p_perm, roi_display, output_dir):
    """Histogram of null distribution with observed line."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(null_dist, bins=80, color='steelblue', alpha=0.7, edgecolor='white',
            linewidth=0.3, density=True, label='Null distribution')
    ax.axvline(observed, color='red', linewidth=2, linestyle='--',
               label=f'Observed = {observed:.4f}')

    # Shade p-value area
    extreme = null_dist[null_dist >= observed]
    if len(extreme) > 0:
        ax.axvspan(observed, np.max(null_dist) * 1.05, alpha=0.1, color='red')

    ax.set_xlabel('Mean HC LOCO voxel correlation (r)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'{roi_display} — Permutation Test (n={len(null_dist):,})\n'
                 f'p = {p_perm:.4f}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f'permutation_test_{roi_display}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Permutation test for LOCO')
    parser.add_argument('--baseline_dir', type=str, required=True,
                        help='Path to full_dataset_C010')
    parser.add_argument('--validation_dir', type=str, required=True,
                        help='Path to validation results directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--n_perms', type=int, default=10000,
                        help='Number of permutations (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'V4'],
                        help='ROIs to test')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print(f'Permutation Test for LOCO (n_perms={args.n_perms}, seed={args.seed})')
    print('=' * 60)

    results = run_permutation_test(
        baseline_dir=Path(args.baseline_dir),
        validation_dir=Path(args.validation_dir),
        output_dir=output_dir,
        n_perms=args.n_perms,
        seed=args.seed,
        rois=args.rois,
    )

    # Save results JSON
    perm_out = {
        'n_perms': args.n_perms,
        'seed': args.seed,
        'results': results,
    }
    with open(output_dir / 'permutation_test.json', 'w') as f:
        json.dump(perm_out, f, indent=2)
    print(f'\nSaved: {output_dir / "permutation_test.json"}')

    # Update config
    config_path = output_dir / 'config.json'
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {}
    config['permutation_test'] = {
        'script': 'permutation_test_loco.py',
        'n_perms': args.n_perms,
        'seed': args.seed,
        'baseline_dir': str(args.baseline_dir),
        'validation_dir': str(args.validation_dir),
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)

    # Summary
    print('\n' + '=' * 60)
    print('SUMMARY')
    print('=' * 60)
    for roi_disp, res in results.items():
        sig = '*' if res['p_perm'] < 0.05 else ''
        print(f'  {roi_disp}: observed={res["observed"]:.4f}, '
              f'p_perm={res["p_perm"]:.4f}{sig}, '
              f'null_mean={res["null_mean"]:.4f}')

    print('\nDone.')


if __name__ == '__main__':
    main()
