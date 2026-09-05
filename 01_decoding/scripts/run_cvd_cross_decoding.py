#!/usr/bin/env python3
"""
CVD Cross-Decoding with HC-only SRM (RT-7 fix)

Trains SRM on HC subjects only, projects CVD via SVD, then tests LDA decoding.
This eliminates the circularity of the old all-subjects SRM approach.

Pipeline:
1. Load raw amplitudes (6 runs x 8 colors x n_voxels) per subject
2. Train SRM on 7 HC (mean betas), get transformation weights
3. Transform HC data using srm.w_[i]
4. Project CVD data via SVD into HC-trained space
5. HC baseline: LOSO LDA decoding (train on 6 HC, test on held-out)
6. CVD test: Train LDA on all 7 HC, test on each CVD
7. Permutation test (shuffle training labels, 1000 iterations)

Usage:
    mpirun -np 1 python run_cvd_cross_decoding.py [--n_permutations 1000]
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase3_decoder_comparing/model_comparison_validation/scripts/run_cvd_cross_decoding.py  (development repository, commit 53c81c2)
# Role      : Control-trained encoder applied to CVD participants (RT-7, HC-only SRM).
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


import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

# BrainIAK for SRM
from brainiak.funcalign.srm import SRM

# ============================================================================
# Configuration
# ============================================================================

HC_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 8)]   # sub-01 ~ sub-07
CVD_SUBJECTS = [f"sub-{i:02d}" for i in range(8, 11)]  # sub-08 ~ sub-10
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS

ROIS = ['V1', 'V2', 'V3', 'hV4']
ROI_DIR_MAP = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'hV4': 'V4'}

# Corrected k values (from mean rank aggregation 2026-02-18)
SRM_K = {'V1': 4, 'V2': 4, 'V3': 3, 'hV4': 3}

N_COLORS = 8
N_RUNS = 6

# Data path
DATA_DIR = Path(__file__).resolve().parents[3] / "phase1_preprocess_decoding" / "results" / "full_dataset_C010"


# ============================================================================
# SRM Functions
# ============================================================================

def project_new_subject(srm_model, new_data):
    """Project a new subject into SRM space via SVD.

    Args:
        srm_model: Trained SRM model
        new_data: (n_voxels, n_colors) for new subject

    Returns:
        W_new: (n_voxels, k) projection matrix
    """
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def load_amplitudes(subject, roi):
    """Load amplitudes_procrustes.npy for a subject/ROI.

    Returns:
        data: (6, 8, n_voxels) array
    """
    roi_dir = ROI_DIR_MAP[roi]
    fpath = DATA_DIR / subject / roi_dir / "amplitudes_procrustes.npy"
    return np.load(fpath)


def train_hc_srm_and_align(roi, k):
    """Train SRM on HC only, align HC via weights, project CVD via SVD.

    Returns:
        hc_aligned: dict {subject: (n_runs, 8, k)} for HC
        cvd_aligned: dict {subject: (n_runs, 8, k)} for CVD
        hc_mean_aligned: dict {subject: (8, k)} mean across runs for HC
        cvd_mean_aligned: dict {subject: (8, k)} mean across runs for CVD
    """
    # Load all data
    hc_data = {}
    cvd_data = {}
    for s in HC_SUBJECTS:
        hc_data[s] = load_amplitudes(s, roi)  # (6, 8, n_voxels)
    for s in CVD_SUBJECTS:
        cvd_data[s] = load_amplitudes(s, roi)

    # HC mean betas for SRM training: (n_voxels, 8)
    hc_betas = []
    for s in HC_SUBJECTS:
        mean_beta = hc_data[s].mean(axis=0)  # (8, n_voxels)
        hc_betas.append(mean_beta.T)          # (n_voxels, 8) for BrainIAK

    # Train SRM on HC only
    srm = SRM(n_iter=10, features=k)
    srm.fit(hc_betas)

    # Align HC subjects using trained weights
    hc_aligned = {}
    hc_mean_aligned = {}
    for i, s in enumerate(HC_SUBJECTS):
        W_i = srm.w_[i]  # (n_voxels, k)
        aligned = hc_data[s] @ W_i  # (6, 8, k)
        hc_aligned[s] = aligned
        hc_mean_aligned[s] = aligned.mean(axis=0)  # (8, k)

    # Project CVD via SVD
    cvd_aligned = {}
    cvd_mean_aligned = {}
    for s in CVD_SUBJECTS:
        cvd_beta = cvd_data[s].mean(axis=0).T  # (n_voxels, 8)
        W_cvd = project_new_subject(srm, cvd_beta)  # (n_voxels, k)
        aligned = cvd_data[s] @ W_cvd  # (6, 8, k)
        cvd_aligned[s] = aligned
        cvd_mean_aligned[s] = aligned.mean(axis=0)  # (8, k)

    return hc_aligned, cvd_aligned, hc_mean_aligned, cvd_mean_aligned


# ============================================================================
# Cross-Decoding
# ============================================================================

def hc_loso_decoding(hc_mean_aligned, hc_subjects):
    """HC LOSO: train LDA on 6 HC, test on held-out HC.

    Uses mean-across-runs betas (8 samples per subject).
    """
    labels = np.arange(N_COLORS)
    per_subject_acc = {}

    for test_subj in hc_subjects:
        train_subjects = [s for s in hc_subjects if s != test_subj]

        X_train = np.vstack([hc_mean_aligned[s] for s in train_subjects])  # (48, k)
        y_train = np.tile(labels, len(train_subjects))

        X_test = hc_mean_aligned[test_subj]  # (8, k)
        y_test = labels

        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)

        per_subject_acc[test_subj] = float(accuracy_score(y_test, y_pred))

    return per_subject_acc


def hc_loso_decoding_allruns(hc_aligned, hc_subjects):
    """HC LOSO with all runs (48 train samples per subject, 8 test).

    Uses individual run betas for richer training signal.
    """
    labels = np.arange(N_COLORS)
    per_subject_acc = {}

    for test_subj in hc_subjects:
        train_subjects = [s for s in hc_subjects if s != test_subj]

        # Pool all runs from training subjects
        X_train = np.vstack([
            hc_aligned[s].reshape(-1, hc_aligned[s].shape[-1])  # (48, k)
            for s in train_subjects
        ])
        y_train = np.tile(labels, len(train_subjects) * N_RUNS)

        # Test on mean betas of held-out subject
        X_test = hc_aligned[test_subj].mean(axis=0)  # (8, k)
        y_test = labels

        lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)

        per_subject_acc[test_subj] = float(accuracy_score(y_test, y_pred))

    return per_subject_acc


def cvd_cross_decoding(hc_mean_aligned, cvd_mean_aligned, hc_subjects, cvd_subjects):
    """Train LDA on all 7 HC mean betas, test on each CVD."""
    labels = np.arange(N_COLORS)

    X_train = np.vstack([hc_mean_aligned[s] for s in hc_subjects])  # (56, k)
    y_train = np.tile(labels, len(hc_subjects))

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_train, y_train)

    per_subject_acc = {}
    for cvd_subj in cvd_subjects:
        X_test = cvd_mean_aligned[cvd_subj]  # (8, k)
        y_pred = lda.predict(X_test)
        per_subject_acc[cvd_subj] = float(accuracy_score(labels, y_pred))

    return per_subject_acc


def cvd_cross_decoding_allruns(hc_aligned, cvd_aligned, hc_subjects, cvd_subjects):
    """Train on all HC runs, test on CVD mean betas."""
    labels = np.arange(N_COLORS)

    X_train = np.vstack([
        hc_aligned[s].reshape(-1, hc_aligned[s].shape[-1])
        for s in hc_subjects
    ])
    y_train = np.tile(labels, len(hc_subjects) * N_RUNS)

    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_train, y_train)

    per_subject_acc = {}
    for cvd_subj in cvd_subjects:
        X_test = cvd_aligned[cvd_subj].mean(axis=0)  # (8, k)
        y_pred = lda.predict(X_test)
        per_subject_acc[cvd_subj] = float(accuracy_score(labels, y_pred))

    return per_subject_acc


def permutation_test(hc_mean_aligned, cvd_mean_aligned, hc_subjects, cvd_subject,
                     n_permutations=1000, seed=42):
    """Permutation test: shuffle training labels to build null distribution."""
    labels = np.arange(N_COLORS)

    X_train = np.vstack([hc_mean_aligned[s] for s in hc_subjects])  # (56, k)
    y_train = np.tile(labels, len(hc_subjects))
    X_test = cvd_mean_aligned[cvd_subject]  # (8, k)

    # Observed
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    lda.fit(X_train, y_train)
    observed_acc = float(accuracy_score(labels, lda.predict(X_test)))

    # Null distribution
    rng = np.random.RandomState(seed)
    null_accs = []
    for _ in range(n_permutations):
        y_perm = rng.permutation(y_train)
        lda_perm = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        lda_perm.fit(X_train, y_perm)
        null_acc = float(accuracy_score(labels, lda_perm.predict(X_test)))
        null_accs.append(null_acc)

    null_accs = np.array(null_accs)
    p_value = float((null_accs >= observed_acc).sum() / n_permutations)

    return observed_acc, p_value, float(null_accs.mean()), float(null_accs.std())


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="CVD cross-decoding with HC-only SRM (RT-7 fix)")
    parser.add_argument('--n_permutations', type=int, default=1000)
    parser.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'hV4'])
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent.parent / "results" / "cvd_cross_decoding"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CVD Cross-Decoding with HC-only SRM (RT-7 fix)")
    print(f"{'='*70}")
    print(f"Data: {DATA_DIR}")
    print(f"K values: {SRM_K}")
    print(f"Permutations: {args.n_permutations}")
    print(f"Output: {output_dir}")
    print(f"{'='*70}\n")

    all_results = {}

    for roi in args.rois:
        k = SRM_K[roi]
        print(f"\n{'='*50}")
        print(f"ROI: {roi} (k={k})")
        print(f"{'='*50}")

        # Step 1: Train HC-only SRM + align
        print("  Training HC-only SRM and aligning subjects...")
        hc_aligned, cvd_aligned, hc_mean, cvd_mean = train_hc_srm_and_align(roi, k)
        print(f"  HC aligned shape: {hc_aligned[HC_SUBJECTS[0]].shape}")
        print(f"  CVD aligned shape: {cvd_aligned[CVD_SUBJECTS[0]].shape}")

        # Step 2: HC LOSO baseline (mean betas)
        print("\n  HC LOSO decoding (mean betas)...")
        hc_loso_mean = hc_loso_decoding(hc_mean, HC_SUBJECTS)
        hc_mean_acc = np.mean(list(hc_loso_mean.values()))
        for s, a in hc_loso_mean.items():
            print(f"    {s}: {a:.3f}")
        print(f"    >> HC mean: {hc_mean_acc:.3f}")

        # Step 3: HC LOSO (all runs)
        print("\n  HC LOSO decoding (all runs)...")
        hc_loso_all = hc_loso_decoding_allruns(hc_aligned, HC_SUBJECTS)
        hc_all_acc = np.mean(list(hc_loso_all.values()))
        for s, a in hc_loso_all.items():
            print(f"    {s}: {a:.3f}")
        print(f"    >> HC mean: {hc_all_acc:.3f}")

        # Step 4: CVD cross-decoding (mean betas)
        print("\n  CVD cross-decoding (mean betas)...")
        cvd_cross_mean = cvd_cross_decoding(hc_mean, cvd_mean, HC_SUBJECTS, CVD_SUBJECTS)
        for s, a in cvd_cross_mean.items():
            print(f"    {s}: {a:.3f}")

        # Step 5: CVD cross-decoding (all runs)
        print("\n  CVD cross-decoding (all runs)...")
        cvd_cross_all = cvd_cross_decoding_allruns(hc_aligned, cvd_aligned, HC_SUBJECTS, CVD_SUBJECTS)
        for s, a in cvd_cross_all.items():
            print(f"    {s}: {a:.3f}")

        # Step 6: Permutation tests
        print(f"\n  Permutation tests ({args.n_permutations} perms)...")
        cvd_perm = {}
        for cvd_subj in CVD_SUBJECTS:
            obs_acc, p_val, null_mean, null_std = permutation_test(
                hc_mean, cvd_mean, HC_SUBJECTS, cvd_subj, args.n_permutations)
            cvd_perm[cvd_subj] = {
                'observed_acc': obs_acc,
                'p_value': p_val,
                'null_mean': null_mean,
                'null_std': null_std
            }
            sig = "*" if p_val < 0.05 else ""
            print(f"    {cvd_subj}: acc={obs_acc:.3f}, p={p_val:.4f} {sig} (null: {null_mean:.3f} +/- {null_std:.3f})")

        all_results[roi] = {
            'k': k,
            'hc_loso_mean_betas': hc_loso_mean,
            'hc_loso_allruns': hc_loso_all,
            'hc_mean_acc_mean_betas': float(hc_mean_acc),
            'hc_mean_acc_allruns': float(hc_all_acc),
            'cvd_cross_mean_betas': cvd_cross_mean,
            'cvd_cross_allruns': cvd_cross_all,
            'cvd_permutation': cvd_perm,
            'chance': float(1.0 / N_COLORS),
            'n_hc': len(HC_SUBJECTS),
            'n_cvd': len(CVD_SUBJECTS),
        }

    # Save results
    results_data = {
        'method': 'hc_only_srm_svd_projection',
        'description': 'HC-only SRM training, CVD projected via SVD (RT-7 fix)',
        'k_values': SRM_K,
        'n_permutations': args.n_permutations,
        'data_dir': str(DATA_DIR),
        'timestamp': datetime.now().isoformat(),
        'results': all_results,
    }

    output_file = output_dir / 'cvd_cross_decoding_hconly.json'
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'ROI':<6} {'k':>3} {'HC LOSO':>8} {'sub-08':>8} {'sub-09':>8} {'sub-10':>8}  (chance=0.125)")
    print("-" * 60)
    for roi in args.rois:
        if roi not in all_results:
            continue
        res = all_results[roi]
        row = f"{roi:<6} {res['k']:>3} {res['hc_mean_acc_mean_betas']:>8.3f}"
        for s in CVD_SUBJECTS:
            acc = res['cvd_cross_mean_betas'][s]
            p = res['cvd_permutation'][s]['p_value']
            sig = "*" if p < 0.05 else " "
            row += f" {acc:>7.3f}{sig}"
        print(row)

    print(f"\nResults saved: {output_file}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
