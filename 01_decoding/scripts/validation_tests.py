#!/usr/bin/env python3
"""
Validation Tests for Decoder Models

Implements 4 critical validation tests:
1. Permutation Test (Section 6.2) - Label shuffle
2. Bootstrap CI (Section 6.4) - Confidence intervals
3. Test-Retest Reliability (Section 6.3) - Split-half correlation
4. Cross-Subject Generalization (Section 6.5) - HC→HC vs HC→CVD

Usage:
    python run_validation_tests.py \
        --baseline_dir /path/to/full_dataset_C010 \
        --performance_dir /path/to/results/{timestamp} \
        --output_dir /path/to/results/{timestamp}
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase3_decoder_comparing/model_comparison_validation/scripts/validation_tests.py  (development repository, commit 53c81c2)
# Role      : Cross-participant encoder transfer HC->HC vs HC->CVD in SRM space (Results section 1, S5). Output: cross_subject_generalization.json
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
import json
import argparse
from pathlib import Path
import numpy as np
import warnings
from datetime import datetime
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

# Add project root
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root / "analysis"))

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

HC_SUBJECTS = [f"{i:02d}" for i in range(1, 8)]
CVD_SUBJECTS = [f"{i:02d}" for i in range(8, 11)]
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS

ROIS = ['V1', 'V2', 'V3', 'V4']

N_PERMUTATIONS = 1000
N_BOOTSTRAP = 1000
N_SPLIT_HALF = 1000

HUE_ANGLES = [i * 45 for i in range(8)]  # 0, 45, ..., 315


# ============================================================================
# Helper Functions
# ============================================================================

def load_amplitudes(baseline_dir, subject, roi, alignment='raw'):
    """Load amplitudes from full_dataset_C010"""
    subject_roi_dir = Path(baseline_dir) / f"sub-{subject}" / roi

    if alignment == 'raw':
        amp_path = subject_roi_dir / "amplitudes_raw.npy"
    elif alignment == 'procrustes':
        amp_path = subject_roi_dir / "amplitudes_procrustes.npy"
    elif alignment == 'srm':
        amp_path = subject_roi_dir / "amplitudes_srm.npy"
    else:
        raise ValueError(f"Unknown alignment: {alignment}")

    if not amp_path.exists():
        raise FileNotFoundError(f"Amplitudes not found: {amp_path}")

    return np.load(amp_path)


def circular_diff_deg(hue1, hue2):
    """Compute circular difference in degrees"""
    diff = hue1 - hue2
    diff = np.mod(diff + 180, 360) - 180
    return diff


def labels_to_hue(labels):
    """Convert labels (0-7) to hue angles"""
    return np.array([HUE_ANGLES[int(l)] for l in labels])


def hue_to_labels(hue_angles):
    """Convert hue angles to nearest labels"""
    labels = []
    for hue in hue_angles:
        diffs = [abs(circular_diff_deg(hue, target_hue)) for target_hue in HUE_ANGLES]
        labels.append(np.argmin(diffs))
    return np.array(labels)


# ============================================================================
# 1. Permutation Test
# ============================================================================

def permutation_test_single(amplitudes, model_class, model_name, observed_acc,
                            n_permutations=1000):
    """
    Permutation test for a single subject-ROI-model

    Args:
        amplitudes: (n_runs, n_colors, n_voxels)
        model_class: Decoder class
        model_name: Model name
        observed_acc: Observed accuracy (from LORO CV)
        n_permutations: Number of permutations

    Returns:
        results: Dict with null distribution, p-value, z-score
    """
    n_runs, n_colors, n_voxels = amplitudes.shape
    labels = np.arange(n_colors)

    null_distribution = []

    # Determine if model uses labels or hue
    uses_labels = model_name in ['LDA', 'SVM', 'MLP', 'ForwardEncoding']

    for iteration in range(n_permutations):
        # Shuffle labels WITHIN each run (preserve temporal structure)
        shuffled_amplitudes = np.zeros_like(amplitudes)

        for run in range(n_runs):
            shuffled_labels = np.random.permutation(labels)
            shuffled_amplitudes[run] = amplitudes[run][shuffled_labels]

        # LORO CV with shuffled data
        fold_accs = []

        for test_run in range(n_runs):
            train_runs = [r for r in range(n_runs) if r != test_run]
            X_train = shuffled_amplitudes[train_runs].reshape(-1, n_voxels)
            X_test = shuffled_amplitudes[test_run]

            if uses_labels:
                y_train = np.tile(labels, len(train_runs))
                y_test = labels
            else:
                hue_angles = labels_to_hue(labels)
                y_train = np.tile(hue_angles, len(train_runs))
                y_test = hue_angles

            # Train and predict
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Compute accuracy
            if uses_labels:
                acc = np.mean(y_test == y_pred)
            else:
                y_pred_labels = hue_to_labels(y_pred)
                y_test_labels = hue_to_labels(y_test)
                acc = np.mean(y_test_labels == y_pred_labels)

            fold_accs.append(acc)

        null_acc = np.mean(fold_accs)
        null_distribution.append(null_acc)

    null_distribution = np.array(null_distribution)

    # Compute p-value
    p_value = (null_distribution >= observed_acc).sum() / n_permutations

    # Compute z-score
    null_mean = np.mean(null_distribution)
    null_std = np.std(null_distribution)
    z_score = (observed_acc - null_mean) / null_std if null_std > 0 else 0

    results = {
        'observed_acc': float(observed_acc),
        'null_mean': float(null_mean),
        'null_std': float(null_std),
        'null_distribution': null_distribution.tolist(),
        'p_value': float(p_value),
        'z_score': float(z_score),
        'n_permutations': n_permutations
    }

    return results


def run_permutation_test(baseline_dir, performance_dir, alignment='procrustes'):
    """
    Run permutation test for all subjects-ROIs-models

    Args:
        baseline_dir: Path to full_dataset_C010
        performance_dir: Path to performance results
        alignment: Which alignment to test

    Returns:
        all_results: Dict with permutation test results
    """
    print(f"\n{'='*80}")
    print(f"Permutation Test (Alignment: {alignment})")
    print(f"{'='*80}\n")

    # Load performance data to get observed accuracies
    performance_files = list(Path(performance_dir).glob("sub-*_performance_raw.json"))

    if len(performance_files) == 0:
        raise FileNotFoundError(f"No performance files found in {performance_dir}")

    all_results = {}

    for perf_file in performance_files:
        with open(perf_file, 'r') as f:
            perf_data = json.load(f)

        subject = perf_data['subject']
        models = perf_data['models']
        rois = perf_data['rois']

        print(f"Subject: sub-{subject}")

        all_results[subject] = {}

        for roi in rois:
            print(f"  ROI: {roi}")

            # Load amplitudes
            try:
                amplitudes = load_amplitudes(baseline_dir, subject, roi, alignment)
            except FileNotFoundError:
                print(f"    ERROR: Amplitudes not found")
                continue

            all_results[subject][roi] = {}

            for model_name in models:
                print(f"    Model: {model_name}...", end=' ', flush=True)

                # Get observed accuracy
                try:
                    fold_results = perf_data['results'][alignment][roi][model_name]
                    observed_acc = np.mean([f['acc_exact'] for f in fold_results])
                except KeyError:
                    print("SKIP (no performance data)")
                    continue

                # Import model class
                from loro_baseline import (
                    LDADecoder, RidgeDecoder, KernelRidgeDecoder,
                    SVMDecoder, MLPDecoder, ForwardEncodingDecoder
                )

                model_map = {
                    'LDA': LDADecoder,
                    'Ridge': RidgeDecoder,
                    'KernelRidge': KernelRidgeDecoder,
                    'SVM': SVMDecoder,
                    'MLP': MLPDecoder,
                    'ForwardEncoding': ForwardEncodingDecoder
                }

                if model_name not in model_map:
                    print("SKIP (unknown model)")
                    continue

                model_class = model_map[model_name]

                # Run permutation test
                try:
                    results = permutation_test_single(
                        amplitudes,
                        model_class,
                        model_name,
                        observed_acc,
                        n_permutations=N_PERMUTATIONS
                    )

                    all_results[subject][roi][model_name] = results

                    print(f"p={results['p_value']:.4f}, z={results['z_score']:.2f}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

    return all_results


# ============================================================================
# 2. Bootstrap Confidence Intervals
# ============================================================================

def bootstrap_ci_group(performance_data, alignment='procrustes', n_bootstrap=1000):
    """
    Bootstrap CI for group means (subject-level resampling)

    Args:
        performance_data: List of performance dicts
        alignment: Which alignment
        n_bootstrap: Number of bootstrap iterations

    Returns:
        ci_results: Dict with mean, CI_lower, CI_upper per model
    """
    print(f"\n{'='*80}")
    print(f"Bootstrap CI - Group Level (Alignment: {alignment})")
    print(f"{'='*80}\n")

    # Collect data per model
    model_data = {}  # model -> list of (subject, roi, accuracies)

    for perf_dict in performance_data:
        subject = perf_dict['subject']

        for roi in perf_dict['rois']:
            for model in perf_dict['models']:
                try:
                    fold_results = perf_dict['results'][alignment][roi][model]
                    accs = [f['acc_exact'] for f in fold_results]

                    if model not in model_data:
                        model_data[model] = []

                    model_data[model].append({
                        'subject': subject,
                        'roi': roi,
                        'accuracies': accs,
                        'mean_acc': np.mean(accs)
                    })

                except KeyError:
                    continue

    # Bootstrap for each model
    ci_results = {}

    for model, data_list in model_data.items():
        print(f"Model: {model}")

        # Bootstrap: resample subject-ROI pairs
        bootstrap_means = []

        for iteration in range(n_bootstrap):
            # Resample with replacement
            sampled_indices = np.random.choice(len(data_list), size=len(data_list), replace=True)
            sampled_data = [data_list[i] for i in sampled_indices]

            # Compute mean
            mean_acc = np.mean([d['mean_acc'] for d in sampled_data])
            bootstrap_means.append(mean_acc)

        bootstrap_means = np.array(bootstrap_means)

        # Compute CI
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        mean_acc = np.mean([d['mean_acc'] for d in data_list])

        ci_results[model] = {
            'mean': float(mean_acc),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'bootstrap_distribution': bootstrap_means.tolist()
        }

        print(f"  Mean: {mean_acc:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    return ci_results


# ============================================================================
# 3. Test-Retest Reliability
# ============================================================================

def split_half_reliability(performance_data, alignment='procrustes', n_iterations=1000):
    """
    Split-half reliability with Spearman-Brown correction

    Args:
        performance_data: List of performance dicts
        alignment: Which alignment
        n_iterations: Number of split-half iterations

    Returns:
        reliability_results: Dict with reliability per model
    """
    print(f"\n{'='*80}")
    print(f"Split-Half Reliability (Alignment: {alignment})")
    print(f"{'='*80}\n")

    # Collect fold-wise accuracies per model
    model_data = {}  # model -> list of (subject, roi, [6 fold accuracies])

    for perf_dict in performance_data:
        subject = perf_dict['subject']

        for roi in perf_dict['rois']:
            for model in perf_dict['models']:
                try:
                    fold_results = perf_dict['results'][alignment][roi][model]
                    fold_accs = [f['acc_exact'] for f in fold_results]

                    if len(fold_accs) != 6:
                        continue  # Skip if not 6 folds

                    if model not in model_data:
                        model_data[model] = []

                    model_data[model].append({
                        'subject': subject,
                        'roi': roi,
                        'fold_accuracies': fold_accs
                    })

                except KeyError:
                    continue

    # Compute split-half reliability for each model
    reliability_results = {}

    for model, data_list in model_data.items():
        print(f"Model: {model}")

        reliability_estimates = []

        for iteration in range(n_iterations):
            # For each subject-ROI, compute split-half
            scores_A = []
            scores_B = []

            for data in data_list:
                fold_accs = np.array(data['fold_accuracies'])

                # Random split into two halves
                indices = np.random.permutation(6)
                half_A = indices[:3]
                half_B = indices[3:]

                score_A = np.mean(fold_accs[half_A])
                score_B = np.mean(fold_accs[half_B])

                scores_A.append(score_A)
                scores_B.append(score_B)

            # Correlation across subjects
            if len(scores_A) > 1:
                r, _ = stats.spearmanr(scores_A, scores_B)

                # Spearman-Brown correction
                r_corrected = 2 * r / (1 + r) if (1 + r) > 0 else 0

                reliability_estimates.append(r_corrected)

        reliability_estimates = np.array(reliability_estimates)

        # Compute mean and CI
        mean_reliability = np.mean(reliability_estimates)
        ci_lower = np.percentile(reliability_estimates, 2.5)
        ci_upper = np.percentile(reliability_estimates, 97.5)

        reliability_results[model] = {
            'mean_reliability': float(mean_reliability),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'reliability_distribution': reliability_estimates.tolist()
        }

        print(f"  Reliability: {mean_reliability:.3f}, 95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

    return reliability_results


# ============================================================================
# 4. Cross-Subject Generalization
# ============================================================================

def cross_subject_generalization(baseline_dir, alignment='procrustes'):
    """
    Test cross-subject generalization: HC→HC vs HC→CVD

    Args:
        baseline_dir: Path to full_dataset_C010
        alignment: Which alignment

    Returns:
        generalization_results: Dict with HC→HC and HC→CVD accuracies
    """
    print(f"\n{'='*80}")
    print(f"Cross-Subject Generalization (Alignment: {alignment})")
    print(f"{'='*80}\n")

    from loro_baseline import (
        LDADecoder, RidgeDecoder, KernelRidgeDecoder,
        SVMDecoder, MLPDecoder, ForwardEncodingDecoder
    )

    model_map = {
        'LDA': LDADecoder,
        'Ridge': RidgeDecoder,
        'KernelRidge': KernelRidgeDecoder,
        'SVM': SVMDecoder,
        'MLP': MLPDecoder,
        'ForwardEncoding': ForwardEncodingDecoder
    }

    models = ['LDA', 'Ridge', 'SVM', 'MLP', 'ForwardEncoding']  # Skip KernelRidge (too slow)
    generalization_results = {}

    for model_name in models:
        print(f"\nModel: {model_name}")

        model_class = model_map[model_name]
        uses_labels = model_name in ['LDA', 'SVM', 'MLP', 'ForwardEncoding']

        # Test on each ROI
        hc_to_hc_scores = []
        hc_to_cvd_scores = []

        for roi in ROIS:
            print(f"  ROI: {roi}")

            # Load HC data
            hc_amplitudes = []
            for subject in HC_SUBJECTS:
                try:
                    amp = load_amplitudes(baseline_dir, subject, roi, alignment)
                    hc_amplitudes.append(amp)
                except FileNotFoundError:
                    continue

            if len(hc_amplitudes) < 3:
                print("    SKIP (not enough HC subjects)")
                continue

            # --- HC→HC (LOSO) ---
            for test_idx in range(len(hc_amplitudes)):
                train_indices = [i for i in range(len(hc_amplitudes)) if i != test_idx]

                # Pool training data
                X_train_list = []
                y_train_list = []

                labels = np.arange(8)
                hue_angles = labels_to_hue(labels)

                for train_idx in train_indices:
                    amp = hc_amplitudes[train_idx]
                    n_runs, n_colors, n_voxels = amp.shape

                    X_train_list.append(amp.reshape(-1, n_voxels))

                    if uses_labels:
                        y_train_list.append(np.tile(labels, n_runs))
                    else:
                        y_train_list.append(np.tile(hue_angles, n_runs))

                X_train = np.vstack(X_train_list)
                y_train = np.concatenate(y_train_list)

                # Test data
                test_amp = hc_amplitudes[test_idx]
                X_test = test_amp.reshape(-1, n_voxels)

                if uses_labels:
                    y_test = np.tile(labels, test_amp.shape[0])
                else:
                    y_test = np.tile(hue_angles, test_amp.shape[0])

                # Train and predict
                model = model_class()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Compute accuracy
                if uses_labels:
                    acc = np.mean(y_test == y_pred)
                else:
                    y_pred_labels = hue_to_labels(y_pred)
                    y_test_labels = hue_to_labels(y_test)
                    acc = np.mean(y_test_labels == y_pred_labels)

                hc_to_hc_scores.append(acc)

            # --- HC→CVD ---
            # Train on all HC
            X_train_list = []
            y_train_list = []

            for amp in hc_amplitudes:
                n_runs, n_colors, n_voxels = amp.shape
                X_train_list.append(amp.reshape(-1, n_voxels))

                if uses_labels:
                    y_train_list.append(np.tile(labels, n_runs))
                else:
                    y_train_list.append(np.tile(hue_angles, n_runs))

            X_train = np.vstack(X_train_list)
            y_train = np.concatenate(y_train_list)

            model = model_class()
            model.fit(X_train, y_train)

            # Test on CVD subjects
            for cvd_subject in CVD_SUBJECTS:
                try:
                    cvd_amp = load_amplitudes(baseline_dir, cvd_subject, roi, alignment)

                    X_test = cvd_amp.reshape(-1, n_voxels)

                    if uses_labels:
                        y_test = np.tile(labels, cvd_amp.shape[0])
                    else:
                        y_test = np.tile(hue_angles, cvd_amp.shape[0])

                    y_pred = model.predict(X_test)

                    if uses_labels:
                        acc = np.mean(y_test == y_pred)
                    else:
                        y_pred_labels = hue_to_labels(y_pred)
                        y_test_labels = hue_to_labels(y_test)
                        acc = np.mean(y_test_labels == y_pred_labels)

                    hc_to_cvd_scores.append(acc)

                except FileNotFoundError:
                    continue

        # Compute statistics
        hc_to_hc_mean = np.mean(hc_to_hc_scores)
        hc_to_cvd_mean = np.mean(hc_to_cvd_scores)

        difference = hc_to_hc_mean - hc_to_cvd_mean

        # Bootstrap difference test
        n_bootstrap = 1000
        boot_diffs = []

        for _ in range(n_bootstrap):
            sampled_hc = np.random.choice(hc_to_hc_scores, size=len(hc_to_hc_scores), replace=True)
            sampled_cvd = np.random.choice(hc_to_cvd_scores, size=len(hc_to_cvd_scores), replace=True)

            boot_diff = np.mean(sampled_hc) - np.mean(sampled_cvd)
            boot_diffs.append(boot_diff)

        boot_diffs = np.array(boot_diffs)
        ci_diff_lower = np.percentile(boot_diffs, 2.5)
        ci_diff_upper = np.percentile(boot_diffs, 97.5)

        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(hc_to_hc_scores, hc_to_cvd_scores, alternative='two-sided')

        generalization_results[model_name] = {
            'hc_to_hc': {
                'scores': hc_to_hc_scores,
                'mean': float(hc_to_hc_mean),
                'std': float(np.std(hc_to_hc_scores))
            },
            'hc_to_cvd': {
                'scores': hc_to_cvd_scores,
                'mean': float(hc_to_cvd_mean),
                'std': float(np.std(hc_to_cvd_scores))
            },
            'difference': {
                'mean': float(difference),
                'ci_lower': float(ci_diff_lower),
                'ci_upper': float(ci_diff_upper),
                'mann_whitney_u': float(u_stat),
                'p_value': float(p_value)
            }
        }

        print(f"  HC→HC: {hc_to_hc_mean:.3f} ± {np.std(hc_to_hc_scores):.3f}")
        print(f"  HC→CVD: {hc_to_cvd_mean:.3f} ± {np.std(hc_to_cvd_scores):.3f}")
        print(f"  Difference: {difference:+.3f}, 95% CI: [{ci_diff_lower:+.3f}, {ci_diff_upper:+.3f}], p={p_value:.4f}")

    return generalization_results


# ============================================================================
# 5. LDA Reliability Diagnostics (RT-5)
# ============================================================================

def compute_fold_cv(performance_data, alignment='procrustes'):
    """
    Analysis A: Within-subject fold-level coefficient of variation.

    For each subject-ROI-model: CV = std(fold_accs) / mean(fold_accs).
    High CV indicates unstable performance across folds.

    Args:
        performance_data: List of performance dicts
        alignment: Which alignment to analyze

    Returns:
        cv_results: Dict with CV per model, subject, ROI
    """
    print(f"\n{'='*80}")
    print(f"Fold-Level CV Analysis (Alignment: {alignment})")
    print(f"{'='*80}\n")

    cv_results = {}

    for perf_dict in performance_data:
        subject = perf_dict['subject']

        for roi in perf_dict['rois']:
            for model in perf_dict['models']:
                try:
                    fold_results = perf_dict['results'][alignment][roi][model]
                    fold_accs = [f['acc_exact'] for f in fold_results]

                    if len(fold_accs) != 6:
                        continue

                    mean_acc = np.mean(fold_accs)
                    std_acc = np.std(fold_accs)
                    cv = std_acc / mean_acc if mean_acc > 0 else float('inf')

                    if model not in cv_results:
                        cv_results[model] = []

                    cv_results[model].append({
                        'subject': subject,
                        'roi': roi,
                        'mean_acc': float(mean_acc),
                        'std_acc': float(std_acc),
                        'cv': float(cv),
                        'fold_accs': fold_accs
                    })

                except KeyError:
                    continue

    # Print summary per model
    summary = {}
    for model, entries in cv_results.items():
        cvs = [e['cv'] for e in entries]
        means = [e['mean_acc'] for e in entries]
        summary[model] = {
            'mean_cv': float(np.mean(cvs)),
            'std_cv': float(np.std(cvs)),
            'mean_acc': float(np.mean(means)),
            'n_entries': len(entries),
            'entries': entries
        }
        print(f"  {model}: CV={np.mean(cvs):.3f} +/- {np.std(cvs):.3f}, "
              f"mean_acc={np.mean(means):.3f}")

    return summary


def forward_encoding_weight_stability(baseline_dir, subjects, rois, alignment='procrustes'):
    """
    Analysis B: ForwardEncoding W matrix stability across LORO folds.

    Runs ForwardEncoding LORO independently to extract model.weights per fold.
    Computes pairwise cosine similarity across 6 folds (15 pairs from C(6,2)).

    Args:
        baseline_dir: Path to full_dataset_C010
        subjects: List of subject IDs
        rois: List of ROI names
        alignment: Which alignment

    Returns:
        stability_results: Dict with cosine similarity per subject-ROI
    """
    print(f"\n{'='*80}")
    print(f"ForwardEncoding W Stability (Alignment: {alignment})")
    print(f"{'='*80}\n")

    from loro_baseline import ForwardEncodingDecoder, load_amplitudes as mc_load_amplitudes

    stability_results = {}

    for subject in subjects:
        stability_results[subject] = {}

        for roi in rois:
            try:
                amplitudes = mc_load_amplitudes(baseline_dir, subject, roi, alignment)
            except FileNotFoundError:
                continue

            n_runs, n_colors, n_voxels = amplitudes.shape
            labels = np.arange(n_colors)

            # Collect W matrices per fold
            W_per_fold = []

            for test_run in range(n_runs):
                train_runs = [r for r in range(n_runs) if r != test_run]
                X_train = amplitudes[train_runs].reshape(-1, n_voxels)
                y_train = np.tile(labels, len(train_runs))

                model = ForwardEncodingDecoder(alpha=0, n_channels=6)
                model.fit(X_train, y_train)

                W_per_fold.append(model.weights.flatten())

            # Compute pairwise cosine similarity (15 pairs from 6 folds)
            W_matrix = np.array(W_per_fold)  # (6, n_channels * n_voxels)
            sim_matrix = cosine_similarity(W_matrix)

            # Extract upper triangle (15 pairs)
            triu_indices = np.triu_indices(n_runs, k=1)
            pairwise_sims = sim_matrix[triu_indices]

            stability_results[subject][roi] = {
                'mean_cosine_sim': float(np.mean(pairwise_sims)),
                'std_cosine_sim': float(np.std(pairwise_sims)),
                'min_cosine_sim': float(np.min(pairwise_sims)),
                'max_cosine_sim': float(np.max(pairwise_sims)),
                'pairwise_sims': pairwise_sims.tolist(),
                'n_pairs': len(pairwise_sims)
            }

            print(f"  sub-{subject} {roi}: cosine_sim = "
                  f"{np.mean(pairwise_sims):.3f} +/- {np.std(pairwise_sims):.3f}")

    return stability_results


def run_pair_reliability(performance_data, alignment='procrustes'):
    """
    Analysis C: Run-pair reliability.

    For all 15 run pairs (i,j): correlate acc_on_run_i vs acc_on_run_j
    across subject-ROIs. Reveals whether specific run combinations drive instability.

    Args:
        performance_data: List of performance dicts
        alignment: Which alignment

    Returns:
        pair_results: Dict with correlations per run pair per model
    """
    print(f"\n{'='*80}")
    print(f"Run-Pair Reliability (Alignment: {alignment})")
    print(f"{'='*80}\n")

    # Collect fold-wise accuracies per model
    model_data = {}  # model -> list of 6-element arrays

    for perf_dict in performance_data:
        subject = perf_dict['subject']

        for roi in perf_dict['rois']:
            for model in perf_dict['models']:
                try:
                    fold_results = perf_dict['results'][alignment][roi][model]
                    fold_accs = [f['acc_exact'] for f in fold_results]

                    if len(fold_accs) != 6:
                        continue

                    if model not in model_data:
                        model_data[model] = []

                    model_data[model].append({
                        'subject': subject,
                        'roi': roi,
                        'fold_accs': np.array(fold_accs)
                    })

                except KeyError:
                    continue

    pair_results = {}

    for model, data_list in model_data.items():
        if len(data_list) < 3:
            continue

        # For all 15 run pairs (i, j)
        n_runs = 6
        pair_correlations = {}

        for i in range(n_runs):
            for j in range(i + 1, n_runs):
                accs_i = [d['fold_accs'][i] for d in data_list]
                accs_j = [d['fold_accs'][j] for d in data_list]

                if np.std(accs_i) > 0 and np.std(accs_j) > 0:
                    r, p = stats.spearmanr(accs_i, accs_j)
                else:
                    r, p = 0.0, 1.0

                pair_correlations[f"run_{i}_vs_{j}"] = {
                    'r': float(r),
                    'p': float(p)
                }

        # Summary
        all_rs = [v['r'] for v in pair_correlations.values()]
        pair_results[model] = {
            'mean_r': float(np.mean(all_rs)),
            'std_r': float(np.std(all_rs)),
            'min_r': float(np.min(all_rs)),
            'max_r': float(np.max(all_rs)),
            'pairs': pair_correlations
        }

        print(f"  {model}: mean_r = {np.mean(all_rs):.3f} "
              f"[{np.min(all_rs):.3f}, {np.max(all_rs):.3f}]")

    return pair_results


def run_lda_reliability(baseline_dir, performance_data, alignment='procrustes'):
    """
    Run all 3 LDA reliability analyses (RT-5).

    Args:
        baseline_dir: Path to full_dataset_C010
        performance_data: List of performance dicts
        alignment: Which alignment

    Returns:
        results: Dict with all 3 analyses
    """
    print(f"\n{'='*80}")
    print(f"LDA Reliability Diagnostics (RT-5)")
    print(f"{'='*80}\n")

    results = {}

    # Analysis A: Fold-level CV
    results['fold_cv'] = compute_fold_cv(performance_data, alignment)

    # Analysis B: ForwardEncoding W stability
    subjects = ALL_SUBJECTS
    results['w_stability'] = forward_encoding_weight_stability(
        baseline_dir, subjects, ROIS, alignment)

    # Analysis C: Run-pair reliability
    results['run_pair_reliability'] = run_pair_reliability(performance_data, alignment)

    return results


# ============================================================================
# 6. LOCO Validation Tests
# ============================================================================

def load_loco_results(loco_dir):
    """
    Load sub-*_loco.json files from a LOCO results directory.

    Args:
        loco_dir: Path to directory containing sub-*_loco.json files

    Returns:
        loco_data: List of dicts (one per subject)
    """
    loco_path = Path(loco_dir)
    loco_files = sorted(loco_path.glob("sub-*_loco.json"))

    if len(loco_files) == 0:
        raise FileNotFoundError(f"No LOCO files found in {loco_dir}")

    loco_data = []
    for f in loco_files:
        with open(f, 'r') as fh:
            loco_data.append(json.load(fh))

    print(f"Loaded {len(loco_data)} LOCO result files from {loco_dir}")
    return loco_data


def loco_bootstrap_ci(loco_data, n_bootstrap=1000):
    """
    Bootstrap CI on overall_mae per model x ROI across subjects.

    Args:
        loco_data: List of LOCO result dicts
        n_bootstrap: Number of bootstrap iterations

    Returns:
        ci_results: Dict[roi][model] with mean, ci_lower, ci_upper
    """
    print(f"\n{'='*80}")
    print(f"LOCO Bootstrap CI (n_bootstrap={n_bootstrap})")
    print(f"{'='*80}\n")

    # Collect MAE values: {roi: {model: [mae_per_subject]}}
    mae_data = {}

    for subj_data in loco_data:
        results = subj_data['results']
        for roi in results:
            if roi not in mae_data:
                mae_data[roi] = {}
            for model in results[roi]:
                if model not in mae_data[roi]:
                    mae_data[roi][model] = []
                mae_data[roi][model].append(results[roi][model]['overall_mae'])

    ci_results = {}

    for roi in sorted(mae_data.keys()):
        ci_results[roi] = {}
        print(f"ROI: {roi}")

        for model in sorted(mae_data[roi].keys()):
            mae_values = np.array(mae_data[roi][model])
            n_subjects = len(mae_values)

            bootstrap_means = []
            for _ in range(n_bootstrap):
                sampled = np.random.choice(mae_values, size=n_subjects, replace=True)
                bootstrap_means.append(np.mean(sampled))

            bootstrap_means = np.array(bootstrap_means)
            ci_lower = np.percentile(bootstrap_means, 2.5)
            ci_upper = np.percentile(bootstrap_means, 97.5)
            mean_mae = np.mean(mae_values)

            ci_results[roi][model] = {
                'mean_mae': float(mean_mae),
                'std_mae': float(np.std(mae_values)),
                'ci_lower': float(ci_lower),
                'ci_upper': float(ci_upper),
                'n_subjects': n_subjects
            }

            print(f"  {model}: MAE={mean_mae:.1f}, 95% CI=[{ci_lower:.1f}, {ci_upper:.1f}]")

    return ci_results


def loco_group_comparison(loco_data):
    """
    HC vs CVD group comparison on MAE per model x ROI.
    Uses Mann-Whitney U test + bootstrap difference CI.

    Args:
        loco_data: List of LOCO result dicts

    Returns:
        group_results: Dict[roi][model] with HC/CVD means, U-stat, p-value
    """
    print(f"\n{'='*80}")
    print(f"LOCO Group Comparison (HC vs CVD)")
    print(f"{'='*80}\n")

    # Split by group
    hc_data = [d for d in loco_data if d['subject_group'] == 'HC']
    cvd_data = [d for d in loco_data if d['subject_group'] == 'CVD']

    print(f"HC: {len(hc_data)} subjects, CVD: {len(cvd_data)} subjects")

    group_results = {}

    # Get ROIs and models from first subject
    sample_results = loco_data[0]['results']
    rois = sorted(sample_results.keys())

    for roi in rois:
        group_results[roi] = {}
        models = sorted(sample_results[roi].keys())
        print(f"\nROI: {roi}")

        for model in models:
            hc_maes = [d['results'][roi][model]['overall_mae'] for d in hc_data
                       if roi in d['results'] and model in d['results'][roi]]
            cvd_maes = [d['results'][roi][model]['overall_mae'] for d in cvd_data
                        if roi in d['results'] and model in d['results'][roi]]

            if len(hc_maes) < 2 or len(cvd_maes) < 2:
                continue

            hc_mean = np.mean(hc_maes)
            cvd_mean = np.mean(cvd_maes)

            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(
                hc_maes, cvd_maes, alternative='two-sided')

            # Bootstrap difference CI
            n_boot = 1000
            boot_diffs = []
            for _ in range(n_boot):
                s_hc = np.random.choice(hc_maes, size=len(hc_maes), replace=True)
                s_cvd = np.random.choice(cvd_maes, size=len(cvd_maes), replace=True)
                boot_diffs.append(np.mean(s_cvd) - np.mean(s_hc))

            boot_diffs = np.array(boot_diffs)
            ci_lower = np.percentile(boot_diffs, 2.5)
            ci_upper = np.percentile(boot_diffs, 97.5)

            group_results[roi][model] = {
                'hc_mean': float(hc_mean),
                'hc_std': float(np.std(hc_maes)),
                'cvd_mean': float(cvd_mean),
                'cvd_std': float(np.std(cvd_maes)),
                'diff_cvd_minus_hc': float(cvd_mean - hc_mean),
                'diff_ci_lower': float(ci_lower),
                'diff_ci_upper': float(ci_upper),
                'mann_whitney_u': float(u_stat),
                'p_value': float(p_value),
                'n_hc': len(hc_maes),
                'n_cvd': len(cvd_maes)
            }

            sig = '*' if p_value < 0.05 else ''
            print(f"  {model}: HC={hc_mean:.1f}, CVD={cvd_mean:.1f}, "
                  f"diff={cvd_mean - hc_mean:+.1f}, p={p_value:.3f}{sig}")

    return group_results


def loco_permutation_summary(loco_data):
    """
    Aggregate pre-computed permutation p-values from LOCO JSONs.
    Count significant subjects per model x ROI.

    Args:
        loco_data: List of LOCO result dicts

    Returns:
        perm_results: Dict[roi][model] with subject-level p-values and summary
    """
    print(f"\n{'='*80}")
    print(f"LOCO Permutation Summary")
    print(f"{'='*80}\n")

    perm_results = {}

    sample_results = loco_data[0]['results']
    rois = sorted(sample_results.keys())

    for roi in rois:
        perm_results[roi] = {}
        models = sorted(sample_results[roi].keys())
        print(f"ROI: {roi}")

        for model in models:
            subject_pvals = []

            for subj_data in loco_data:
                subject = subj_data['subject']
                group = subj_data['subject_group']

                if roi not in subj_data['results']:
                    continue
                if model not in subj_data['results'][roi]:
                    continue

                perm = subj_data['results'][roi][model].get('permutation', {})
                p_val = perm.get('p_value', None)
                z_score = perm.get('z_score', None)

                if p_val is not None:
                    subject_pvals.append({
                        'subject': subject,
                        'group': group,
                        'p_value': float(p_val),
                        'z_score': float(z_score) if z_score is not None else None,
                        'significant_05': p_val < 0.05,
                        'significant_01': p_val < 0.01
                    })

            n_total = len(subject_pvals)
            n_sig_05 = sum(1 for s in subject_pvals if s['significant_05'])
            n_sig_01 = sum(1 for s in subject_pvals if s['significant_01'])

            # Separate HC and CVD
            hc_pvals = [s for s in subject_pvals if s['group'] == 'HC']
            cvd_pvals = [s for s in subject_pvals if s['group'] == 'CVD']

            perm_results[roi][model] = {
                'n_total': n_total,
                'n_significant_05': n_sig_05,
                'n_significant_01': n_sig_01,
                'proportion_sig_05': n_sig_05 / n_total if n_total > 0 else 0,
                'hc_significant_05': sum(1 for s in hc_pvals if s['significant_05']),
                'hc_total': len(hc_pvals),
                'cvd_significant_05': sum(1 for s in cvd_pvals if s['significant_05']),
                'cvd_total': len(cvd_pvals),
                'subjects': subject_pvals
            }

            print(f"  {model}: {n_sig_05}/{n_total} sig (p<.05), "
                  f"{n_sig_01}/{n_total} sig (p<.01), "
                  f"HC={sum(1 for s in hc_pvals if s['significant_05'])}/{len(hc_pvals)}, "
                  f"CVD={sum(1 for s in cvd_pvals if s['significant_05'])}/{len(cvd_pvals)}")

    return perm_results


# ============================================================================
# 7. Group Prior Validation Tests
# ============================================================================

def load_gp_results(gp_dir):
    """
    Load GP results from group_prior.py output.

    Format: {metadata: {...}, results: {roi: {sub-XX: {baseline, GP_nested/lambda_X}}}}
    Files: gp_*_results.json (all subjects) or gp_*_sub-XX.json (per-subject)

    Args:
        gp_dir: Path to GP results directory

    Returns:
        gp_data: Dict with 'results' (roi→subject→data) and 'metadata'
    """
    gp_path = Path(gp_dir)

    # Try aggregated file first (gp_loco_nested_results.json etc.)
    agg_files = sorted(gp_path.glob("gp_*_results.json"))
    if agg_files:
        with open(agg_files[0], 'r') as f:
            raw = json.load(f)
        print(f"Loaded GP results from {agg_files[0]}")
        return {
            'results': raw['results'],
            'metadata': raw.get('metadata', {})
        }

    # Try per-subject files (gp_loco_nested_sub-01.json etc.)
    subj_files = sorted(gp_path.glob("gp_*_sub-*.json"))
    if len(subj_files) == 0:
        raise FileNotFoundError(f"No GP result files (gp_*.json) found in {gp_dir}")

    # Merge per-subject files into unified results[roi][subject]
    merged_results = {}
    metadata = {}
    for f in subj_files:
        with open(f, 'r') as fh:
            data = json.load(fh)
        if not metadata:
            metadata = data.get('metadata', {})
        for roi, roi_data in data.get('results', {}).items():
            if roi not in merged_results:
                merged_results[roi] = {}
            merged_results[roi].update(roi_data)

    print(f"Loaded and merged {len(subj_files)} GP subject files from {gp_dir}")
    return {'results': merged_results, 'metadata': metadata}


def gp_improvement_test(gp_data):
    """
    Test whether Group Prior improves over baseline.
    Wilcoxon signed-rank test: GP MAE vs baseline MAE.
    Reports per-subject improvement (delta, %) per ROI.

    Expects group_prior.py output format:
        results[roi][sub-XX] = {baseline: float, GP_nested: {mean_mae, best_lambda}}
        or for fixed mode: {baseline: float, lambda_0.5: {mean_mae, fold_maes}}

    Args:
        gp_data: Dict from load_gp_results() with 'results' and 'metadata'

    Returns:
        gp_results: Dict[roi] with test results and per-subject details
    """
    print(f"\n{'='*80}")
    print(f"Group Prior Improvement Test")
    print(f"{'='*80}\n")

    results = gp_data['results']
    mode = gp_data.get('metadata', {}).get('mode', 'nested')
    gp_results = {}

    for roi in sorted(results.keys()):
        print(f"ROI: {roi}")

        baseline_vals = []
        gp_vals = []
        subject_details = []

        for subj_key, subj_data in sorted(results[roi].items()):
            # subj_key is "sub-01" etc.
            if 'baseline' not in subj_data:
                continue

            bl = subj_data['baseline']

            # Extract GP MAE based on mode
            gp = None
            best_lambda = None
            if 'GP_nested' in subj_data:
                gp = subj_data['GP_nested']['mean_mae']
                best_lambda = subj_data['GP_nested'].get('best_lambda')
            else:
                # Fixed mode: find best lambda key (lowest MAE)
                lambda_keys = [k for k in subj_data if k.startswith('lambda_')]
                if lambda_keys:
                    best_key = min(lambda_keys,
                                   key=lambda k: subj_data[k]['mean_mae'])
                    gp = subj_data[best_key]['mean_mae']
                    best_lambda = float(best_key.replace('lambda_', ''))

            if gp is None:
                continue

            # Determine group from subject ID
            subj_num = int(subj_key.replace('sub-', ''))
            group = 'HC' if subj_num <= 7 else 'CVD'

            baseline_vals.append(bl)
            gp_vals.append(gp)

            delta = gp - bl
            pct = (delta / bl * 100) if bl != 0 else 0

            subject_details.append({
                'subject': subj_key,
                'group': group,
                'baseline_mae': float(bl),
                'gp_mae': float(gp),
                'best_lambda': float(best_lambda) if best_lambda is not None else None,
                'delta': float(delta),
                'pct_change': float(pct)
            })

        baseline_arr = np.array(baseline_vals)
        gp_arr = np.array(gp_vals)

        # Wilcoxon signed-rank test (two-sided)
        if len(baseline_arr) >= 5:
            try:
                w_stat, w_pvalue = stats.wilcoxon(baseline_arr, gp_arr)
            except ValueError:
                w_stat, w_pvalue = np.nan, np.nan
        else:
            w_stat, w_pvalue = np.nan, np.nan

        # Split by group
        hc_details = [d for d in subject_details if d['group'] == 'HC']
        cvd_details = [d for d in subject_details if d['group'] == 'CVD']

        hc_improved = sum(1 for d in hc_details if d['delta'] < 0)
        cvd_improved = sum(1 for d in cvd_details if d['delta'] < 0)

        gp_results[roi] = {
            'baseline_mean': float(np.mean(baseline_arr)),
            'gp_mean': float(np.mean(gp_arr)),
            'mean_delta': float(np.mean(gp_arr - baseline_arr)),
            'wilcoxon_w': float(w_stat) if not np.isnan(w_stat) else None,
            'wilcoxon_p': float(w_pvalue) if not np.isnan(w_pvalue) else None,
            'n_subjects': len(baseline_arr),
            'n_improved': int(np.sum(gp_arr < baseline_arr)),
            'hc_improved': f"{hc_improved}/{len(hc_details)}",
            'cvd_improved': f"{cvd_improved}/{len(cvd_details)}",
            'subjects': subject_details
        }

        sig = '*' if (not np.isnan(w_pvalue) and w_pvalue < 0.05) else ''
        print(f"  Baseline: {np.mean(baseline_arr):.1f}, GP: {np.mean(gp_arr):.1f}, "
              f"delta={np.mean(gp_arr - baseline_arr):+.1f}, "
              f"Wilcoxon p={w_pvalue:.3f}{sig}")
        print(f"  Improved: {int(np.sum(gp_arr < baseline_arr))}/{len(baseline_arr)} "
              f"(HC: {hc_improved}/{len(hc_details)}, CVD: {cvd_improved}/{len(cvd_details)})")

    return gp_results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validation tests for decoder models"
    )
    parser.add_argument('--baseline_dir', type=str, required=True,
                       help='Path to full_dataset_C010')
    parser.add_argument('--performance_dir', type=str, required=True,
                       help='Directory with performance results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--alignment', type=str, default='procrustes',
                       choices=['raw', 'procrustes', 'srm'],
                       help='Which alignment to test')
    parser.add_argument('--loco_dir', type=str, default=None,
                       help='Directory with LOCO results (sub-*_loco.json)')
    parser.add_argument('--gp_dir', type=str, default=None,
                       help='Directory with GP results (fe_group_prior_results.json or sub-*_fe_group_prior.json)')
    parser.add_argument('--tests', nargs='+',
                       default=['permutation', 'bootstrap', 'reliability', 'generalization'],
                       help='Tests: permutation, bootstrap, reliability, generalization, '
                            'lda_reliability, loco_bootstrap, loco_group, loco_permutation, '
                            'gp_comparison')

    args = parser.parse_args()

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Validation Tests")
    print(f"{'='*80}")
    print(f"Baseline: {args.baseline_dir}")
    print(f"Performance: {args.performance_dir}")
    print(f"Output: {output_path}")
    print(f"Tests: {args.tests}")
    print(f"{'='*80}\n")

    # Load performance data
    performance_files = list(Path(args.performance_dir).glob("sub-*_performance_raw.json"))
    performance_data = []

    for perf_file in performance_files:
        with open(perf_file, 'r') as f:
            performance_data.append(json.load(f))

    print(f"Loaded {len(performance_data)} performance files")

    # Run tests
    results = {}

    if 'permutation' in args.tests:
        results['permutation_test'] = run_permutation_test(
            args.baseline_dir,
            args.performance_dir,
            args.alignment
        )

        # Save
        output_file = output_path / 'permutation_test.json'
        with open(output_file, 'w') as f:
            json.dump(results['permutation_test'], f, indent=2)
        print(f"\nSaved: {output_file}")

    if 'bootstrap' in args.tests:
        results['bootstrap_ci'] = bootstrap_ci_group(
            performance_data,
            args.alignment,
            N_BOOTSTRAP
        )

        # Save
        output_file = output_path / 'bootstrap_ci.json'
        with open(output_file, 'w') as f:
            json.dump(results['bootstrap_ci'], f, indent=2)
        print(f"\nSaved: {output_file}")

    if 'reliability' in args.tests:
        results['reliability'] = split_half_reliability(
            performance_data,
            args.alignment,
            N_SPLIT_HALF
        )

        # Save
        output_file = output_path / 'reliability.json'
        with open(output_file, 'w') as f:
            json.dump(results['reliability'], f, indent=2)
        print(f"\nSaved: {output_file}")

    if 'generalization' in args.tests:
        results['cross_subject_generalization'] = cross_subject_generalization(
            args.baseline_dir,
            args.alignment
        )

        # Save
        output_file = output_path / 'cross_subject_generalization.json'
        with open(output_file, 'w') as f:
            json.dump(results['cross_subject_generalization'], f, indent=2)
        print(f"\nSaved: {output_file}")

    if 'lda_reliability' in args.tests:
        results['lda_reliability'] = run_lda_reliability(
            args.baseline_dir,
            performance_data,
            args.alignment
        )

        # Save
        output_file = output_path / 'lda_reliability.json'
        with open(output_file, 'w') as f:
            json.dump(results['lda_reliability'], f, indent=2)
        print(f"\nSaved: {output_file}")

    # --- LOCO tests ---
    loco_tests = [t for t in args.tests if t.startswith('loco_')]
    if loco_tests:
        if args.loco_dir is None:
            print("\nERROR: --loco_dir required for LOCO tests")
        else:
            loco_data = load_loco_results(args.loco_dir)

            if 'loco_bootstrap' in args.tests:
                results['loco_bootstrap'] = loco_bootstrap_ci(loco_data)
                output_file = output_path / 'loco_bootstrap_ci.json'
                with open(output_file, 'w') as f:
                    json.dump(results['loco_bootstrap'], f, indent=2)
                print(f"\nSaved: {output_file}")

            if 'loco_group' in args.tests:
                results['loco_group'] = loco_group_comparison(loco_data)
                output_file = output_path / 'loco_group_comparison.json'
                with open(output_file, 'w') as f:
                    json.dump(results['loco_group'], f, indent=2)
                print(f"\nSaved: {output_file}")

            if 'loco_permutation' in args.tests:
                results['loco_permutation'] = loco_permutation_summary(loco_data)
                output_file = output_path / 'loco_permutation_summary.json'
                with open(output_file, 'w') as f:
                    json.dump(results['loco_permutation'], f, indent=2)
                print(f"\nSaved: {output_file}")

    # --- GP test ---
    if 'gp_comparison' in args.tests:
        if args.gp_dir is None:
            print("\nERROR: --gp_dir required for gp_comparison test")
        else:
            gp_data = load_gp_results(args.gp_dir)
            results['gp_comparison'] = gp_improvement_test(gp_data)
            output_file = output_path / 'gp_comparison.json'
            with open(output_file, 'w') as f:
                json.dump(results['gp_comparison'], f, indent=2)
            print(f"\nSaved: {output_file}")

    print(f"\n{'='*80}")
    print(f"All validation tests complete!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
