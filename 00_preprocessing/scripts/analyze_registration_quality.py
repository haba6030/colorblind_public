#!/usr/bin/env python3
"""
Comprehensive Registration Quality Analysis
- Analyze all subjects' metrics
- Identify outliers across the cohort
- Special focus on sub-07's discrepancy
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/analyze_registration_quality.py  (development repository, commit 53c81c2)
# Role      : ROI coverage = atlas ROI intersected with the BOLD brain mask; GLM valid-voxel ratio.
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
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(results_path):
    """Load results.json"""
    with open(results_path, 'r') as f:
        return json.load(f)

def extract_metrics_summary(results):
    """Extract key metrics for each subject-run-ROI combination"""
    rows = []

    for subject_id, subject_data in results['subjects'].items():
        for run_id, run_data in subject_data['runs'].items():
            run_num = int(run_id.split('-')[1])

            # Per-ROI metrics
            for roi_name, roi_metrics in run_data['per_roi_metrics'].items():
                row = {
                    'subject': subject_id,
                    'run': run_num,
                    'roi': roi_name,
                    'roi_voxels': roi_metrics['roi_voxels'],
                    'intersection_voxels': roi_metrics['intersection_voxels'],
                    'roi_coverage_ratio': roi_metrics['roi_coverage_ratio'],
                    'bold_in_roi_ratio': roi_metrics['bold_in_roi_ratio'],
                    'mean_signal_in_roi': roi_metrics['mean_signal_in_roi'],
                    'signal_ratio_roi_vs_brain': roi_metrics['signal_ratio_roi_vs_brain']
                }

                # Add GLM metrics if available
                if 'glm' in roi_metrics:
                    glm = roi_metrics['glm']
                    row.update({
                        'glm_n_roi_voxels': glm['n_roi_voxels'],
                        'glm_n_valid_voxels': glm['n_valid_voxels'],
                        'glm_valid_ratio': glm['valid_ratio'],
                        'glm_mean_amplitude': glm['mean_amplitude'],
                        'glm_std_amplitude': glm['std_amplitude'],
                        'glm_mean_variance_across_colors': glm['mean_variance_across_colors']
                    })

                rows.append(row)

    return pd.DataFrame(rows)

def compute_subject_statistics(df):
    """Compute mean statistics per subject across all runs and ROIs"""
    grouped = df.groupby('subject').agg({
        'roi_coverage_ratio': ['mean', 'std', 'min', 'max'],
        'bold_in_roi_ratio': ['mean', 'std'],
        'signal_ratio_roi_vs_brain': ['mean', 'std'],
        'glm_valid_ratio': ['mean', 'std', 'min', 'max'],
        'glm_mean_amplitude': ['mean', 'std'],
        'glm_mean_variance_across_colors': ['mean', 'std']
    }).round(4)

    return grouped

def identify_outlier_subjects(df, metric='roi_coverage_ratio', z_threshold=2):
    """Identify subjects that are outliers for a given metric"""
    subject_means = df.groupby('subject')[metric].mean()

    population_mean = subject_means.mean()
    population_std = subject_means.std()

    z_scores = (subject_means - population_mean) / population_std

    outliers = z_scores[abs(z_scores) > z_threshold].sort_values()

    return outliers, population_mean, population_std

def analyze_per_roi_quality(df):
    """Analyze registration quality per ROI across all subjects"""
    roi_stats = df.groupby('roi').agg({
        'roi_coverage_ratio': ['mean', 'std', 'min', 'max'],
        'glm_valid_ratio': ['mean', 'std', 'min', 'max'],
        'glm_mean_amplitude': ['mean', 'std']
    }).round(4)

    return roi_stats

def analyze_subject_consistency(df):
    """Check consistency across runs within each subject"""
    subject_run_variability = df.groupby('subject').agg({
        'roi_coverage_ratio': 'std',
        'glm_valid_ratio': 'std',
        'glm_mean_amplitude': 'std'
    }).round(4)

    subject_run_variability.columns = ['coverage_std', 'glm_valid_std', 'amplitude_std']

    return subject_run_variability

def analyze_specific_subject(df, subject_id='sub-07'):
    """Deep dive analysis for a specific subject"""
    subject_data = df[df['subject'] == subject_id].copy()
    other_data = df[df['subject'] != subject_id].copy()

    results = {
        'subject': subject_id,
        'n_runs': subject_data['run'].nunique(),
        'metrics': {}
    }

    # Compare key metrics
    key_metrics = ['roi_coverage_ratio', 'glm_valid_ratio', 'glm_mean_amplitude']

    for metric in key_metrics:
        subject_mean = subject_data[metric].mean()
        subject_std = subject_data[metric].std()

        pop_mean = other_data[metric].mean()
        pop_std = other_data[metric].std()

        z_score = (subject_mean - pop_mean) / pop_std if pop_std > 0 else 0

        results['metrics'][metric] = {
            'subject_mean': float(subject_mean),
            'subject_std': float(subject_std),
            'population_mean': float(pop_mean),
            'population_std': float(pop_std),
            'z_score': float(z_score),
            'is_outlier': abs(z_score) > 2
        }

    # Per-ROI breakdown
    results['per_roi'] = {}
    for roi in subject_data['roi'].unique():
        roi_data_subject = subject_data[subject_data['roi'] == roi]
        roi_data_pop = other_data[other_data['roi'] == roi]

        results['per_roi'][roi] = {
            'coverage_mean': float(roi_data_subject['roi_coverage_ratio'].mean()),
            'coverage_pop_mean': float(roi_data_pop['roi_coverage_ratio'].mean()),
            'glm_valid_mean': float(roi_data_subject['glm_valid_ratio'].mean()),
            'glm_valid_pop_mean': float(roi_data_pop['glm_valid_ratio'].mean())
        }

    return results

def main():
    results_path = Path('/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/colorBlind_analysis/analysis/prep_trials/results/Method_method3_header_mi/results.json')

    print("="*80)
    print("Comprehensive Registration Quality Analysis - Method3 Header MI")
    print("="*80)
    print()

    # Load results
    print("Loading results...")
    results = load_results(results_path)
    print(f"✅ Loaded data for {len(results['subjects'])} subjects")
    print()

    # Extract metrics
    print("Extracting metrics...")
    df = extract_metrics_summary(results)
    print(f"✅ Extracted {len(df)} subject-run-ROI combinations")
    print(f"   Subjects: {sorted(df['subject'].unique())}")
    print(f"   Runs per subject: {df.groupby('subject')['run'].nunique().unique()}")
    print(f"   ROIs: {sorted(df['roi'].unique())}")
    print()

    # ========================================================================
    # 1. OVERALL COHORT STATISTICS
    # ========================================================================
    print("="*80)
    print("1. OVERALL COHORT STATISTICS")
    print("="*80)
    print()

    overall_stats = df[['roi_coverage_ratio', 'glm_valid_ratio', 'glm_mean_amplitude',
                        'glm_mean_variance_across_colors']].describe()
    print(overall_stats.round(4))
    print()

    # ========================================================================
    # 2. SUBJECT-LEVEL STATISTICS
    # ========================================================================
    print("="*80)
    print("2. SUBJECT-LEVEL STATISTICS (Mean across all runs & ROIs)")
    print("="*80)
    print()

    subject_stats = compute_subject_statistics(df)
    print(subject_stats)
    print()

    # ========================================================================
    # 3. OUTLIER DETECTION
    # ========================================================================
    print("="*80)
    print("3. OUTLIER SUBJECTS (|Z-score| > 2)")
    print("="*80)
    print()

    metrics_to_check = ['roi_coverage_ratio', 'glm_valid_ratio', 'glm_mean_amplitude']
    outlier_summary = {}

    for metric in metrics_to_check:
        outliers, pop_mean, pop_std = identify_outlier_subjects(df, metric)

        print(f"\n{metric}:")
        print(f"  Population: mean={pop_mean:.4f}, std={pop_std:.4f}")

        if len(outliers) > 0:
            print(f"  Outliers detected:")
            for subject, z_score in outliers.items():
                subject_mean = df[df['subject'] == subject][metric].mean()
                print(f"    {subject}: mean={subject_mean:.4f}, z-score={z_score:.2f}")

                if subject not in outlier_summary:
                    outlier_summary[subject] = []
                outlier_summary[subject].append(metric)
        else:
            print(f"  No outliers detected ✅")

    print()

    # Summary of subjects with multiple outlier metrics
    if outlier_summary:
        print("Subjects with outlier metrics:")
        for subject, metrics in outlier_summary.items():
            print(f"  {subject}: {', '.join(metrics)}")
    else:
        print("✅ No subjects show outlier behavior")

    print()

    # ========================================================================
    # 4. PER-ROI QUALITY ANALYSIS
    # ========================================================================
    print("="*80)
    print("4. PER-ROI QUALITY ANALYSIS (All subjects)")
    print("="*80)
    print()

    roi_stats = analyze_per_roi_quality(df)
    print(roi_stats)
    print()

    # ========================================================================
    # 5. WITHIN-SUBJECT CONSISTENCY (across runs)
    # ========================================================================
    print("="*80)
    print("5. WITHIN-SUBJECT CONSISTENCY (Variability across runs)")
    print("="*80)
    print()

    subject_variability = analyze_subject_consistency(df)
    print(subject_variability)
    print()

    # Flag subjects with high variability
    high_variability_threshold = subject_variability['coverage_std'].mean() + 2*subject_variability['coverage_std'].std()
    high_var_subjects = subject_variability[subject_variability['coverage_std'] > high_variability_threshold]

    if len(high_var_subjects) > 0:
        print(f"⚠️  Subjects with high run-to-run variability:")
        print(high_var_subjects)
    else:
        print("✅ All subjects show consistent quality across runs")

    print()

    # ========================================================================
    # 6. SUB-07 SPECIFIC ANALYSIS
    # ========================================================================
    print("="*80)
    print("6. SUB-07 DETAILED ANALYSIS")
    print("="*80)
    print()

    sub07_analysis = analyze_specific_subject(df, 'sub-07')

    print(f"Subject: {sub07_analysis['subject']}")
    print(f"Runs analyzed: {sub07_analysis['n_runs']}")
    print()

    print("Metric Comparison vs Population:")
    print("-" * 80)
    for metric, stats in sub07_analysis['metrics'].items():
        print(f"\n{metric}:")
        print(f"  Sub-07:     {stats['subject_mean']:.4f} ± {stats['subject_std']:.4f}")
        print(f"  Population: {stats['population_mean']:.4f} ± {stats['population_std']:.4f}")
        print(f"  Z-score:    {stats['z_score']:.2f} {'⚠️ OUTLIER' if stats['is_outlier'] else '✅'}")

    print()
    print("Per-ROI Comparison:")
    print("-" * 80)
    for roi, stats in sub07_analysis['per_roi'].items():
        print(f"\n{roi}:")
        print(f"  Coverage: {stats['coverage_mean']:.4f} (pop: {stats['coverage_pop_mean']:.4f})")
        print(f"  GLM Valid: {stats['glm_valid_mean']:.4f} (pop: {stats['glm_valid_pop_mean']:.4f})")

    print()

    # ========================================================================
    # 7. DIAGNOSIS & RECOMMENDATIONS
    # ========================================================================
    print("="*80)
    print("7. DIAGNOSIS & RECOMMENDATIONS")
    print("="*80)
    print()

    # Check overall quality
    mean_coverage = df['roi_coverage_ratio'].mean()
    mean_glm_valid = df['glm_valid_ratio'].mean()
    mean_amplitude = df['glm_mean_amplitude'].mean()

    print("Overall Registration Quality:")
    print(f"  Mean ROI Coverage: {mean_coverage:.4f} {'✅ GOOD' if mean_coverage > 0.85 else '⚠️  LOW'}")
    print(f"  Mean GLM Valid Ratio: {mean_glm_valid:.4f} {'✅ EXCELLENT' if mean_glm_valid > 0.95 else '✅ GOOD' if mean_glm_valid > 0.85 else '⚠️  LOW'}")
    print(f"  Mean GLM Amplitude: {mean_amplitude:.2f}")
    print()

    # Sub-07 specific diagnosis
    print("Sub-07 Diagnosis:")
    sub07_metrics = sub07_analysis['metrics']

    is_outlier = any(m['is_outlier'] for m in sub07_metrics.values())

    if is_outlier:
        print("  ⚠️  Sub-07 shows statistically significant differences from population")
        print("  Affected metrics:")
        for metric, stats in sub07_metrics.items():
            if stats['is_outlier']:
                print(f"    - {metric}: z={stats['z_score']:.2f}")
    else:
        print("  ✅ Sub-07 metrics are within normal range")
        print()
        print("  The visualization discrepancy is likely due to:")
        print("    1. Automatic cut_coords selection by nilearn.plotting")
        print("    2. Individual brain anatomy variations (shifted visual cortex position)")
        print("    3. Atlas ROI center-of-mass may not align with BOLD's peak signal location")
        print()
        print("  Recommendation:")
        print("    - Registration quality is quantitatively good (GLM valid ratio near 1.0)")
        print("    - For better visualization, consider:")
        print("      a) Manual cut_coords specification based on ROI center-of-mass")
        print("      b) Use plot_stat_map with ROI mask overlay instead of plot_roi")
        print("      c) Multiple view angles (sagittal, coronal, axial)")

    print()

    # General recommendations
    if len(outlier_summary) > 0:
        print("General Recommendations:")
        print(f"  - {len(outlier_summary)} subjects show outlier behavior")
        print("  - Consider visual inspection of these subjects' alignment")
        print("  - May need subject-specific registration parameter tuning")
    else:
        print("✅ Overall quality is consistent across all subjects")

    print()
    print("="*80)

if __name__ == '__main__':
    main()
