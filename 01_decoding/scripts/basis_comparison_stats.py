#!/usr/bin/env python3
"""
basis_comparison_stats.py — Paired t-tests for basis comparison results.

Compares each candidate basis against FE-6 baseline using paired t-tests (HC, n=7).
Also reports per-ROI optimal basis and interaction (ROI × basis).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/basis_comparison_stats.py  (development repository, commit 53c81c2)
# Role      : Statistics for the basis ablation.
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
import numpy as np
from scipy import stats
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parents[1] / 'results' / 'basis_comparison' / 'basis_comparison_full.json'

HC_SUBJECTS = [f'{i:02d}' for i in range(1, 8)]
CVD_SUBJECTS = [f'{i:02d}' for i in range(8, 11)]
ROIS = ['V1', 'V2', 'V3', 'V4']
BASES = ['FE-2', 'FE-3', 'FE-4', 'FE-6', 'FE-8', 'FE-12', 'LF-2', 'LF-4']
BASELINE = 'FE-6'


def extract_values(results, basis, roi, protocol, group_subjects):
    """Extract per-subject values for a given basis × ROI × protocol."""
    vals = []
    for subj in group_subjects:
        if subj in results[basis][roi][protocol]:
            vals.append(results[basis][roi][protocol][subj]['mean_voxel_corr'])
        else:
            vals.append(np.nan)
    return np.array(vals)


def main():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print('=' * 90)
    print('PAIRED T-TESTS: Each basis vs FE-6 (HC, n=7)')
    print('=' * 90)

    for protocol in ['loro', 'loco']:
        print(f'\n{"=" * 90}')
        print(f'  {protocol.upper()}')
        print(f'{"=" * 90}')

        # Get FE-6 baseline values
        for roi in ROIS:
            baseline_vals = extract_values(results, BASELINE, roi, protocol, HC_SUBJECTS)

            print(f'\n--- {roi} (FE-6 baseline: {np.nanmean(baseline_vals):.3f} ± {np.nanstd(baseline_vals):.3f}) ---')
            print(f'{"Basis":<8} {"Mean":>7} {"SD":>7} {"Δ":>7} {"t(6)":>7} {"p":>8} {"d":>7} {"Sig":>5}')

            for basis in BASES:
                vals = extract_values(results, basis, roi, protocol, HC_SUBJECTS)
                valid = ~np.isnan(vals) & ~np.isnan(baseline_vals)

                if valid.sum() < 3:
                    print(f'{basis:<8} {"N/A":>7}')
                    continue

                v = vals[valid]
                b = baseline_vals[valid]
                delta = v - b
                mean_delta = np.mean(delta)

                if basis == BASELINE:
                    print(f'{basis:<8} {np.mean(v):>7.3f} {np.std(v):>7.3f} {"—":>7} {"—":>7} {"—":>8} {"—":>7} {"base":>5}')
                    continue

                t_stat, p_val = stats.ttest_rel(v, b)
                cohens_d = mean_delta / np.std(delta) if np.std(delta) > 0 else 0
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else '†' if p_val < 0.1 else ''

                print(f'{basis:<8} {np.mean(v):>7.3f} {np.std(v):>7.3f} {mean_delta:>+7.3f} {t_stat:>7.2f} {p_val:>8.4f} {cohens_d:>+7.2f} {sig:>5}')

    # === Optimal basis per ROI ===
    print(f'\n\n{"=" * 90}')
    print('OPTIMAL BASIS PER ROI (HC LOCO)')
    print('=' * 90)

    fe_bases = ['FE-2', 'FE-3', 'FE-4', 'FE-6', 'FE-8', 'FE-12']

    for roi in ROIS:
        best_basis = None
        best_mean = -np.inf

        for basis in fe_bases:
            vals = extract_values(results, basis, roi, 'loco', HC_SUBJECTS)
            m = np.nanmean(vals)
            if m > best_mean:
                best_mean = m
                best_basis = basis

        baseline_vals = extract_values(results, BASELINE, roi, 'loco', HC_SUBJECTS)
        best_vals = extract_values(results, best_basis, roi, 'loco', HC_SUBJECTS)

        if best_basis != BASELINE:
            t_stat, p_val = stats.ttest_rel(best_vals[~np.isnan(best_vals)],
                                             baseline_vals[~np.isnan(baseline_vals)])
            print(f'{roi}: {best_basis} ({best_mean:.3f}) vs FE-6 ({np.nanmean(baseline_vals):.3f})  '
                  f'Δ={best_mean - np.nanmean(baseline_vals):+.3f}  t={t_stat:.2f}  p={p_val:.4f}')
        else:
            print(f'{roi}: FE-6 is already optimal ({best_mean:.3f})')

    # === CVD comparison with optimal bases ===
    print(f'\n\n{"=" * 90}')
    print('HC vs CVD LOCO (FE-6 vs optimal basis)')
    print('=' * 90)

    for roi in ROIS:
        # Find optimal FE basis for this ROI
        best_basis = None
        best_mean = -np.inf
        for basis in fe_bases:
            vals = extract_values(results, basis, roi, 'loco', HC_SUBJECTS)
            m = np.nanmean(vals)
            if m > best_mean:
                best_mean = m
                best_basis = basis

        for basis in [BASELINE, best_basis]:
            if basis == BASELINE and best_basis == BASELINE:
                hc = extract_values(results, basis, roi, 'loco', HC_SUBJECTS)
                cvd = extract_values(results, basis, roi, 'loco', CVD_SUBJECTS)
                hc_m, cvd_m = np.nanmean(hc), np.nanmean(cvd)
                pooled_sd = np.sqrt((np.nanvar(hc) * 6 + np.nanvar(cvd) * 2) / 8)
                d = (hc_m - cvd_m) / pooled_sd if pooled_sd > 0 else 0
                t_stat, p_val = stats.ttest_ind(hc[~np.isnan(hc)], cvd[~np.isnan(cvd)],
                                                 equal_var=False)
                print(f'{roi} ({basis}): HC={hc_m:+.3f}  CVD={cvd_m:+.3f}  d={d:+.2f}  p={p_val:.4f}')
                continue

            hc = extract_values(results, basis, roi, 'loco', HC_SUBJECTS)
            cvd = extract_values(results, basis, roi, 'loco', CVD_SUBJECTS)
            hc_m, cvd_m = np.nanmean(hc), np.nanmean(cvd)
            pooled_sd = np.sqrt((np.nanvar(hc) * 6 + np.nanvar(cvd) * 2) / 8)
            d = (hc_m - cvd_m) / pooled_sd if pooled_sd > 0 else 0
            t_stat, p_val = stats.ttest_ind(hc[~np.isnan(hc)], cvd[~np.isnan(cvd)],
                                             equal_var=False)
            label = f'{basis}' if basis != BASELINE else f'{basis} (baseline)'
            print(f'{roi} ({label}): HC={hc_m:+.3f}  CVD={cvd_m:+.3f}  d={d:+.2f}  p={p_val:.4f}')

    # === One-sample t-test: HC LOCO > 0 for optimal bases ===
    print(f'\n\n{"=" * 90}')
    print('ONE-SAMPLE T-TEST: HC LOCO > 0 (optimal basis per ROI)')
    print('=' * 90)

    for roi in ROIS:
        best_basis = None
        best_mean = -np.inf
        for basis in fe_bases:
            vals = extract_values(results, basis, roi, 'loco', HC_SUBJECTS)
            m = np.nanmean(vals)
            if m > best_mean:
                best_mean = m
                best_basis = basis

        for basis in [BASELINE, best_basis]:
            if basis == BASELINE and best_basis == BASELINE:
                continue  # skip duplicate

            vals = extract_values(results, basis, roi, 'loco', HC_SUBJECTS)
            v = vals[~np.isnan(vals)]
            t_stat, p_val = stats.ttest_1samp(v, 0)
            p_one = p_val / 2 if t_stat > 0 else 1 - p_val / 2
            ci = stats.t.interval(0.95, len(v)-1, loc=np.mean(v), scale=stats.sem(v))
            label = f'{basis}' if basis != BASELINE else f'{basis} (baseline)'
            sig = '*' if p_one < 0.05 else ''
            print(f'{roi} ({label}): M={np.mean(v):.3f}  95%CI=[{ci[0]:.3f}, {ci[1]:.3f}]  '
                  f't({len(v)-1})={t_stat:.3f}  p(one)={p_one:.4f} {sig}')


if __name__ == '__main__':
    main()
