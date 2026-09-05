#!/usr/bin/env python3
"""Compute paper-ready statistics for forward model validation + basis ablation."""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/_compute_paper_stats.py  (development repository, commit 53c81c2)
# Role      : Single-case (Crawford-Howell) statistics for the decoding endpoints.
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
from pathlib import Path
from scipy import stats

RESULTS_BASE = Path(__file__).parent.parent / 'results'

# Directories
FE6_DIR = RESULTS_BASE / 'validation' / 'validation'
if not FE6_DIR.exists():
    FE6_DIR = RESULTS_BASE / 'validation'
LF4_DIR = RESULTS_BASE / 'validation_lf4'
LF6_DIR = RESULTS_BASE / 'validation_lf6'

HC = [f'{i:02d}' for i in range(1, 8)]
CVD = [f'{i:02d}' for i in range(8, 11)]
ALL = HC + CVD
ROIS = ['V1', 'V2', 'V3', 'V4']
ROI_LABEL = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def cohen_d(group1, group2):
    """Cohen's d (pooled SD)."""
    n1, n2 = len(group1), len(group2)
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (m1 - m2) / sp if sp > 0 else 0.0


def ci_95(arr):
    """95% CI via t-distribution."""
    n = len(arr)
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0
    t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0
    return m - t_crit * se, m + t_crit * se


def crawford_howell(x_i, control_group):
    """Crawford & Howell (1998) single-case test."""
    n = len(control_group)
    m = np.mean(control_group)
    s = np.std(control_group, ddof=1)
    if s == 0:
        return 0.0, 1.0
    t = (x_i - m) / (s * np.sqrt((n + 1) / n))
    p = 2 * stats.t.sf(abs(t), n - 1)
    return float(t), float(p)


# ============================================================================
# 1. FE-6 Complete Results (all 10 subjects, 4 models)
# ============================================================================
print('=' * 80)
print('SECTION 1: FE-6 COMPLETE VALIDATION (10 subjects, sub-07 re-run)')
print('=' * 80)

models_fe = ['ols', 'ridge_gcv', 'prior_only', 'prior_finetune']
eval_types = ['loro', 'loco']

for eval_type in eval_types:
    print(f'\n### {eval_type.upper()} — voxel_corr')
    for model in models_fe:
        print(f'\n  Model: {model}')
        for roi in ROIS:
            hc_vals, cvd_vals = [], []
            for subj in ALL:
                data = load_json(FE6_DIR / f'sub-{subj}_{eval_type}.json')
                if roi in data and model in data[roi]:
                    v = data[roi][model]['mean_voxel_corr']
                    if subj in HC:
                        hc_vals.append(v)
                    else:
                        cvd_vals.append(v)
            hc_m, hc_sd = np.mean(hc_vals), np.std(hc_vals, ddof=1)
            cvd_m, cvd_sd = np.mean(cvd_vals), np.std(cvd_vals, ddof=1)
            all_vals = hc_vals + cvd_vals
            all_m = np.mean(all_vals)
            d = cohen_d(hc_vals, cvd_vals)
            lo, hi = ci_95(all_vals)
            # t-test (unequal var)
            if len(hc_vals) > 1 and len(cvd_vals) > 1:
                t, p = stats.ttest_ind(hc_vals, cvd_vals, equal_var=False)
            else:
                t, p = 0, 1
            print(f'    {ROI_LABEL[roi]:>4}: HC={hc_m:+.3f}±{hc_sd:.3f}(n={len(hc_vals)})  '
                  f'CVD={cvd_m:+.3f}±{cvd_sd:.3f}(n={len(cvd_vals)})  '
                  f'All={all_m:+.3f} [{lo:+.3f},{hi:+.3f}]  '
                  f'd={d:+.2f}  p={p:.3f}')

# MAE for LOCO
print(f'\n### LOCO — MAE (degrees)')
for model in models_fe:
    print(f'\n  Model: {model}')
    for roi in ROIS:
        hc_vals, cvd_vals = [], []
        for subj in ALL:
            data = load_json(FE6_DIR / f'sub-{subj}_loco.json')
            if roi in data and model in data[roi]:
                v = data[roi][model]['mean_mae']
                if subj in HC:
                    hc_vals.append(v)
                else:
                    cvd_vals.append(v)
        hc_m, hc_sd = np.mean(hc_vals), np.std(hc_vals, ddof=1)
        cvd_m, cvd_sd = np.mean(cvd_vals), np.std(cvd_vals, ddof=1)
        all_m = np.mean(hc_vals + cvd_vals)
        lo, hi = ci_95(hc_vals + cvd_vals)
        print(f'    {ROI_LABEL[roi]:>4}: HC={hc_m:.1f}±{hc_sd:.1f}  '
              f'CVD={cvd_m:.1f}±{cvd_sd:.1f}  All={all_m:.1f} [{lo:.1f},{hi:.1f}]')


# ============================================================================
# 2. Reliability (corrected noise ceiling labels)
# ============================================================================
print('\n\n' + '=' * 80)
print('SECTION 2: RELIABILITY (noise ceiling corrected)')
print('=' * 80)

for roi in ROIS:
    hc_sh, cvd_sh = [], []
    hc_nc_lo, hc_nc_hi = [], []
    cvd_nc_lo, cvd_nc_hi = [], []
    for subj in ALL:
        data = load_json(FE6_DIR / f'sub-{subj}_reliability.json')
        if roi not in data:
            continue
        sh = data[roi]['split_half_rdm']
        # Fix swapped labels for old-code subjects (all except sub-07)
        if subj == '07':
            nc_lo = data[roi]['noise_ceiling_lower']
            nc_hi = data[roi]['noise_ceiling_upper']
        else:
            nc_lo = data[roi]['noise_ceiling_upper']  # swapped
            nc_hi = data[roi]['noise_ceiling_lower']  # swapped
        if subj in HC:
            hc_sh.append(sh)
            hc_nc_lo.append(nc_lo)
            hc_nc_hi.append(nc_hi)
        else:
            cvd_sh.append(sh)
            cvd_nc_lo.append(nc_lo)
            cvd_nc_hi.append(nc_hi)

    print(f'\n  {ROI_LABEL[roi]}:')
    print(f'    Split-half RDM:  HC={np.mean(hc_sh):.3f}±{np.std(hc_sh,ddof=1):.3f}  '
          f'CVD={np.mean(cvd_sh):.3f}±{np.std(cvd_sh,ddof=1):.3f}')
    print(f'    NC lower:        HC={np.mean(hc_nc_lo):.3f}±{np.std(hc_nc_lo,ddof=1):.3f}  '
          f'CVD={np.mean(cvd_nc_lo):.3f}±{np.std(cvd_nc_lo,ddof=1):.3f}')
    print(f'    NC upper:        HC={np.mean(hc_nc_hi):.3f}±{np.std(hc_nc_hi,ddof=1):.3f}  '
          f'CVD={np.mean(cvd_nc_hi):.3f}±{np.std(cvd_nc_hi,ddof=1):.3f}')


# ============================================================================
# 3. Basis Ablation Summary
# ============================================================================
print('\n\n' + '=' * 80)
print('SECTION 3: BASIS ABLATION (OLS + Ridge, 10 subjects)')
print('=' * 80)

conditions = {'FE-6': FE6_DIR, 'LF-4': LF4_DIR, 'LF-6': LF6_DIR}

for eval_type in ['loro', 'loco']:
    for metric_name in ['mean_voxel_corr', 'mean_mae']:
        if eval_type == 'loro' and metric_name == 'mean_mae':
            continue  # LORO doesn't have MAE
        print(f'\n### {eval_type.upper()} — {metric_name}')
        for model in ['ols', 'ridge_gcv']:
            print(f'\n  Model: {model}')
            for roi in ROIS:
                vals_by_cond = {}
                for cond_name, cond_dir in conditions.items():
                    all_vals = []
                    for subj in ALL:
                        path = cond_dir / f'sub-{subj}_{eval_type}.json'
                        if path.exists():
                            data = load_json(path)
                            if roi in data and model in data[roi]:
                                v = data[roi][model].get(metric_name)
                                if v is not None:
                                    all_vals.append(v)
                    vals_by_cond[cond_name] = all_vals

                line = f'    {ROI_LABEL[roi]:>4}:'
                for cond_name in conditions:
                    arr = vals_by_cond[cond_name]
                    if arr:
                        m = np.mean(arr)
                        sd = np.std(arr, ddof=1)
                        line += f'  {cond_name}={m:+.3f}±{sd:.3f}' if 'corr' in metric_name else f'  {cond_name}={m:.1f}±{sd:.1f}'
                    else:
                        line += f'  {cond_name}=N/A'

                # Paired t-test FE-6 vs LF-4 (same subjects)
                fe = vals_by_cond.get('FE-6', [])
                lf4 = vals_by_cond.get('LF-4', [])
                if len(fe) == len(lf4) == 10:
                    t, p = stats.ttest_rel(fe, lf4)
                    line += f'  (FE6 vs LF4: t={t:.2f}, p={p:.3f})'
                print(line)


# ============================================================================
# 4. Individual CVD Profiles (ridge_gcv — best model)
# ============================================================================
print('\n\n' + '=' * 80)
print('SECTION 4: INDIVIDUAL CVD PROFILES (ridge_gcv)')
print('=' * 80)

for subj in CVD:
    print(f'\n  sub-{subj}:')
    for roi in ROIS:
        # Get HC distribution for Crawford-Howell
        hc_loco = []
        for hc in HC:
            data = load_json(FE6_DIR / f'sub-{hc}_loco.json')
            if roi in data and 'ridge_gcv' in data[roi]:
                hc_loco.append(data[roi]['ridge_gcv']['mean_voxel_corr'])

        # CVD value
        data = load_json(FE6_DIR / f'sub-{subj}_loco.json')
        cvd_r = data[roi]['ridge_gcv']['mean_voxel_corr']
        cvd_mae = data[roi]['ridge_gcv']['mean_mae']

        # Crawford & Howell
        t_ch, p_ch = crawford_howell(cvd_r, hc_loco)
        z = (cvd_r - np.mean(hc_loco)) / np.std(hc_loco, ddof=1) if np.std(hc_loco, ddof=1) > 0 else 0

        # LORO for reference
        data_loro = load_json(FE6_DIR / f'sub-{subj}_loro.json')
        loro_r = data_loro[roi]['ridge_gcv']['mean_voxel_corr']

        print(f'    {ROI_LABEL[roi]:>4}: LOCO_r={cvd_r:+.3f}  MAE={cvd_mae:.1f}°  '
              f'LORO_r={loro_r:.3f}  z={z:+.2f}  CH_t={t_ch:+.2f}  CH_p={p_ch:.3f}')


# ============================================================================
# 5. Metric Appropriateness Notes
# ============================================================================
print('\n\n' + '=' * 80)
print('SECTION 5: METRIC APPROPRIATENESS NOTES')
print('=' * 80)

# Check correlation between n_voxels and LOCO performance
print('\n  Voxel count vs LOCO performance (ridge_gcv):')
voxel_counts = {
    'V1': {'01':568,'02':405,'03':858,'04':858,'05':858,'06':858,'07':330,
           '08':568,'09':568,'10':568},  # approx for CVD
    'V2': {'01':402,'02':335,'03':557,'04':557,'05':557,'06':557,'07':258,
           '08':402,'09':402,'10':402},
    'V3': {'01':106,'02':94,'03':115,'04':115,'05':115,'06':115,'07':59,
           '08':106,'09':106,'10':106},
    'V4': {'01':67,'02':69,'03':70,'04':70,'05':70,'06':70,'07':16,
           '08':67,'09':67,'10':67},
}

for roi in ROIS:
    nvox_list, loco_list = [], []
    for subj in HC:  # HC only (CVD voxels not in table)
        data = load_json(FE6_DIR / f'sub-{subj}_loco.json')
        if roi in data and 'ridge_gcv' in data[roi]:
            if subj in voxel_counts.get(roi, {}):
                nvox_list.append(voxel_counts[roi][subj])
                loco_list.append(data[roi]['ridge_gcv']['mean_voxel_corr'])
    if len(nvox_list) > 2:
        r, p = stats.pearsonr(nvox_list, loco_list)
        print(f'    {ROI_LABEL[roi]}: r(nvox, LOCO_r)={r:.3f}, p={p:.3f} (n={len(nvox_list)})')

# Noise-ceiling normalized LOCO
print('\n  Noise-ceiling normalized LOCO (ridge_gcv / NC_lower):')
for roi in ROIS:
    hc_norm = []
    for subj in HC:
        data_loco = load_json(FE6_DIR / f'sub-{subj}_loco.json')
        data_rel = load_json(FE6_DIR / f'sub-{subj}_reliability.json')
        if roi in data_loco and 'ridge_gcv' in data_loco[roi] and roi in data_rel:
            r = data_loco[roi]['ridge_gcv']['mean_voxel_corr']
            # Fix noise ceiling swap for old subjects
            if subj == '07':
                nc = data_rel[roi]['noise_ceiling_lower']
            else:
                nc = data_rel[roi]['noise_ceiling_upper']  # swapped → this is the real lower
            if nc > 0:
                hc_norm.append(r / nc)
    if hc_norm:
        print(f'    {ROI_LABEL[roi]}: HC mean normalized LOCO = {np.mean(hc_norm):.3f} ± {np.std(hc_norm,ddof=1):.3f}')

# One-sample t-test: LOCO voxel_corr > 0 (HC only, ridge_gcv)
print('\n  One-sample t-test: HC LOCO ridge_gcv > 0')
for roi in ROIS:
    hc_vals = []
    for subj in HC:
        data = load_json(FE6_DIR / f'sub-{subj}_loco.json')
        if roi in data and 'ridge_gcv' in data[roi]:
            hc_vals.append(data[roi]['ridge_gcv']['mean_voxel_corr'])
    if len(hc_vals) > 1:
        t, p = stats.ttest_1samp(hc_vals, 0)
        lo, hi = ci_95(hc_vals)
        print(f'    {ROI_LABEL[roi]}: M={np.mean(hc_vals):.3f}, t({len(hc_vals)-1})={t:.3f}, '
              f'p={p:.3f} (two-tail), p_one={p/2:.3f} (one-tail), '
              f'95%CI=[{lo:.3f},{hi:.3f}]')
