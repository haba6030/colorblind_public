#!/usr/bin/env python3
"""Aggregate basis ablation results: FE-6 vs LF-4 vs LF-6."""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/aggregate_basis_ablation.py  (development repository, commit 53c81c2)
# Role      : Aggregates the basis ablation.
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

RESULTS_BASE = Path(__file__).parent.parent / 'results'

# Three conditions
CONDITIONS = {
    'FE-6': RESULTS_BASE / 'validation' / 'validation',  # nested from scp
    'LF-4': RESULTS_BASE / 'validation_lf4',
    'LF-6': RESULTS_BASE / 'validation_lf6',
}

# Fallback for FE-6 if no nested dir
if not CONDITIONS['FE-6'].exists():
    CONDITIONS['FE-6'] = RESULTS_BASE / 'validation'

SUBJECTS = [f'{i:02d}' for i in range(1, 11)]
HC = [f'{i:02d}' for i in range(1, 8)]
CVD = [f'{i:02d}' for i in range(8, 11)]
ROIS = ['V1', 'V2', 'V3', 'V4']
MODELS_COMMON = ['ols', 'ridge_gcv']  # available in all conditions


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_metrics(result_dir, subj, eval_type):
    """Extract key metrics from a subject's JSON."""
    path = result_dir / f'sub-{subj}_{eval_type}.json'
    if not path.exists():
        return None
    return load_json(path)


# ============================================================================
# LOCO comparison (main table)
# ============================================================================
print('=' * 90)
print('LOCO VOXEL CORRELATION: OLS')
print('=' * 90)
header = f'{"Subj":>5} {"Grp":>3}'
for roi in ROIS:
    header += f'  {"FE-6":>6} {"LF-4":>6} {"LF-6":>6}  |'
print(header)
print('-' * 90)

loco_data = {cond: {roi: [] for roi in ROIS} for cond in CONDITIONS}

for subj in SUBJECTS:
    grp = 'HC' if subj in HC else 'CVD'
    line = f'{subj:>5} {grp:>3}'
    for roi in ROIS:
        vals = []
        for cond_name, cond_dir in CONDITIONS.items():
            data = extract_metrics(cond_dir, subj, 'loco')
            if data and roi in data and 'ols' in data[roi]:
                v = data[roi]['ols']['mean_voxel_corr']
                vals.append(v)
                loco_data[cond_name][roi].append(v)
                line += f'  {v:>6.3f}'
            else:
                vals.append(None)
                line += f'  {"N/A":>6}'
        line += '  |'
    print(line)

# Group means
print('-' * 90)
line_mean = f'{"Mean":>5} {"":>3}'
for roi in ROIS:
    for cond_name in CONDITIONS:
        arr = loco_data[cond_name][roi]
        if arr:
            line_mean += f'  {np.mean(arr):>6.3f}'
        else:
            line_mean += f'  {"N/A":>6}'
    line_mean += '  |'
print(line_mean)

# HC-only means
line_hc = f'{"HC":>5} {"":>3}'
for roi in ROIS:
    for cond_name, cond_dir in CONDITIONS.items():
        vals = []
        for subj in HC:
            data = extract_metrics(cond_dir, subj, 'loco')
            if data and roi in data and 'ols' in data[roi]:
                vals.append(data[roi]['ols']['mean_voxel_corr'])
        if vals:
            line_hc += f'  {np.mean(vals):>6.3f}'
        else:
            line_hc += f'  {"N/A":>6}'
    line_hc += '  |'
print(line_hc)


# ============================================================================
# LOCO Ridge comparison
# ============================================================================
print('\n' + '=' * 90)
print('LOCO VOXEL CORRELATION: Ridge-GCV')
print('=' * 90)
print(header)
print('-' * 90)

loco_ridge = {cond: {roi: [] for roi in ROIS} for cond in CONDITIONS}

for subj in SUBJECTS:
    grp = 'HC' if subj in HC else 'CVD'
    line = f'{subj:>5} {grp:>3}'
    for roi in ROIS:
        for cond_name, cond_dir in CONDITIONS.items():
            data = extract_metrics(cond_dir, subj, 'loco')
            if data and roi in data and 'ridge_gcv' in data[roi]:
                v = data[roi]['ridge_gcv']['mean_voxel_corr']
                loco_ridge[cond_name][roi].append(v)
                line += f'  {v:>6.3f}'
            else:
                line += f'  {"N/A":>6}'
        line += '  |'
    print(line)

print('-' * 90)
line_mean = f'{"Mean":>5} {"":>3}'
for roi in ROIS:
    for cond_name in CONDITIONS:
        arr = loco_ridge[cond_name][roi]
        if arr:
            line_mean += f'  {np.mean(arr):>6.3f}'
        else:
            line_mean += f'  {"N/A":>6}'
    line_mean += '  |'
print(line_mean)


# ============================================================================
# LOCO MAE comparison
# ============================================================================
print('\n' + '=' * 90)
print('LOCO MAE (degrees): OLS')
print('=' * 90)
print(header)
print('-' * 90)

loco_mae = {cond: {roi: [] for roi in ROIS} for cond in CONDITIONS}

for subj in SUBJECTS:
    grp = 'HC' if subj in HC else 'CVD'
    line = f'{subj:>5} {grp:>3}'
    for roi in ROIS:
        for cond_name, cond_dir in CONDITIONS.items():
            data = extract_metrics(cond_dir, subj, 'loco')
            if data and roi in data and 'ols' in data[roi]:
                v = data[roi]['ols']['mean_mae']
                loco_mae[cond_name][roi].append(v)
                line += f'  {v:>6.1f}'
            else:
                line += f'  {"N/A":>6}'
        line += '  |'
    print(line)

print('-' * 90)
line_mean = f'{"Mean":>5} {"":>3}'
for roi in ROIS:
    for cond_name in CONDITIONS:
        arr = loco_mae[cond_name][roi]
        if arr:
            line_mean += f'  {np.mean(arr):>6.1f}'
        else:
            line_mean += f'  {"N/A":>6}'
    line_mean += '  |'
print(line_mean)


# ============================================================================
# LORO comparison (OLS only)
# ============================================================================
print('\n' + '=' * 90)
print('LORO VOXEL CORRELATION: OLS')
print('=' * 90)
print(header)
print('-' * 90)

loro_data = {cond: {roi: [] for roi in ROIS} for cond in CONDITIONS}

for subj in SUBJECTS:
    grp = 'HC' if subj in HC else 'CVD'
    line = f'{subj:>5} {grp:>3}'
    for roi in ROIS:
        for cond_name, cond_dir in CONDITIONS.items():
            data = extract_metrics(cond_dir, subj, 'loro')
            if data and roi in data and 'ols' in data[roi]:
                v = data[roi]['ols']['mean_voxel_corr']
                loro_data[cond_name][roi].append(v)
                line += f'  {v:>6.3f}'
            else:
                line += f'  {"N/A":>6}'
        line += '  |'
    print(line)

print('-' * 90)
line_mean = f'{"Mean":>5} {"":>3}'
for roi in ROIS:
    for cond_name in CONDITIONS:
        arr = loro_data[cond_name][roi]
        if arr:
            line_mean += f'  {np.mean(arr):>6.3f}'
        else:
            line_mean += f'  {"N/A":>6}'
    line_mean += '  |'
print(line_mean)


# ============================================================================
# FE-6 prior models (for reference)
# ============================================================================
print('\n' + '=' * 90)
print('FE-6 LOCO: ALL 4 MODELS (voxel_corr)')
print('=' * 90)
fe_dir = CONDITIONS['FE-6']
fe_models = ['ols', 'ridge_gcv', 'prior_only', 'prior_finetune']
header_fe = f'{"Subj":>5} {"Grp":>3}'
for roi in ROIS:
    header_fe += f'  {"OLS":>6} {"Ridge":>6} {"Prior":>6} {"Finet":>6}  |'
print(header_fe)
print('-' * 120)

fe_loco = {m: {roi: [] for roi in ROIS} for m in fe_models}

for subj in SUBJECTS:
    grp = 'HC' if subj in HC else 'CVD'
    line = f'{subj:>5} {grp:>3}'
    data = extract_metrics(fe_dir, subj, 'loco')
    for roi in ROIS:
        for m in fe_models:
            if data and roi in data and m in data[roi]:
                v = data[roi][m]['mean_voxel_corr']
                fe_loco[m][roi].append(v)
                line += f'  {v:>6.3f}'
            else:
                line += f'  {"N/A":>6}'
        line += '  |'
    print(line)

print('-' * 120)
line_mean = f'{"Mean":>5} {"":>3}'
for roi in ROIS:
    for m in fe_models:
        arr = fe_loco[m][roi]
        if arr:
            line_mean += f'  {np.mean(arr):>6.3f}'
        else:
            line_mean += f'  {"N/A":>6}'
    line_mean += '  |'
print(line_mean)


# ============================================================================
# Noise ceiling & reliability (from FE-6, sub-07 has corrected labels)
# ============================================================================
print('\n' + '=' * 70)
print('RELIABILITY (FE-6 data, noise ceiling labels corrected for sub-07 only)')
print('=' * 70)
header_rel = f'{"Subj":>5}'
for roi in ROIS:
    header_rel += f'  {"SH_RDM":>6} {"NC_lo":>6} {"NC_hi":>6}  |'
print(header_rel)
print('-' * 70)

for subj in SUBJECTS:
    data = extract_metrics(fe_dir, subj, 'reliability')
    line = f'{subj:>5}'
    for roi in ROIS:
        if data and roi in data:
            sh = data[roi].get('split_half_rdm', 0)
            # For sub-07 (new code), labels are correct
            # For others (old code), labels are SWAPPED — fix here
            if subj == '07':
                nc_lo = data[roi].get('noise_ceiling_lower', 0)
                nc_hi = data[roi].get('noise_ceiling_upper', 0)
            else:
                # Old code had upper/lower swapped
                nc_lo = data[roi].get('noise_ceiling_upper', 0)
                nc_hi = data[roi].get('noise_ceiling_lower', 0)
            line += f'  {sh:>6.3f} {nc_lo:>6.3f} {nc_hi:>6.3f}  |'
        else:
            line += f'  {"N/A":>6} {"N/A":>6} {"N/A":>6}  |'
    print(line)


# ============================================================================
# Summary: best basis per ROI
# ============================================================================
print('\n' + '=' * 70)
print('SUMMARY: Mean LOCO voxel_corr (OLS) by basis x ROI')
print('=' * 70)
print(f'{"":>8}', end='')
for roi in ROIS:
    print(f'  {roi:>8}', end='')
print()
for cond_name in CONDITIONS:
    print(f'{cond_name:>8}', end='')
    for roi in ROIS:
        arr = loco_data[cond_name][roi]
        if arr:
            print(f'  {np.mean(arr):>8.4f}', end='')
        else:
            print(f'  {"N/A":>8}', end='')
    print()

print(f'\n{"":>8}', end='')
for roi in ROIS:
    # Find winner
    best = None
    best_name = ''
    for cond_name in CONDITIONS:
        arr = loco_data[cond_name][roi]
        if arr:
            m = np.mean(arr)
            if best is None or m > best:
                best = m
                best_name = cond_name
    print(f'  {best_name:>8}', end='')
print('  <-- Winner')

print('\nSummary: Mean LOCO MAE (OLS) by basis x ROI')
print(f'{"":>8}', end='')
for roi in ROIS:
    print(f'  {roi:>8}', end='')
print()
for cond_name in CONDITIONS:
    print(f'{cond_name:>8}', end='')
    for roi in ROIS:
        arr = loco_mae[cond_name][roi]
        if arr:
            print(f'  {np.mean(arr):>8.1f}', end='')
        else:
            print(f'  {"N/A":>8}', end='')
    print()
