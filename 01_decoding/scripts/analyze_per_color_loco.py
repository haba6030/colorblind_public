#!/usr/bin/env python3
"""
analyze_per_color_loco.py — Per-Color LOCO Breakdown + CVD Vulnerability + Polar Plots.

Reads existing LOCO validation JSONs (ridge_gcv). No re-fitting needed.

Analyses:
  9f-2: Per-color breakdown — HC 7x8 matrix, per-color mean +/- SEM, Friedman test
  CVD vulnerability — Crawford-Howell per color against HC distribution
  Polar plots — 8-spoke polar of per-color LOCO_r, HC mean +/- SD band + CVD overlay

Usage:
    python analyze_per_color_loco.py \
        --validation_dir ../results/validation \
        --output_dir ../results/loco_reinforcement
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase4_forward_model/scripts/analyze_per_color_loco.py  (development repository, commit 53c81c2)
# Role      : Per-hue LOCO breakdown (hue vulnerability profile).
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
from scipy import stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ============================================================================
# Constants
# ============================================================================

HC_SUBJECTS = [f'{i:02d}' for i in range(1, 8)]
CVD_SUBJECTS = [f'{i:02d}' for i in range(8, 11)]
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS

ROIS = ['V1', 'V2', 'V3', 'V4']
ROI_DISPLAY = {'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'hV4'}

HUE_ANGLES = np.array([0, 45, 90, 135, 180, 225, 270, 315])
COLOR_NAMES = ['Red', 'Orange', 'Yellow', 'Green', 'Cyan', 'Blue', 'Purple', 'Magenta']

# CIELab → RGB for stimulus colors (inlined to avoid heavy imports)
COLOR_LAB = {
    0: [59.90, 62.69, 3.78],
    1: [64.20, 49.20, 45.58],
    2: [57.27, 13.06, 41.69],
    3: [69.08, -55.02, 47.38],
    4: [74.61, -41.33, -4.89],
    5: [69.14, -11.45, -40.91],
    6: [60.68, 19.18, -54.13],
    7: [60.17, 46.82, -40.31],
}


def lab2rgb(L, a, b):
    """CIELab to sRGB."""
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    xyz = np.array([x, y, z])
    xyz = np.where(xyz > 0.206893, xyz**3, (xyz - 16/116) / 7.787)
    xyz *= [0.95047, 1., 1.08883]
    rgb = np.dot([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]], xyz)
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)
    return tuple(np.clip(rgb, 0, 1))


def get_color_rgb(color_idx):
    """Get RGB for stimulus color by index (0-7)."""
    L, a, b = COLOR_LAB[color_idx]
    return lab2rgb(L, a, b)


# ============================================================================
# Stats helpers
# ============================================================================

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


def ci_95(arr):
    """95% CI via t-distribution."""
    n = len(arr)
    m = np.mean(arr)
    se = np.std(arr, ddof=1) / np.sqrt(n) if n > 1 else 0
    t_crit = stats.t.ppf(0.975, n - 1) if n > 1 else 0
    return m - t_crit * se, m + t_crit * se


# ============================================================================
# Data loading
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


def extract_per_color_voxel_corr(data, roi, model='ridge_gcv'):
    """Extract per-fold voxel_corr grouped by test_color.

    Returns:
        dict: {test_color_idx: voxel_corr_value}
    """
    if roi not in data or model not in data[roi]:
        return None
    folds = data[roi][model]['folds']
    result = {}
    for fold in folds:
        tc = fold['test_color']
        result[tc] = fold['voxel_corr']
    return result


# ============================================================================
# Analysis functions
# ============================================================================

def compute_per_color_breakdown(validation_dir):
    """9f-2: Per-color breakdown of LOCO voxel_corr for ridge_gcv.

    Returns:
        breakdown: dict with per-ROI stats
    """
    breakdown = {}

    for roi in ROIS:
        # HC matrix: 7 subjects x 8 colors
        hc_matrix = np.full((len(HC_SUBJECTS), 8), np.nan)
        cvd_matrix = np.full((len(CVD_SUBJECTS), 8), np.nan)

        for i, subj in enumerate(HC_SUBJECTS):
            try:
                path = find_loco_json(validation_dir, subj)
                data = json.loads(path.read_text())
                per_color = extract_per_color_voxel_corr(data, roi)
                if per_color:
                    for c in range(8):
                        hc_matrix[i, c] = per_color.get(c, np.nan)
            except FileNotFoundError:
                print(f'  Warning: missing sub-{subj}')

        for i, subj in enumerate(CVD_SUBJECTS):
            try:
                path = find_loco_json(validation_dir, subj)
                data = json.loads(path.read_text())
                per_color = extract_per_color_voxel_corr(data, roi)
                if per_color:
                    for c in range(8):
                        cvd_matrix[i, c] = per_color.get(c, np.nan)
            except FileNotFoundError:
                print(f'  Warning: missing sub-{subj}')

        # Per-color stats (HC)
        hc_mean = np.nanmean(hc_matrix, axis=0)  # (8,)
        hc_sem = np.nanstd(hc_matrix, axis=0, ddof=1) / np.sqrt(
            np.sum(~np.isnan(hc_matrix), axis=0))
        hc_sd = np.nanstd(hc_matrix, axis=0, ddof=1)

        # Friedman test (requires no NaN)
        valid_rows = ~np.any(np.isnan(hc_matrix), axis=1)
        hc_valid = hc_matrix[valid_rows]
        if hc_valid.shape[0] >= 3:
            fri_stat, fri_p = stats.friedmanchisquare(
                *[hc_valid[:, c] for c in range(8)])
        else:
            fri_stat, fri_p = np.nan, np.nan

        # Overall HC mean (should match known LOCO mean)
        overall_hc_mean = np.nanmean(hc_matrix)

        # CVD per-color
        cvd_per_color = {}
        for i, subj in enumerate(CVD_SUBJECTS):
            subj_vals = cvd_matrix[i]
            per_color_ch = {}
            for c in range(8):
                if np.isnan(subj_vals[c]) or np.any(np.isnan(hc_matrix[:, c])):
                    continue
                hc_col = hc_matrix[~np.isnan(hc_matrix[:, c]), c]
                t_ch, p_ch = crawford_howell(subj_vals[c], hc_col)
                per_color_ch[c] = {
                    'value': float(subj_vals[c]),
                    'hc_mean': float(np.mean(hc_col)),
                    'hc_sd': float(np.std(hc_col, ddof=1)),
                    't': t_ch,
                    'p': p_ch,
                    'significant': p_ch < 0.05,
                }
            cvd_per_color[f'sub-{subj}'] = per_color_ch

        breakdown[ROI_DISPLAY[roi]] = {
            'hc_matrix': hc_matrix.tolist(),
            'hc_per_color_mean': hc_mean.tolist(),
            'hc_per_color_sem': hc_sem.tolist(),
            'hc_per_color_sd': hc_sd.tolist(),
            'overall_hc_mean': float(overall_hc_mean),
            'friedman_chi2': float(fri_stat) if np.isfinite(fri_stat) else None,
            'friedman_p': float(fri_p) if np.isfinite(fri_p) else None,
            'cvd_per_color': cvd_per_color,
        }

    return breakdown


# ============================================================================
# Plotting functions
# ============================================================================

def plot_per_color_bars(breakdown, roi_display, output_dir):
    """Bar plot of per-color HC LOCO voxel_corr with CVD dots."""
    info = breakdown[roi_display]
    hc_mean = np.array(info['hc_per_color_mean'])
    hc_sem = np.array(info['hc_per_color_sem'])
    colors_rgb = [get_color_rgb(c) for c in range(8)]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(8)
    bars = ax.bar(x, hc_mean, yerr=hc_sem, capsize=4,
                  color=colors_rgb, edgecolor='k', linewidth=0.8, alpha=0.85,
                  zorder=3)

    # CVD individual dots
    cvd_markers = {'sub-08': 's', 'sub-09': '^', 'sub-10': 'D'}
    cvd_labels_added = set()
    for subj_key, per_color_ch in info['cvd_per_color'].items():
        marker = cvd_markers.get(subj_key, 'o')
        for c_str, ch_info in per_color_ch.items():
            c = int(c_str)
            label = subj_key if subj_key not in cvd_labels_added else None
            edge = 'red' if ch_info['significant'] else 'gray'
            ax.scatter(c, ch_info['value'], marker=marker, s=60,
                       edgecolor=edge, facecolor='white', linewidths=1.5,
                       zorder=5, label=label)
            cvd_labels_added.add(subj_key)

    ax.axhline(0, color='k', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{COLOR_NAMES[c]}\n({HUE_ANGLES[c]}°)' for c in range(8)],
                       fontsize=9)
    ax.set_ylabel('LOCO voxel correlation (r)', fontsize=11)
    ax.set_title(f'{roi_display} — Per-Color LOCO Breakdown (ridge_gcv)\n'
                 f'HC mean = {info["overall_hc_mean"]:.3f} | '
                 f'Friedman χ²={info["friedman_chi2"]:.2f}, p={info["friedman_p"]:.3f}'
                 if info['friedman_chi2'] is not None else
                 f'{roi_display} — Per-Color LOCO Breakdown (ridge_gcv)',
                 fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out_path = output_dir / f'per_color_{roi_display}.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_cvd_vulnerability_table(breakdown, output_dir):
    """Summary table figure of CVD per-color Crawford-Howell results."""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('off')

    # Build table data
    col_labels = [f'{COLOR_NAMES[c]}\n({HUE_ANGLES[c]}°)' for c in range(8)]
    row_labels = []
    cell_text = []
    cell_colors_arr = []

    for roi_display in [ROI_DISPLAY[r] for r in ROIS]:
        info = breakdown.get(roi_display)
        if not info:
            continue
        for subj_key in ['sub-08', 'sub-09', 'sub-10']:
            per_color = info['cvd_per_color'].get(subj_key, {})
            row_labels.append(f'{subj_key} / {roi_display}')
            row_text = []
            row_colors = []
            for c in range(8):
                ch = per_color.get(c) or per_color.get(str(c))
                if ch:
                    txt = f'{ch["value"]:.3f}\np={ch["p"]:.3f}'
                    if ch['significant']:
                        row_colors.append('#ffcccc')
                    else:
                        row_colors.append('white')
                else:
                    txt = 'N/A'
                    row_colors.append('#f0f0f0')
                row_text.append(txt)
            cell_text.append(row_text)
            cell_colors_arr.append(row_colors)

    table = ax.table(cellText=cell_text, rowLabels=row_labels,
                     colLabels=col_labels, loc='center',
                     cellColours=cell_colors_arr)
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    ax.set_title('CVD Per-Color Vulnerability (Crawford-Howell vs HC)\n'
                 'Red cells: p < 0.05', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    out_path = output_dir / 'cvd_vulnerability_table.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_polar_summary(breakdown, roi_display, validation_dir, output_dir):
    """8-spoke polar plot: HC mean +/- SD band + CVD overlay."""
    info = breakdown[roi_display]
    hc_matrix = np.array(info['hc_matrix'])
    hc_mean = np.array(info['hc_per_color_mean'])
    hc_sd = np.array(info['hc_per_color_sd'])

    # Angles for polar (convert hue to radians)
    angles = np.deg2rad(HUE_ANGLES)
    # Close the polygon
    angles_closed = np.concatenate([angles, [angles[0]]])
    hc_mean_closed = np.concatenate([hc_mean, [hc_mean[0]]])
    hc_upper = np.concatenate([hc_mean + hc_sd, [hc_mean[0] + hc_sd[0]]])
    hc_lower = np.concatenate([hc_mean - hc_sd, [hc_mean[0] - hc_sd[0]]])

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location('E')
    ax.set_theta_direction(1)

    # HC band
    ax.fill_between(angles_closed, hc_lower, hc_upper, alpha=0.2,
                     color='steelblue', label='HC mean +/- SD')
    ax.plot(angles_closed, hc_mean_closed, 'o-', color='steelblue',
            linewidth=2, markersize=6, label='HC mean')

    # CVD overlays
    cvd_colors = {'sub-08': '#e74c3c', 'sub-09': '#f39c12', 'sub-10': '#8e44ad'}
    cvd_styles = {'sub-08': '--', 'sub-09': '-.', 'sub-10': ':'}
    for subj in CVD_SUBJECTS:
        subj_key = f'sub-{subj}'
        try:
            path = find_loco_json(validation_dir, subj)
            data = json.loads(path.read_text())
            # Find the matching ROI key
            roi_key = None
            for r in ROIS:
                if ROI_DISPLAY[r] == roi_display:
                    roi_key = r
                    break
            per_color = extract_per_color_voxel_corr(data, roi_key)
            if per_color:
                vals = [per_color.get(c, np.nan) for c in range(8)]
                vals_closed = vals + [vals[0]]
                ax.plot(angles_closed, vals_closed,
                        linestyle=cvd_styles[subj_key],
                        color=cvd_colors[subj_key],
                        linewidth=1.8, marker='o', markersize=5,
                        label=subj_key)
        except FileNotFoundError:
            pass

    # Zero line
    ax.plot(angles_closed, np.zeros(len(angles_closed)), 'k--', alpha=0.3,
            linewidth=0.5)

    # Spoke labels
    ax.set_xticks(angles)
    ax.set_xticklabels([f'{COLOR_NAMES[c]}\n({HUE_ANGLES[c]}°)' for c in range(8)],
                       fontsize=8)

    ax.set_title(f'{roi_display} — Per-Color LOCO Pattern Correlation',
                 fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    out_path = output_dir / f'polar_{roi_display}_summary.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {out_path}')


def plot_polar_individual(validation_dir, output_dir):
    """Individual polar plots: 10 subjects x 4 ROIs."""
    indiv_dir = output_dir / 'polar_individual'
    indiv_dir.mkdir(exist_ok=True)

    for subj in ALL_SUBJECTS:
        try:
            path = find_loco_json(validation_dir, subj)
            data = json.loads(path.read_text())
        except FileNotFoundError:
            continue

        for roi in ROIS:
            per_color = extract_per_color_voxel_corr(data, roi)
            if not per_color:
                continue

            angles = np.deg2rad(HUE_ANGLES)
            vals = [per_color.get(c, 0) for c in range(8)]
            angles_closed = np.concatenate([angles, [angles[0]]])
            vals_closed = vals + [vals[0]]
            colors_rgb = [get_color_rgb(c) for c in range(8)]

            fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(projection='polar'))
            ax.set_theta_zero_location('E')
            ax.set_theta_direction(1)

            ax.plot(angles_closed, vals_closed, 'o-', color='gray',
                    linewidth=1.5, markersize=4)
            for c in range(8):
                ax.scatter(angles[c], vals[c], color=colors_rgb[c], s=60,
                           edgecolor='k', linewidth=0.5, zorder=5)

            ax.plot(angles_closed, np.zeros(len(angles_closed)), 'k--',
                    alpha=0.3, linewidth=0.5)

            group = 'HC' if subj in HC_SUBJECTS else 'CVD'
            roi_disp = ROI_DISPLAY[roi]
            ax.set_title(f'sub-{subj} ({group}) — {roi_disp}\n'
                         f'Mean r = {np.mean(vals):.3f}',
                         fontsize=11, fontweight='bold', pad=15)
            ax.set_xticks(angles)
            ax.set_xticklabels([f'{HUE_ANGLES[c]}°' for c in range(8)], fontsize=7)

            plt.tight_layout()
            fig_path = indiv_dir / f'polar_sub-{subj}_{roi_disp}.png'
            plt.savefig(fig_path, dpi=120, bbox_inches='tight')
            plt.close()

    print(f'  Saved individual polars to: {indiv_dir}')


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Per-Color LOCO analysis')
    parser.add_argument('--validation_dir', type=str, required=True,
                        help='Path to validation results directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for results')
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Per-Color LOCO Analysis (ridge_gcv)')
    print('=' * 60)

    # ---- 9f-2: Per-Color Breakdown ----
    print('\n[1/4] Computing per-color breakdown...')
    breakdown = compute_per_color_breakdown(validation_dir)

    # Save JSON
    with open(output_dir / 'per_color_breakdown.json', 'w') as f:
        json.dump(breakdown, f, indent=2)
    print(f'  Saved: {output_dir / "per_color_breakdown.json"}')

    # Print summary
    for roi_disp in [ROI_DISPLAY[r] for r in ROIS]:
        info = breakdown.get(roi_disp)
        if not info:
            continue
        print(f'\n  {roi_disp}: HC mean={info["overall_hc_mean"]:.3f}')
        print(f'    Per-color HC means: {[f"{v:.3f}" for v in info["hc_per_color_mean"]]}')
        if info['friedman_chi2'] is not None:
            print(f'    Friedman chi2={info["friedman_chi2"]:.2f}, p={info["friedman_p"]:.3f}')

    # ---- Plots ----
    print('\n[2/4] Generating per-color bar plots...')
    for roi in ROIS:
        roi_disp = ROI_DISPLAY[roi]
        if roi_disp in breakdown:
            plot_per_color_bars(breakdown, roi_disp, output_dir)

    # ---- CVD Vulnerability ----
    print('\n[3/4] Generating CVD vulnerability table...')
    # Save CVD vulnerability JSON
    cvd_vuln = {}
    for roi_disp in [ROI_DISPLAY[r] for r in ROIS]:
        info = breakdown.get(roi_disp, {})
        cvd_vuln[roi_disp] = info.get('cvd_per_color', {})
    with open(output_dir / 'cvd_vulnerability.json', 'w') as f:
        json.dump(cvd_vuln, f, indent=2)
    print(f'  Saved: {output_dir / "cvd_vulnerability.json"}')

    plot_cvd_vulnerability_table(breakdown, output_dir)

    # ---- Polar Plots ----
    print('\n[4/4] Generating polar plots...')
    for roi in ROIS:
        roi_disp = ROI_DISPLAY[roi]
        if roi_disp in breakdown:
            plot_polar_summary(breakdown, roi_disp, validation_dir, output_dir)

    plot_polar_individual(validation_dir, output_dir)

    # ---- Save config ----
    config = {
        'script': 'analyze_per_color_loco.py',
        'validation_dir': str(validation_dir),
        'output_dir': str(output_dir),
        'model': 'ridge_gcv',
        'hc_subjects': HC_SUBJECTS,
        'cvd_subjects': CVD_SUBJECTS,
        'rois': ROIS,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2, default=str)

    print('\nDone.')


if __name__ == '__main__':
    main()
