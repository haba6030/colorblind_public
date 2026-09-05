#!/usr/bin/env python3
"""Generate Forward Encoding Model pipeline figure (1x4 horizontal layout) for paper.

Adapted from docs/OHBM_abstract/create_encoding_pipeline_figure.py — same 4 stages
(Color → Channel → Voxel → Channel → Color), rearranged 2x2 → 1x4 landscape.

Output: docs/PAPER/Figures/fig_forward_encoder.{png,pdf}

2026-09-02 (author request, same policy as fig7_filter): the suptitle and the
descriptive second lines of the stage titles are removed -- the LaTeX caption
(fig:forward) carries the full stage descriptions.  Remaining text is enlarged
so it stays legible after the ~3x reduction to \textwidth, and the font is
pinned to Arial per the submission requirement.

2026-09-03 (author request): the equation boxes sat directly under the stage
diagrams while empty canvas remained beneath them.  They are moved down into
that band, which both separates them from the diagrams and removes the dead
margin at the foot of the figure.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : figures
# Manuscript: shared module
# Original  : docs/PAPER/Figures/scripts/generate_forward_encoder_fig.py  (development repository, commit 53c81c2)
# Role      : figure generator or asset
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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import matplotlib
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

COLOR_LAB = {
    'color_1': [59.90, 62.69, 3.78],
    'color_2': [64.20, 49.20, 45.58],
    'color_3': [57.27, 13.06, 41.69],
    'color_4': [69.08, -55.02, 47.38],
    'color_5': [74.61, -41.33, -4.89],
    'color_6': [69.14, -11.45, -40.91],
    'color_7': [60.68, 19.18, -54.13],
    'color_8': [60.17, 46.82, -40.31],
}


def lab2rgb_accurate(L, a, b, clip=True):
    L, a, b = float(L), float(a), float(b)
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
    if clip:
        rgb = np.clip(rgb, 0, 1)
    return tuple(rgb)


def create_basis_function_plot(hue_deg=0):
    angles = np.linspace(0, 360, 360)
    dist = np.abs(angles - hue_deg)
    dist = np.where(dist > 180, 360 - dist, dist)
    response = np.cos(np.deg2rad(dist))
    response = np.where(response > 0, response ** 2, 0)
    return angles, response


def create_pipeline_figure(out_dir):
    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 4, wspace=0.30, left=0.04, right=0.99, top=0.90, bottom=0.13)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])

    colors_bf = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue']
    channel_centers = [0, 60, 120, 180, 240, 300]

    # ====================== STAGE 1: Color → Channel ======================
    ax1.set_title('Stage 1', fontsize=20, fontweight='bold', pad=12)
    theta_bg = np.linspace(0, 360, 360)
    for ang in theta_bg:
        r = max(0, np.cos(np.deg2rad(ang)))
        g = max(0, np.cos(np.deg2rad(ang - 120)))
        b = max(0, np.cos(np.deg2rad(ang + 120)))
        norm = max(r, g, b)
        if norm > 0:
            r, g, b = r/norm, g/norm, b/norm
        ax1.axvline(ang, color=(r, g, b), alpha=0.08, linewidth=1)
    for center, color in zip(channel_centers, colors_bf):
        angles, response = create_basis_function_plot(center)
        ax1.plot(angles, response, linewidth=2.2, color=color, alpha=0.85)
        ax1.plot([center], [1.0], 'o', markersize=8, color=color,
                 markeredgecolor='black', markeredgewidth=1.2, zorder=10)
    ax1.set_xlabel('Hue angle (deg)', fontsize=16)
    ax1.set_ylabel('Channel response', fontsize=16)
    ax1.set_xlim(0, 360)
    ax1.set_ylim(-0.05, 1.15)
    ax1.tick_params(labelsize=13)
    ax1.grid(True, alpha=0.25, linestyle='--')

    # ====================== STAGE 2: Channel → Voxel (Training) ======================
    ax2.set_title('Stage 2', fontsize=20, fontweight='bold', pad=12)
    ax2.axis('off')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)

    channel_y = np.linspace(7, 3, 6)
    for i, y in enumerate(channel_y):
        circle = Circle((2, y), 0.22, facecolor=colors_bf[i],
                        edgecolor='black', linewidth=1.5, zorder=5)
        ax2.add_patch(circle)
        ax2.text(1.4, y, f'C{i+1}', ha='right', va='center',
                 fontsize=14, fontweight='bold')

    voxel_y = np.linspace(2.5, 7.5, 8)
    for i, y in enumerate(voxel_y):
        rect = Rectangle((7.5, y-0.18), 0.7, 0.36, facecolor='lightblue',
                         edgecolor='black', linewidth=1.2, zorder=5)
        ax2.add_patch(rect)
        ax2.text(8.95, y, f'V{i+1}', ha='left', va='center',
                 fontsize=13, fontweight='bold')

    np.random.seed(42)
    for _ in range(20):
        c_idx = np.random.randint(0, 6)
        v_idx = np.random.randint(0, 8)
        weight = np.random.randn()
        alpha_ = min(abs(weight) * 0.3, 0.5)
        color = 'blue' if weight > 0 else 'red'
        lw = abs(weight) * 0.4 + 0.3
        ax2.plot([2.22, 7.5], [channel_y[c_idx], voxel_y[v_idx]],
                 color=color, alpha=alpha_, linewidth=lw, zorder=1)

    ax2.text(2, 8.7, 'Channels', ha='center', va='center',
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))
    ax2.text(8.2, 8.7, 'Voxels', ha='center', va='center',
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', alpha=0.8))
    ax2.text(5, 0.35, r'$\mathbf{B} = \mathbf{C}\mathbf{W}^\mathsf{T}$',
             ha='center', va='center', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       edgecolor='black', linewidth=1.5))

    # ====================== STAGE 3: Voxel → Channel (Testing) ======================
    ax3.set_title('Stage 3', fontsize=20, fontweight='bold', pad=12)
    ax3.axis('off')
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)

    voxel_y_test = np.linspace(2.5, 7.5, 8)
    for i, y in enumerate(voxel_y_test):
        rect = Rectangle((1.2, y-0.18), 0.7, 0.36, facecolor='lightcoral',
                         edgecolor='black', linewidth=1.2, zorder=5)
        ax3.add_patch(rect)
        ax3.text(0.55, y, f'V{i+1}', ha='right', va='center',
                 fontsize=13, fontweight='bold')

    channel_y_pred = np.linspace(3, 7, 6)
    for i, y in enumerate(channel_y_pred):
        circle = Circle((8, y), 0.22, facecolor=colors_bf[i],
                        edgecolor='black', linewidth=1.5,
                        linestyle='--', zorder=5)
        ax3.add_patch(circle)
        ax3.text(8.45, y, f'$\\hat{{C}}_{i+1}$', ha='left', va='center',
                 fontsize=14, fontweight='bold')

    for _ in range(20):
        v_idx = np.random.randint(0, 8)
        c_idx = np.random.randint(0, 6)
        weight = np.random.randn()
        alpha_ = min(abs(weight) * 0.3, 0.5)
        color = 'darkgreen' if weight > 0 else 'purple'
        lw = abs(weight) * 0.4 + 0.3
        ax3.plot([1.9, 7.78], [voxel_y_test[v_idx], channel_y_pred[c_idx]],
                 color=color, alpha=alpha_, linewidth=lw, zorder=1)

    ax3.text(1.55, 8.7, 'Test voxels', ha='center', va='center',
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcoral', alpha=0.8))
    ax3.text(8, 8.7, 'Pred. channels', ha='center', va='center',
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))
    ax3.text(5, 0.35, r'$\hat{\mathbf{c}} = \mathbf{W}\mathbf{x}$',
             ha='center', va='center', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       edgecolor='black', linewidth=1.5))

    # ====================== STAGE 4: Channel → Color ======================
    ax4.set_title('Stage 4', fontsize=20, fontweight='bold', pad=12)
    ax4.axis('off')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)

    channel_bars_x = np.linspace(1.0, 3.6, 6)
    predicted_values = [0.3, 0.1, 0.05, 0.2, 0.8, 0.9]
    for i, (x, val) in enumerate(zip(channel_bars_x, predicted_values)):
        height = val * 2.7
        rect = Rectangle((x-0.18, 5-height/2), 0.36, height,
                         facecolor=colors_bf[i], edgecolor='black',
                         linewidth=1.2, alpha=0.85, zorder=5)
        ax4.add_patch(rect)
        # per-bar channel labels removed (2026-09-02): duplicated Stage 3 and
        # overlapped after the font enlargement; the caption defines the bars.

    ax4.text(2.3, 7.6, 'Predicted pattern', ha='center', va='center',
             fontsize=14, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8))

    arrow = FancyArrowPatch((4.0, 5), (4.8, 5),
                            arrowstyle='->', mutation_scale=22,
                            linewidth=2.5, color='black', zorder=10)
    ax4.add_patch(arrow)

    center_x, center_y = 7.5, 5
    radius = 1.4
    bg_circle = Circle((center_x, center_y), radius, fill=False,
                       edgecolor='gray', linewidth=1.5, linestyle='-',
                       alpha=0.5, zorder=1)
    ax4.add_patch(bg_circle)
    ax4.plot([center_x - radius - 0.2, center_x + radius + 0.2],
             [center_y, center_y], color='lightgray', linewidth=0.8,
             alpha=0.5, zorder=1)
    ax4.plot([center_x, center_x],
             [center_y - radius - 0.2, center_y + radius + 0.2],
             color='lightgray', linewidth=0.8, alpha=0.5, zorder=1)

    stimulus_angles = [0, 45, 90, 135, 180, 225, 270, 315]
    color_circle_radius = 0.20
    for i, ang_deg in enumerate(stimulus_angles, 1):
        ang_rad = np.deg2rad(ang_deg)
        xp = center_x + radius * np.cos(ang_rad)
        yp = center_y + radius * np.sin(ang_rad)
        rgb = lab2rgb_accurate(*COLOR_LAB[f'color_{i}'])
        ax4.add_patch(Circle((xp, yp), color_circle_radius,
                             facecolor=rgb, edgecolor='black',
                             linewidth=1.5, zorder=4))

    recon_angle = 255
    rr = np.deg2rad(recon_angle)
    rx = center_x + radius * 0.75 * np.cos(rr)
    ry = center_y + radius * 0.75 * np.sin(rr)
    ax4.add_patch(FancyArrowPatch((center_x, center_y), (rx, ry),
                                  arrowstyle='->', mutation_scale=20,
                                  linewidth=2.5, color='red', zorder=15))

    ax4.text(5, 0.35,
             r'$\hat\theta = \arg\max_{\theta}\,\mathrm{corr}(\hat{\mathbf{c}},\mathbf{c}(\theta))$',
             ha='center', va='center', fontsize=16, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='white',
                       edgecolor='black', linewidth=1.5))

    # suptitle removed (2026-09-02): figure title belongs in the LaTeX caption.

    out_png = os.path.join(out_dir, 'fig_forward_encoder.png')
    out_pdf = os.path.join(out_dir, 'fig_forward_encoder.pdf')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    fig.savefig(out_pdf, bbox_inches='tight')
    print(f'Wrote {out_png}')
    print(f'Wrote {out_pdf}')
    return fig


if __name__ == '__main__':
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    create_pipeline_figure(os.path.normpath(out))
