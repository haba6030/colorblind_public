#!/usr/bin/env python3
"""
utils_color_decoding.py
=======================

Common utility functions for color decoding analysis (ANOVA, RFE, PCA)

이 파일은 feature selection 방법들(ANOVA, RFE, PCA) 간 공통으로 사용되는
함수들을 제공합니다:
- Color space utilities (circular distance, Lab->RGB conversion)
- Reconstruction functions (forward encoding model, basis functions)
- Visualization functions (circular color space, brain plots)
- SNR calculation

Author: Claude Code
Date: 2025-11-29
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : common
# Manuscript: shared module
# Original  : analysis/utils/utils_color_decoding.py  (development repository, commit 53c81c2)
# Role      : Stimulus colour definitions and CIELab helpers shared by the response-estimation and rendering scripts.
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[1]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import nibabel as nib
from nilearn import plotting as niplot
from matplotlib.lines import Line2D
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Constants: Color Mappings and Definitions
# ============================================================================

# Color mappings for TEST subjects (regular 45° spacing)
LABEL2HUE_DEG = {
    'color_1': 0.0,
    'color_2': 45.0,
    'color_3': 90.0,
    'color_4': 135.0,
    'color_5': 180.0,
    'color_6': 225.0,
    'color_7': 270.0,
    'color_8': 315.0,
}

# CIELab color definitions - Extracted from experimental screenshots
# These values are derived from actual RGB values in screenshots (visual inspection)
COLOR_LAB = {
    'color_1': [59.90, 62.69, 3.78],    # 0°: Red/Pink (붉은색)
    'color_2': [64.20, 49.20, 45.58],   # 45°: Orange (주황색)
    'color_3': [57.27, 13.06, 41.69],   # 90°: Brown/Yellow (황갈색)
    'color_4': [69.08, -55.02, 47.38],  # 135°: Green (녹색)
    'color_5': [74.61, -41.33, -4.89],  # 180°: Cyan (청록색)
    'color_6': [69.14, -11.45, -40.91], # 225°: Blue (파란색)
    'color_7': [60.68, 19.18, -54.13],  # 270°: Dark Blue (진한 파랑)
    'color_8': [60.17, 46.82, -40.31],  # 315°: Purple/Pink (보라/분홍)
}

N_COLORS = 8
N_RUNS = 6

# ============================================================================
# Color Space Utilities
# ============================================================================

def circular_diff_deg(a, b):
    """
    원형 공간에서의 각도 차이 계산 (절댓값)

    Args:
        a, b: Angles in degrees (0-360)

    Returns:
        Absolute circular difference (0-180)

    Example:
        circular_diff_deg(10, 350) = 20  (not 340)
        circular_diff_deg(0, 180) = 180
    """
    diff = np.abs(a - b)
    diff = np.where(diff > 180, 360 - diff, diff)
    return diff


def lab2rgb_accurate(L, a, b, clip=True):
    """
    CIELab → RGB 정확한 색공간 변환

    Parameters:
    -----------
    L : float - Lightness (0-100)
    a : float - Green-Red axis
    b : float - Blue-Yellow axis
    clip : bool - Clip RGB values to [0, 1]

    Returns:
    --------
    tuple : (R, G, B) in [0, 1] range
    """
    L, a, b = float(L), float(a), float(b)

    # Lab to XYZ
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200

    xyz = np.array([x, y, z])
    xyz = np.where(xyz > 0.206893, xyz**3, (xyz - 16/116) / 7.787)
    xyz *= [0.95047, 1., 1.08883]  # D65 white point

    # XYZ to RGB (sRGB matrix)
    rgb = np.dot([[3.2406, -1.5372, -0.4986],
                  [-0.9689, 1.8758, 0.0415],
                  [0.0557, -0.2040, 1.0570]], xyz)

    # Gamma correction
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)

    if clip:
        rgb = np.clip(rgb, 0, 1)

    return tuple(rgb)


def get_stimulus_color_rgb(color_name):
    """
    실제 자극 색상의 RGB 값 반환 (시각화용)

    Parameters:
    -----------
    color_name : str - Color name (e.g., 'color_1')

    Returns:
    --------
    tuple : (R, G, B) in [0, 1] range
    """
    if color_name in COLOR_LAB:
        L, a, b = COLOR_LAB[color_name]
        return lab2rgb_accurate(L, a, b)
    else:
        return (0.5, 0.5, 0.5)  # Gray for unknown


# ============================================================================
# Forward Encoding Model (Brouwer & Heeger 2009)
# ============================================================================

def create_basis_functions(n_channels=6):
    """
    6-channel basis functions 생성 (half-wave rectified squared sinusoids)

    B&H 2009 방법: 각 channel은 특정 hue에서 최대 반응을 보이는
    종 모양(bell-shaped) 함수

    Returns:
    --------
    basis : ndarray (360, 6) - Basis functions for each hue (0-359°)
    """
    hues = np.linspace(0, 360, n_channels, endpoint=False)
    basis = np.zeros((360, n_channels))

    for i, center_hue in enumerate(hues):
        for h in range(360):
            # Circular distance
            dist = np.abs(h - center_hue)
            if dist > 180:
                dist = 360 - dist

            # Half-wave rectified cosine, squared
            response = np.cos(np.deg2rad(dist))
            if response > 0:
                basis[h, i] = response ** 2
            else:
                basis[h, i] = 0

    return basis


def diag_linear_predict(train_X, train_y, test_X):
    """
    Diagonal Linear Discriminant Analysis (B&H 2009 classification 방법)

    LDA의 단순화 버전: 각 feature가 독립적이라고 가정 (diagonal covariance)

    Args:
        train_X: (n_train, n_features) Training data
        train_y: (n_train,) Training labels
        test_X: (n_test, n_features) Test data

    Returns:
        preds: (n_test,) Predicted labels
    """
    classes = np.unique(train_y)
    means = np.stack([train_X[train_y==c].mean(axis=0) for c in classes])
    vars_  = np.stack([train_X[train_y==c].var(axis=0) + 1e-8 for c in classes])

    ll = []
    for k in range(len(classes)):
        ll_k = -0.5 * (
            np.log(2*np.pi*vars_[k]).sum() +
            ((test_X - means[k])**2 / vars_[k]).sum(axis=1)
        )
        ll.append(ll_k)
    ll = np.stack(ll, axis=1)
    preds = classes[ll.argmax(axis=1)]
    return preds


def evaluate_reconstruction(amplitudes_selected, label2hue_deg=LABEL2HUE_DEG,
                           n_runs=N_RUNS, n_colors=N_COLORS, alpha=0.0, **kwargs):
    """
    Forward encoding model로 color reconstruction 수행

    B&H 2009 방법:
    1. Training: B = W × C (B=brain activity, W=weights, C=channel responses)
    2. Test: C_est = (W^T W)^-1 W^T B_test
    3. Reconstruction: 360개 hue 중 C_est와 가장 유사한 것 선택

    Ridge Regression (NEW):
    - If alpha > 0: W = B × C^T × (C × C^T + α·I)^-1
    - If alpha = 0: W = B × C^T × (C × C^T)^-1 (OLS, original)

    Args:
        amplitudes_selected: (n_runs, n_colors, k) Selected voxel amplitudes (z-scored)
        label2hue_deg: Color label to hue mapping
        n_runs: Number of runs (default: 6)
        n_colors: Number of colors (default: 8)
        alpha: Ridge regularization parameter (default: 0.0 for OLS)
               Recommended: 10.0 for k=5, 50-100 for k=40-100
        **kwargs: Additional arguments (e.g., selected_indices - ignored but accepted for compatibility)

    Returns:
        reconstruction_results: List of dict with reconstruction metrics per fold
        mean_error: Mean reconstruction error across folds (degrees)

    주의: amplitudes_selected는 이미 z-scored되어 있어야 함!
    """
    _, _, k = amplitudes_selected.shape

    # Create 6-channel basis functions
    basis_functions = create_basis_functions(n_channels=6)

    def hue_to_channels(hue_deg):
        """Convert hue (0-360) to 6 channel outputs"""
        hue_idx = int(np.round(hue_deg)) % 360
        return basis_functions[hue_idx]

    reconstruction_results = []

    # Leave-one-run-out cross-validation
    for test_run in range(n_runs):
        train_runs = [r for r in range(n_runs) if r != test_run]

        # Prepare data (already z-scored)
        X_train = amplitudes_selected[train_runs].reshape(-1, k)
        y_train = np.tile(np.arange(n_colors), len(train_runs))

        X_test = amplitudes_selected[test_run]
        y_test = np.arange(n_colors)

        # Train forward model: B = W × C
        C_train = []
        for color_idx in y_train:
            color_name = f'color_{color_idx+1}'
            # Safe hue extraction
            if isinstance(label2hue_deg, dict):
                hue_deg = label2hue_deg[color_name]
            else:
                hue_deg = label2hue_deg[int(color_idx)]
            channels = hue_to_channels(hue_deg)
            C_train.append(channels)
        C_train = np.array(C_train).T  # (6, n_train)

        # Estimate weights: W = B × C^T × (C × C^T + α·I)^-1
        # If alpha > 0: Ridge regression
        # If alpha = 0: OLS (original)
        try:
            n_channels = C_train.shape[0]
            if alpha > 0:
                # Ridge: W = B × C^T × (C × C^T + α·I)^-1
                CCT_ridge = C_train @ C_train.T + alpha * np.eye(n_channels)
                W = X_train.T @ C_train.T @ np.linalg.inv(CCT_ridge)
            else:
                # OLS: W = B × C^T × (C × C^T)^-1
                W = X_train.T @ C_train.T @ np.linalg.inv(C_train @ C_train.T)
        except np.linalg.LinAlgError:
            # Singular matrix: use pseudo-inverse
            if alpha > 0:
                CCT_ridge = C_train @ C_train.T + alpha * np.eye(n_channels)
                W = X_train.T @ C_train.T @ np.linalg.pinv(CCT_ridge)
            else:
                W = X_train.T @ C_train.T @ np.linalg.pinv(C_train @ C_train.T)

        # Test: estimate channels from test data
        C_test_est = np.linalg.pinv(W.T @ W) @ W.T @ X_test.T  # (6, n_test)

        # Reconstruct hues
        reconstructed_hues = []
        true_hues = []

        for test_idx, color_idx in enumerate(y_test):
            # Estimated channels
            estimated_channels = C_test_est[:, test_idx]

            # Find best matching hue (0-360)
            correlations = []
            for h in range(360):
                template_channels = basis_functions[h]
                # Safe correlation computation
                corr_matrix = np.corrcoef(estimated_channels, template_channels)
                if corr_matrix.shape == (2, 2):
                    corr = corr_matrix[0, 1]
                else:
                    # Fallback to dot product if correlation fails
                    corr = np.dot(estimated_channels, template_channels) / (np.linalg.norm(estimated_channels) * np.linalg.norm(template_channels) + 1e-10)
                correlations.append(corr)

            correlations = np.array(correlations)
            reconstructed_hue = np.argmax(correlations)

            # True hue (safe extraction)
            color_name = f'color_{color_idx+1}'
            if isinstance(label2hue_deg, dict):
                true_hue = label2hue_deg[color_name]
            else:
                true_hue = label2hue_deg[int(color_idx)]

            reconstructed_hues.append(reconstructed_hue)
            true_hues.append(true_hue)

        # Calculate reconstruction error
        errors = circular_diff_deg(np.array(reconstructed_hues), np.array(true_hues))
        mean_error = errors.mean()

        # ============================================================================
        # NEW: Hit-rate metrics (±22.5°, ±45°)
        # ============================================================================
        hit_rate_22_5 = np.mean(errors <= 22.5)
        hit_rate_45 = np.mean(errors <= 45.0)

        reconstruction_results.append({
            'test_run': test_run + 1,
            'mean_error': mean_error,
            'reconstructed_hues': np.array(reconstructed_hues),  # Convert to numpy array
            'true_hues': np.array(true_hues),  # Convert to numpy array
            'errors': np.array(errors),  # Convert to numpy array
            'hit_rate_22_5': hit_rate_22_5,
            'hit_rate_45': hit_rate_45
        })

    mean_reconstruction_error = np.mean([r['mean_error'] for r in reconstruction_results])
    mean_hit_rate_22_5 = np.mean([r['hit_rate_22_5'] for r in reconstruction_results])
    mean_hit_rate_45 = np.mean([r['hit_rate_45'] for r in reconstruction_results])

    # ============================================================================
    # NEW: Per-color statistics across all folds
    # ============================================================================
    all_errors = []
    all_true_hues = []
    for result in reconstruction_results:
        all_errors.extend(result['errors'])
        all_true_hues.extend(result['true_hues'])

    all_errors = np.array(all_errors)
    all_true_hues = np.array(all_true_hues)

    # Calculate per-color statistics
    per_color_stats = {}
    for color_idx in range(n_colors):
        color_name = f'color_{color_idx+1}'
        # Safe hue extraction
        if isinstance(label2hue_deg, dict):
            true_hue = label2hue_deg[color_name]
        else:
            true_hue = label2hue_deg[int(color_idx)]

        # Find all trials for this color
        color_mask = all_true_hues == true_hue
        color_errors = all_errors[color_mask]

        per_color_stats[color_name] = {
            'mean_error': np.mean(color_errors),
            'std_error': np.std(color_errors),
            'hit_rate_22_5': np.mean(color_errors <= 22.5),
            'hit_rate_45': np.mean(color_errors <= 45.0),
            'n_trials': len(color_errors)
        }

    return reconstruction_results, mean_reconstruction_error, per_color_stats


def evaluate_reconstruction_leave_one_color_out(amplitudes_selected, label2hue_deg=LABEL2HUE_DEG,
                                                n_runs=N_RUNS, n_colors=N_COLORS, alpha=0.0, **kwargs):
    """
    Leave-one-color-out cross-validation for reconstruction

    각 색상을 test set으로 하여 reconstruction 수행
    이를 통해 모델이 특정 색상에 overfitting되지 않았는지 확인

    Ridge Regression (NEW):
    - If alpha > 0: W = B × C^T × (C × C^T + α·I)^-1
    - If alpha = 0: W = B × C^T × (C × C^T)^-1 (OLS, original)

    Args:
        amplitudes_selected: (n_runs, n_colors, k) Selected voxel amplitudes (z-scored)
        label2hue_deg: Color label to hue mapping
        alpha: Ridge regularization parameter (default: 0.0 for OLS)
        n_runs: Number of runs (default: 6)
        n_colors: Number of colors (default: 8)
        **kwargs: Additional arguments (ignored)

    Returns:
        reconstruction_results: List of dict with reconstruction metrics per color
        mean_error: Mean reconstruction error across colors (degrees)
        per_color_stats: Per-color statistics dict
    """
    _, _, k = amplitudes_selected.shape

    # Create 6-channel basis functions
    basis_functions = create_basis_functions(n_channels=6)

    def hue_to_channels(hue_deg):
        """Convert hue (0-360) to 6 channel outputs"""
        hue_idx = int(np.round(hue_deg)) % 360
        return basis_functions[hue_idx]

    reconstruction_results = []

    # Leave-one-color-out cross-validation
    for test_color_idx in range(n_colors):
        train_color_indices = [c for c in range(n_colors) if c != test_color_idx]

        # Prepare training data (all runs, all training colors)
        X_train = amplitudes_selected[:, train_color_indices, :].reshape(-1, k)
        y_train = np.tile(train_color_indices, n_runs)

        # Prepare test data (all runs, test color only)
        X_test = amplitudes_selected[:, test_color_idx, :]  # (n_runs, k)

        # Train forward model: B = W × C
        C_train = []
        for color_idx in y_train:
            color_name = f'color_{color_idx+1}'
            # Safe hue extraction
            if isinstance(label2hue_deg, dict):
                hue_deg = label2hue_deg[color_name]
            else:
                hue_deg = label2hue_deg[int(color_idx)]
            channels = hue_to_channels(hue_deg)
            C_train.append(channels)
        C_train = np.array(C_train).T  # (6, n_train)

        # Estimate weights: W = B × C^T × (C × C^T + α·I)^-1
        # If alpha > 0: Ridge regression
        # If alpha = 0: OLS (original)
        try:
            n_channels = C_train.shape[0]
            if alpha > 0:
                # Ridge: W = B × C^T × (C × C^T + α·I)^-1
                CCT_ridge = C_train @ C_train.T + alpha * np.eye(n_channels)
                W = X_train.T @ C_train.T @ np.linalg.inv(CCT_ridge)
            else:
                # OLS: W = B × C^T × (C × C^T)^-1
                W = X_train.T @ C_train.T @ np.linalg.inv(C_train @ C_train.T)
        except np.linalg.LinAlgError:
            if alpha > 0:
                CCT_ridge = C_train @ C_train.T + alpha * np.eye(n_channels)
                W = X_train.T @ C_train.T @ np.linalg.pinv(CCT_ridge)
            else:
                W = X_train.T @ C_train.T @ np.linalg.pinv(C_train @ C_train.T)

        # Test: estimate channels from test data
        C_test_est = np.linalg.pinv(W.T @ W) @ W.T @ X_test.T  # (6, n_runs)

        # Reconstruct hues for all runs
        reconstructed_hues = []
        true_hues = []

        test_color_name = f'color_{test_color_idx+1}'
        # Safe hue extraction
        if isinstance(label2hue_deg, dict):
            true_hue = label2hue_deg[test_color_name]
        else:
            true_hue = label2hue_deg[test_color_idx]

        for run_idx in range(n_runs):
            # Estimated channels
            estimated_channels = C_test_est[:, run_idx]

            # Find best matching hue (0-360)
            correlations = []
            for h in range(360):
                template_channels = basis_functions[h]
                # Safe correlation computation
                corr_matrix = np.corrcoef(estimated_channels, template_channels)
                if corr_matrix.shape == (2, 2):
                    corr = corr_matrix[0, 1]
                else:
                    # Fallback to dot product if correlation fails
                    corr = np.dot(estimated_channels, template_channels) / (np.linalg.norm(estimated_channels) * np.linalg.norm(template_channels) + 1e-10)
                correlations.append(corr)

            correlations = np.array(correlations)
            reconstructed_hue = np.argmax(correlations)

            reconstructed_hues.append(reconstructed_hue)
            true_hues.append(true_hue)

        # Calculate reconstruction error
        errors = circular_diff_deg(np.array(reconstructed_hues), np.array(true_hues))
        mean_error = errors.mean()

        # Hit-rate metrics
        hit_rate_22_5 = np.mean(errors <= 22.5)
        hit_rate_45 = np.mean(errors <= 45.0)

        reconstruction_results.append({
            'test_color': test_color_name,
            'test_color_idx': test_color_idx,
            'mean_error': mean_error,
            'std_error': np.std(errors),
            'reconstructed_hues': np.array(reconstructed_hues),
            'true_hues': np.array(true_hues),
            'errors': np.array(errors),
            'hit_rate_22_5': hit_rate_22_5,
            'hit_rate_45': hit_rate_45,
            'n_trials': len(errors)
        })

    mean_reconstruction_error = np.mean([r['mean_error'] for r in reconstruction_results])

    # Per-color stats (same as results for LOCO)
    per_color_stats = {}
    for result in reconstruction_results:
        color_name = result['test_color']
        per_color_stats[color_name] = {
            'mean_error': result['mean_error'],
            'std_error': result['std_error'],
            'hit_rate_22_5': result['hit_rate_22_5'],
            'hit_rate_45': result['hit_rate_45'],
            'n_trials': result['n_trials']
        }

    return reconstruction_results, mean_reconstruction_error, per_color_stats


# ============================================================================
# SNR Calculation
# ============================================================================

def compute_voxel_snr(amplitudes_z):
    """
    Voxel-wise SNR 계산

    SNR = signal / noise
    signal = std(mean_amplitude_per_color across runs)  # Color 간 변동
    noise = mean(std_amplitude_per_color_across_runs)   # Run 내 변동

    높은 SNR = 해당 voxel이 색상을 안정적으로 잘 구분함

    Args:
        amplitudes_z: (n_runs, n_colors, n_voxels) Z-scored amplitudes

    Returns:
        voxel_snr: (n_voxels,) SNR for each voxel
    """
    n_runs, n_colors, n_voxels = amplitudes_z.shape
    voxel_snr = np.zeros(n_voxels)

    for v in range(n_voxels):
        # Mean amplitude per color (across runs)
        mean_per_color = amplitudes_z[:, :, v].mean(axis=0)  # (n_colors,)

        # Signal: variability across colors
        signal = np.std(mean_per_color)

        # Noise: mean of within-color variability across runs
        std_per_color = amplitudes_z[:, :, v].std(axis=0)  # (n_colors,)
        noise = np.mean(std_per_color)

        # SNR
        if noise > 1e-10:
            voxel_snr[v] = signal / noise
        else:
            voxel_snr[v] = 0.0

    return voxel_snr


def compute_per_color_metrics(reconstruction_results, n_colors=N_COLORS):
    """
    색상별 reconstruction error 및 hit-rate 계산

    Args:
        reconstruction_results: List of reconstruction results per fold
        n_colors: Number of colors (default: 8)

    Returns:
        per_color_df: DataFrame with per-color metrics
            - color_idx: Color index (0-7)
            - mean_error: Mean reconstruction error for this color
            - hit_rate_22_5: Hit rate at ±22.5°
            - hit_rate_45: Hit rate at ±45°
    """
    per_color_errors = [[] for _ in range(n_colors)]

    # Collect errors for each color across all folds
    for result in reconstruction_results:
        errors = result['errors']
        for color_idx in range(n_colors):
            per_color_errors[color_idx].append(errors[color_idx])

    # Calculate metrics for each color
    per_color_metrics = []
    for color_idx in range(n_colors):
        errors = np.array(per_color_errors[color_idx])

        per_color_metrics.append({
            'color_idx': color_idx + 1,  # 1-indexed for color_1, color_2, ...
            'color_name': f'color_{color_idx + 1}',
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'hit_rate_22_5': np.mean(errors <= 22.5),
            'hit_rate_45': np.mean(errors <= 45.0)
        })

    return pd.DataFrame(per_color_metrics)


# ============================================================================
# Visualization Functions
# ============================================================================

def visualize_circular_color_space(reconstruction_results, mean_error, method_name,
                                   k, roi, subject, output_dir,
                                   label2hue_deg=LABEL2HUE_DEG, n_colors=N_COLORS,
                                   per_color_stats=None, config_suffix='', compact=False):
    """
    Circular color space visualization (B&H 2009 style)

    원형 공간에 true hue는 바깥쪽 테두리에, prediction은 안쪽에 표시
    Prediction의 위치(각도)는 reconstructed hue, 색상은 true color

    Args:
        reconstruction_results: List of reconstruction results per fold
        mean_error: Mean reconstruction error (degrees)
        method_name: Feature selection method name (e.g., 'ANOVA', 'RFE')
        k: Number of features used
        roi: ROI name
        subject: Subject ID
        output_dir: Output directory path
        label2hue_deg: Color to hue mapping
        n_colors: Number of colors
        per_color_stats: Per-color statistics dict (optional)
        config_suffix: Configuration suffix for filename
        compact: If True, use smaller figure size for compact layout (default: False)
    """
    # Use smaller figure size if compact mode
    if compact:
        fig = plt.figure(figsize=(10, 5))
    else:
        fig = plt.figure(figsize=(14, 6))

    # Left: Training colors reconstruction
    ax1 = fig.add_subplot(1, 2, 1, projection='polar')
    ax1.set_theta_zero_location("E")        # 0° at right (middle-right)
    ax1.set_theta_direction(1)              # Anticlockwise positive
    ax1.set_rticks([])                      # Hide radial ticks
    ax1.grid(alpha=0.25)

    rng = np.random.default_rng(42)

    # Use smaller radius for compact mode
    if compact:
        r_center = 0.5
        r_pred_base = 0.42
    else:
        r_center = 1.0
        r_pred_base = 0.85

    # Plot true colors at border and predictions inside
    for color_idx in range(n_colors):
        color_name = f'color_{color_idx + 1}'

        # Safe extraction of true_hue
        try:
            if isinstance(label2hue_deg, dict):
                true_hue = label2hue_deg[color_name]
            else:
                # If it's an array, use color_idx
                true_hue = label2hue_deg[color_idx]
        except (KeyError, IndexError, TypeError) as e:
            print(f"    Warning: Could not get true_hue for {color_name}, skipping. Error: {e}")
            continue

        true_rgb = get_stimulus_color_rgb(color_name)

        # True color at border (large, solid)
        ax1.scatter([np.deg2rad(true_hue)], [r_center], s=110,
                   color=true_rgb, edgecolor='k', linewidths=0.8, zorder=4)

        # All predictions for this color (position=angle, color=truth)
        for result in reconstruction_results:
            try:
                reconstructed_hues_array = result['reconstructed_hues']
                if not isinstance(reconstructed_hues_array, np.ndarray):
                    reconstructed_hues_array = np.array(reconstructed_hues_array)

                # Safe indexing: ensure color_idx is valid integer
                if color_idx < len(reconstructed_hues_array):
                    pred_hue = reconstructed_hues_array[int(color_idx)]

                    # Jitter radius slightly to avoid overlap
                    r_jit = r_pred_base + rng.uniform(-0.03, 0.03)
                    ax1.scatter([np.deg2rad(pred_hue)], [r_jit], s=28,
                               color=true_rgb, alpha=0.65, zorder=3)
            except (IndexError, TypeError, KeyError) as e:
                # Skip this prediction if indexing fails
                continue

    # Set ylim based on compact mode
    if compact:
        ax1.set_ylim([0, 0.6])
    else:
        ax1.set_ylim([0, 1.1])

    ax1.set_title(f'{method_name} (k={k})\nMean error: {mean_error:.1f}°',
                  fontsize=12, fontweight='bold')

    # Right: Legend and summary
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis('off')

    legend_elems = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='k',
               label='True hue (stimulus)', markersize=10),
        Line2D([0],[0], marker='o', color='w', markerfacecolor='gray', alpha=0.65,
               label='Predictions (color=truth)', markersize=6),
    ]
    ax2.legend(handles=legend_elems, loc='center', fontsize=12, frameon=False)

    # Summary text
    summary_text = f"""
{method_name} Feature Selection

Reconstruction Error: {mean_error:.1f}°
Chance Level: 90.0°

Features: {k} voxels
Subject: {subject}
ROI: {roi}
"""

    # Add per-color statistics if available
    if per_color_stats is not None:
        summary_text += "\n\nPer-Color Statistics:\n"
        for color_idx in range(n_colors):
            color_name = f'color_{color_idx+1}'
            try:
                stats = per_color_stats[color_name]
                summary_text += (f"\n{color_name}: {stats['mean_error']:.1f}° ± {stats['std_error']:.1f}°"
                               f"\n  ≤22.5°: {stats['hit_rate_22_5']:.1%}  ≤45°: {stats['hit_rate_45']:.1%}")
            except (KeyError, TypeError, IndexError) as e:
                # Skip this color if stats not available
                summary_text += f"\n{color_name}: N/A"
                continue

    ax2.text(0.5, 0.5, summary_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
             family='monospace')

    plt.suptitle(f'{roi}: Circular Color Space (Reconstruction)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Include config_suffix in filename (for permutation scenarios)
    # Format: {method}_circular_{config_suffix}_sub-{subject}_{roi}_k{k}.png
    if config_suffix:
        fig_path = output_dir / f'{method_name.lower()}_circular_{config_suffix}_sub-{subject}_{roi}_k{k}.png'
    else:
        fig_path = output_dir / f'{method_name.lower()}_circular_sub-{subject}_{roi}_k{k}.png'

    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Circular plot saved: {fig_path}")

    return fig_path


def visualize_selected_voxels_brain(selected_indices, roi_mask_path, feature_values,
                                    method_name, k, roi, subject, output_dir, config_suffix=None):
    """
    Selected voxels의 brain 위치 시각화

    ROI mask를 사용하여 선택된 voxel들의 3D 위치를 표시
    각 voxel의 intensity는 feature importance (F-value, RFE ranking 등)

    Args:
        selected_indices: (k,) Selected voxel indices
        roi_mask_path: Path to ROI mask file (.nii.gz)
        feature_values: (n_voxels,) Feature importance values (e.g., F-values)
        method_name: Feature selection method name
        k: Number of selected features
        roi: ROI name
        subject: Subject ID
        output_dir: Output directory path
        config_suffix: Optional suffix for filename

    Returns:
        fig_path: Path to saved figure (or None if failed)
    """
    try:
        # Load ROI mask
        if not Path(roi_mask_path).exists():
            print(f"  ⚠️  ROI mask not found: {roi_mask_path}")
            print(f"     Skipping brain visualization...")
            return None

        mask_img = nib.load(roi_mask_path)
        mask_data = mask_img.get_fdata()

        # Get voxel coordinates
        voxel_coords = np.argwhere(mask_data > 0)  # All voxels in mask

        # Create a new volume with selected voxels highlighted
        highlight_data = np.zeros_like(mask_data)

        for idx in selected_indices:
            if idx < len(voxel_coords):
                coord = voxel_coords[idx]
                # Use feature value as intensity
                highlight_data[coord[0], coord[1], coord[2]] = feature_values[idx]

        highlight_img = nib.Nifti1Image(highlight_data, mask_img.affine, mask_img.header)

        # Plot brain slices
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Axial view
        niplot.plot_stat_map(highlight_img, display_mode='z', cut_coords=5,
                            title=f'Selected Voxels (k={k}) - Axial',
                            axes=axes[0, 0], colorbar=True, cmap='hot')

        # Sagittal view
        niplot.plot_stat_map(highlight_img, display_mode='x', cut_coords=5,
                            title=f'Selected Voxels (k={k}) - Sagittal',
                            axes=axes[0, 1], colorbar=True, cmap='hot')

        # Coronal view
        niplot.plot_stat_map(highlight_img, display_mode='y', cut_coords=5,
                            title=f'Selected Voxels (k={k}) - Coronal',
                            axes=axes[1, 0], colorbar=True, cmap='hot')

        # Glass brain
        niplot.plot_glass_brain(highlight_img,
                               title=f'Selected Voxels (k={k}) - Glass Brain',
                               axes=axes[1, 1], colorbar=True, cmap='hot')

        plt.suptitle(f'{roi}: Selected Voxels Brain Visualization ({method_name}, k={k})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        # Include config_suffix in filename (for permutation scenarios)
        # Format: {method}_brain_{config_suffix}_sub-{subject}_{roi}_k{k}.png
        if config_suffix:
            fig_path = output_dir / f'{method_name.lower()}_brain_{config_suffix}_sub-{subject}_{roi}_k{k}.png'
        else:
            fig_path = output_dir / f'{method_name.lower()}_brain_sub-{subject}_{roi}_k{k}.png'

        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Brain plot saved: {fig_path}")
        return fig_path

    except Exception as e:
        print(f"  ⚠️  Brain visualization failed: {e}")
        print(f"     Continuing without brain plot...")
        return None


def get_roi_mask_path(subject, roi, timestamp='BHbase'):
    """
    ROI mask 파일 경로 반환

    Args:
        subject: Subject ID (e.g., '01', '02', ..., '10')
        roi: ROI name (e.g., 'V1', 'V2', 'V3', 'hV4')
        timestamp: Preprocessing timestamp (default: 'BHbase')

    Returns:
        Path to ROI mask file (str)
    """
    # Primary path: subject-specific roi_pipeline output
    # Format: derivatives/sub-{subject}/roi_pipeline/{ROI}_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjFalse.nii.gz
    # Example: derivatives/sub-03/roi_pipeline/V1_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjFalse.nii.gz
    mask_path = Path(f'derivatives/sub-{subject}/roi_pipeline/{roi}_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjFalse.nii.gz')

    # Alternative path 1: timestamp-based roi_pipeline (old format)
    if not mask_path.exists():
        mask_path = Path(f'derivatives/{timestamp}/roi_pipeline/{roi}_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjFalse.nii.gz')

    # Alternative path 2: BH2009 directory with subject-specific naming
    if not mask_path.exists():
        mask_path = Path(f'derivatives/BH2009/{timestamp}/sub-{subject}_{roi}_mask.nii.gz')

    # Alternative path 3: Generic roi_masks directory
    if not mask_path.exists():
        mask_path = Path(f'derivatives/roi_masks/sub-{subject}_{roi}_mask.nii.gz')

    return str(mask_path)
