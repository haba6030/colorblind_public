#!/usr/bin/env python3
"""
Utility Functions for Model Comparison

Shared helper functions used across all scripts.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase3_decoder_comparing/model_comparison_validation/scripts/utils.py  (development repository, commit 53c81c2)
# Role      : Decoder definitions and hyper-parameter defaults shared by the baselines.
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


import numpy as np
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

HC_SUBJECTS = [f"{i:02d}" for i in range(1, 8)]
CVD_SUBJECTS = [f"{i:02d}" for i in range(8, 11)]
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS

ROIS = ['V1', 'V2', 'V3', 'V4']

HUE_ANGLES = [i * 45 for i in range(8)]  # 0, 45, 90, ..., 315


# ============================================================================
# Circular Math
# ============================================================================

def circular_diff_deg(hue1, hue2):
    """
    Compute circular difference in degrees

    Args:
        hue1, hue2: Hue angles in degrees (0-360)

    Returns:
        diff: Difference in degrees, range [-180, 180]
    """
    diff = hue1 - hue2
    diff = np.mod(diff + 180, 360) - 180
    return diff


def labels_to_hue(labels):
    """
    Convert labels (0-7) to hue angles (0, 45, ..., 315)

    Args:
        labels: Array of color labels (0-7)

    Returns:
        hues: Array of hue angles in degrees
    """
    return np.array([HUE_ANGLES[int(l)] for l in labels])


def hue_to_labels(hue_angles):
    """
    Convert hue angles to nearest labels (0-7)

    Args:
        hue_angles: Array of hue angles in degrees

    Returns:
        labels: Array of nearest color labels (0-7)
    """
    labels = []
    for hue in hue_angles:
        diffs = [abs(circular_diff_deg(hue, target_hue)) for target_hue in HUE_ANGLES]
        labels.append(np.argmin(diffs))
    return np.array(labels)


# ============================================================================
# Data Loading
# ============================================================================

def load_amplitudes(baseline_dir, subject, roi, alignment='raw'):
    """
    Load amplitudes from full_dataset_C010 structure

    Args:
        baseline_dir: Path to full_dataset_C010
        subject: Subject ID (e.g., '01')
        roi: ROI name (e.g., 'V1')
        alignment: 'raw', 'procrustes', or 'srm'

    Returns:
        amplitudes: (n_runs=6, n_colors=8, n_features) array
    """
    subject_roi_dir = Path(baseline_dir) / f"sub-{subject}" / roi

    if alignment == 'raw':
        amp_path = subject_roi_dir / "amplitudes_raw.npy"
    elif alignment == 'procrustes':
        amp_path = subject_roi_dir / "amplitudes_procrustes.npy"
    elif alignment == 'srm':
        amp_path = subject_roi_dir / "amplitudes_srm.npy"
    else:
        raise ValueError(f"Unknown alignment: {alignment}. Use 'raw', 'procrustes', or 'srm'")

    if not amp_path.exists():
        raise FileNotFoundError(f"Amplitudes not found: {amp_path}")

    return np.load(amp_path)


# ============================================================================
# Statistics
# ============================================================================

def bootstrap_ci(data, n_bootstrap=1000, ci_percentiles=[2.5, 97.5]):
    """
    Compute bootstrap confidence interval

    Args:
        data: Array of values
        n_bootstrap: Number of bootstrap iterations
        ci_percentiles: Percentiles for CI (e.g., [2.5, 97.5] for 95% CI)

    Returns:
        mean: Mean of data
        ci_lower: Lower CI bound
        ci_upper: Upper CI bound
    """
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    mean = np.mean(data)
    ci_lower = np.percentile(bootstrap_means, ci_percentiles[0])
    ci_upper = np.percentile(bootstrap_means, ci_percentiles[1])

    return mean, ci_lower, ci_upper


def spearman_brown_correction(r):
    """
    Spearman-Brown correction for split-half reliability

    Args:
        r: Split-half correlation

    Returns:
        r_corrected: Full-length reliability
    """
    if (1 + r) <= 0:
        return 0
    return 2 * r / (1 + r)


# ============================================================================
# Model Type Classification
# ============================================================================

def is_linear_model(model_name):
    """Check if model is linear"""
    linear_models = ['LDA', 'Ridge', 'ForwardEncoding']
    return model_name in linear_models


def is_nonlinear_model(model_name):
    """Check if model is non-linear"""
    nonlinear_models = ['KernelRidge', 'SVM', 'MLP', 'FE_MLP', 'FE_SVM',
                        'HybridMLP', 'HybridSVR']
    return model_name in nonlinear_models


def uses_labels(model_name):
    """Check if model uses discrete labels (vs continuous hue)"""
    label_models = ['LDA', 'SVM', 'MLP', 'ForwardEncoding', 'FE_MLP', 'FE_SVM',
                    'HybridMLP', 'HybridSVR',
                    'FE_PopVec', 'FE_RidgeEnc', 'FE_GaussML', 'FE_RidgeReg',
                    'FE_Ensemble', 'FE_EnsembleRidge', 'FE_EnsembleGaussML']
    return model_name in label_models


# ============================================================================
# Chance Levels
# ============================================================================

def get_chance_level(metric='acc_exact'):
    """
    Get chance level for a given metric

    Args:
        metric: Metric name

    Returns:
        chance: Chance level (0-1 for accuracy, degrees for MAE)
    """
    chance_levels = {
        'acc_exact': 1/8,      # 12.5% (8 colors)
        'acc_45': 3/8,         # 37.5% (3 out of 8 within 45°)
        'acc_90': 5/8,         # 62.5% (5 out of 8 within 90°)
        'mae': 90,             # 90 degrees (random guess)
        'medae': 90
    }

    return chance_levels.get(metric, 0)


# ============================================================================
# Summary Statistics
# ============================================================================

def compute_summary_stats(values):
    """
    Compute comprehensive summary statistics

    Args:
        values: Array of values

    Returns:
        stats_dict: Dictionary with mean, std, median, range, etc.
    """
    values = np.array(values)

    return {
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'range': [float(np.min(values)), float(np.max(values))],
        'n': len(values)
    }


# ============================================================================
# Group Classification
# ============================================================================

def get_subject_group(subject):
    """
    Get group (HC or CVD) for a subject

    Args:
        subject: Subject ID (e.g., '01')

    Returns:
        group: 'HC' or 'CVD'
    """
    return 'HC' if subject in HC_SUBJECTS else 'CVD'


def filter_by_group(subjects, group):
    """
    Filter subjects by group

    Args:
        subjects: List of subject IDs
        group: 'HC' or 'CVD'

    Returns:
        filtered: List of subjects in that group
    """
    if group == 'HC':
        return [s for s in subjects if s in HC_SUBJECTS]
    elif group == 'CVD':
        return [s for s in subjects if s in CVD_SUBJECTS]
    else:
        return subjects


# ============================================================================
# Model Architecture & Logging Helpers
# ============================================================================

def get_model_architecture(model_name):
    """
    Return architecture metadata for a given model.

    Args:
        model_name: One of 'LDA', 'Ridge', 'KernelRidge', 'SVM', 'MLP', 'ForwardEncoding'

    Returns:
        dict with type, target, description, linearity
    """
    architectures = {
        'LDA': {
            'type': 'classifier',
            'target': 'labels (0-7)',
            'linearity': 'linear',
            'description': 'Linear Discriminant Analysis with shrinkage. '
                           'Finds linear projection maximizing between-class variance.'
        },
        'Ridge': {
            'type': 'regression',
            'target': 'circular hue (sin/cos encoding)',
            'linearity': 'linear',
            'description': 'Ridge regression predicting sin(hue) and cos(hue). '
                           'L2 regularization controls overfitting.'
        },
        'KernelRidge': {
            'type': 'regression',
            'target': 'circular hue (sin/cos encoding)',
            'linearity': 'nonlinear',
            'description': 'Kernel Ridge with RBF kernel. '
                           'Non-linear extension of Ridge via kernel trick.'
        },
        'SVM': {
            'type': 'classifier',
            'target': 'labels (0-7)',
            'linearity': 'nonlinear',
            'description': 'Support Vector Machine with RBF kernel. '
                           'Non-linear decision boundaries in voxel space.'
        },
        'MLP': {
            'type': 'classifier',
            'target': 'labels (0-7)',
            'linearity': 'nonlinear',
            'description': 'Multi-Layer Perceptron with ReLU activation, '
                           'early stopping, and L2 regularization.'
        },
        'ForwardEncoding': {
            'type': 'encoding_model',
            'target': 'labels (0-7) via 6-channel basis',
            'linearity': 'linear',
            'description': 'Brouwer & Heeger 2009 forward encoding model. '
                           '6 idealized color channels, analytical solution.'
        },
        'FE_MLP': {
            'type': 'hybrid',
            'target': 'labels (0-7) via 6-channel + MLP',
            'linearity': 'nonlinear readout',
            'description': 'Stage 1: ForwardEncoding extracts 6 channel responses. '
                           'Stage 2: MLP classifies from 6-dim channel space. '
                           'Tests nonlinearity in channel-to-color mapping.'
        },
        'FE_SVM': {
            'type': 'hybrid',
            'target': 'labels (0-7) via 6-channel + SVM',
            'linearity': 'nonlinear readout',
            'description': 'Stage 1: ForwardEncoding extracts 6 channel responses. '
                           'Stage 2: SVM-RBF classifies from 6-dim channel space.'
        },
        'HybridMLP': {
            'type': 'hybrid_degree',
            'target': 'continuous hue (0-359°) via MLP→6-channel→template',
            'linearity': 'nonlinear encoder',
            'description': 'Stage 1: MLPRegressor maps voxels to 6 channel activations. '
                           'Stage 2: FE template matching selects best hue (0-359°). '
                           'Reverses standard FE→MLP hybrid: nonlinear encoding + linear decoding.'
        },
        'HybridSVR': {
            'type': 'hybrid_degree',
            'target': 'continuous hue (0-359°) via SVR→6-channel→template',
            'linearity': 'nonlinear encoder',
            'description': 'Stage 1: MultiOutput SVR maps voxels to 6 channel activations. '
                           'Stage 2: FE template matching selects best hue (0-359°). '
                           'SVR variant of the degree-based hybrid model.'
        },
        'FE_PopVec': {
            'type': 'encoding_model',
            'target': 'continuous hue (0-359°) via population vector',
            'linearity': 'linear',
            'description': 'FE encoding + population vector decoding. '
                           'Circular weighted mean of 6 channel centers by response magnitude.'
        },
        'FE_RidgeEnc': {
            'type': 'encoding_model',
            'target': 'continuous hue (0-359°) via Ridge + correlation',
            'linearity': 'linear',
            'description': 'FE with Ridge-regularized encoding weights (alpha=1.0). '
                           'Same correlation-based template matching as baseline FE.'
        },
        'FE_GaussML': {
            'type': 'encoding_model',
            'target': 'continuous hue (0-359°) via Gaussian ML',
            'linearity': 'linear',
            'description': 'FE encoding + Gaussian maximum likelihood decoding. '
                           'Uses per-channel noise variance estimated from training residuals.'
        },
        'FE_RidgeReg': {
            'type': 'encoding_model',
            'target': 'continuous hue (0-359°) via Ridge regression',
            'linearity': 'linear',
            'description': 'FE encoding + Ridge regression from 6 channels to sin/cos hue. '
                           'Learned channel-to-hue mapping instead of template matching.'
        },
        'FE_Ensemble': {
            'type': 'encoding_model',
            'target': 'continuous hue (0-359°) via per-run W ensemble',
            'linearity': 'linear',
            'description': 'Per-run W estimation (6 separate W matrices) + correlation-based '
                           'template matching per W + circular mean of ensemble predictions.'
        },
        'FE_EnsembleRidge': {
            'type': 'encoding_model',
            'target': 'continuous hue (0-359°) via per-run Ridge W ensemble',
            'linearity': 'linear',
            'description': 'Per-run W with Ridge regularization (alpha=1.0) + correlation-based '
                           'template matching per W + circular mean of ensemble predictions.'
        },
        'FE_EnsembleGaussML': {
            'type': 'encoding_model',
            'target': 'continuous hue (0-359°) via per-run W + Gaussian ML',
            'linearity': 'linear',
            'description': 'Per-run W estimation + within-color noise variance (from run-to-run '
                           'variability) + Gaussian ML decoding on mean channel responses.'
        },
        'HybridMLP_Ensemble': {
            'type': 'hybrid_ensemble',
            'target': 'continuous hue (0-359°) via per-run MLP→6-channel→template',
            'linearity': 'nonlinear encoder + ensemble',
            'description': 'Per-run MLP regression (7 colors → 6 channel activations), '
                           'then FE template matching per MLP, circular mean of 6 predictions. '
                           'Tests per-run diversity to overcome MLP overfitting.'
        },
        'HybridSVR_Ensemble': {
            'type': 'hybrid_ensemble',
            'target': 'continuous hue (0-359°) via per-run SVR→6-channel→template',
            'linearity': 'nonlinear encoder + ensemble',
            'description': 'Per-run MultiOutput SVR (7 colors → 6 channel activations), '
                           'then FE template matching per SVR, circular mean of 6 predictions. '
                           'Tests per-run diversity to overcome SVR overfitting.'
        },
        'FE_Sequential': {
            'type': 'encoding_model_sequential',
            'target': 'continuous hue (0-359°) via incremental FE',
            'linearity': 'linear',
            'description': 'One W matrix trained sequentially via incremental data accumulation '
                           '(run1 → run1+2 → ... → all 6 runs). Final W identical to pooled (42 samples), '
                           'but learning process mimics continuous brain updates.'
        },
        'HybridMLP_Sequential': {
            'type': 'hybrid_sequential',
            'target': 'continuous hue (0-359°) via FE(pooled) + MLP(warm_start)',
            'linearity': 'nonlinear readout + sequential',
            'description': 'Stage 1: FE with pooled data (stable encoding). '
                           'Stage 2: MLP with warm_start=True trains sequentially on run1, run2, ..., run6. '
                           'Mimics continuous readout plasticity with stable representation.'
        },
        'HybridSVR_Sequential': {
            'type': 'hybrid_sequential',
            'target': 'continuous hue (0-359°) via FE(pooled) + SVR(incremental)',
            'linearity': 'nonlinear readout + sequential',
            'description': 'Stage 1: FE with pooled data (stable encoding). '
                           'Stage 2: SVR trains incrementally (run1 → run1+2 → ... → all runs). '
                           'SVR variant of sequential hybrid learning.'
        }
    }
    return architectures.get(model_name, {'type': 'unknown', 'target': 'unknown',
                                           'linearity': 'unknown', 'description': ''})


def get_model_defaults(model_name):
    """
    Return default hyperparameters used in LOCO (no HP tuning).

    Args:
        model_name: Model name string

    Returns:
        dict with default param values and rationale
    """
    defaults = {
        'LDA': {
            'params': {'solver': 'lsqr', 'shrinkage': 'auto'},
            'rationale': 'Ledoit-Wolf automatic shrinkage; robust for small samples.'
        },
        'Ridge': {
            'params': {'alpha': 1.0},
            'rationale': 'Moderate regularization; standard default.'
        },
        'KernelRidge': {
            'params': {'alpha': 1.0, 'gamma': 0.01},
            'rationale': 'Conservative gamma to avoid overfitting in high-dim voxel space.'
        },
        'SVM': {
            'params': {'C': 1.0, 'gamma': 0.01},
            'rationale': 'Default C=1; conservative gamma for fMRI dimensionality.'
        },
        'MLP': {
            'params': {'hidden_layer_sizes': (64,), 'alpha': 0.1},
            'rationale': 'Single hidden layer; strong L2 to prevent overfitting with n=42.'
        },
        'ForwardEncoding': {
            'params': {'alpha': 0, 'n_channels': 6},
            'rationale': 'Pseudoinverse solution (alpha=0); 6 channels per B&H 2009.'
        },
        'FE_MLP': {
            'params': {'fe_alpha': 0, 'n_channels': 6,
                       'hidden_layer_sizes': (16,), 'mlp_alpha': 0.01},
            'rationale': 'FE stage 1 (alpha=0) + MLP stage 2 (16 units, mild L2).'
        },
        'FE_SVM': {
            'params': {'fe_alpha': 0, 'n_channels': 6, 'C': 1.0, 'gamma': 'scale'},
            'rationale': 'FE stage 1 (alpha=0) + SVM-RBF stage 2 (auto-scaled gamma).'
        },
        'HybridMLP': {
            'params': {'n_channels': 6, 'hidden_layer_sizes': (64, 32), 'alpha': 0.1},
            'rationale': 'MLPRegressor to 6 channels + FE template matching. '
                         'Two hidden layers, moderate L2.'
        },
        'HybridSVR': {
            'params': {'n_channels': 6, 'C': 1.0, 'epsilon': 0.1},
            'rationale': 'MultiOutput SVR to 6 channels + FE template matching. '
                         'RBF kernel, auto-scaled gamma.'
        },
        'FE_PopVec': {
            'params': {'alpha': 0, 'n_channels': 6},
            'rationale': 'Pseudoinverse encoding + population vector decoding (circular weighted mean).'
        },
        'FE_RidgeEnc': {
            'params': {'alpha': 1.0, 'n_channels': 6},
            'rationale': 'Ridge-regularized encoding (alpha=1.0) + correlation template matching.'
        },
        'FE_GaussML': {
            'params': {'alpha': 0, 'n_channels': 6},
            'rationale': 'Pseudoinverse encoding + Gaussian ML decoding with per-channel noise variance.'
        },
        'FE_RidgeReg': {
            'params': {'alpha': 0, 'n_channels': 6, 'ridge_alpha': 1.0},
            'rationale': 'Pseudoinverse encoding + Ridge regression from channels to sin/cos hue.'
        },
        'FE_Ensemble': {
            'params': {'alpha': 0, 'n_channels': 6},
            'rationale': 'Per-run pseudoinverse W (6 independent W matrices) + correlation '
                         'template matching + circular mean ensemble.'
        },
        'FE_EnsembleRidge': {
            'params': {'alpha': 1.0, 'n_channels': 6},
            'rationale': 'Per-run Ridge W (alpha=1.0) + correlation template matching + '
                         'circular mean ensemble.'
        },
        'FE_EnsembleGaussML': {
            'params': {'alpha': 0, 'n_channels': 6},
            'rationale': 'Per-run pseudoinverse W + within-color noise estimation + '
                         'Gaussian ML decoding on mean channel responses.'
        },
        'HybridMLP_Ensemble': {
            'params': {'n_channels': 6, 'hidden_layer_sizes': (64, 32), 'alpha': 0.1},
            'rationale': 'Per-run MLP (7 colors/run → 6 channels) + FE template matching + '
                         'circular mean ensemble. Tests per-run diversity vs pooled overfitting.'
        },
        'HybridSVR_Ensemble': {
            'params': {'n_channels': 6, 'C': 1.0, 'epsilon': 0.1},
            'rationale': 'Per-run MultiOutput SVR (7 colors/run → 6 channels) + FE template matching + '
                         'circular mean ensemble. Tests per-run diversity vs pooled overfitting.'
        },
        'FE_Sequential': {
            'params': {'alpha': 0, 'n_channels': 6},
            'rationale': 'Pseudoinverse encoding with incremental training (run1 → run1+2 → ... → all). '
                         'Final W identical to pooled (42 samples), sequential learning process.'
        },
        'HybridMLP_Sequential': {
            'params': {'fe_alpha': 0, 'n_channels': 6,
                       'hidden_layer_sizes': (64, 32), 'mlp_alpha': 0.1},
            'rationale': 'FE pooled (alpha=0) + MLP warm_start sequential (run1→run2→...→run6). '
                         'Two hidden layers (64, 32), moderate L2 regularization.'
        },
        'HybridSVR_Sequential': {
            'params': {'fe_alpha': 0, 'n_channels': 6, 'C': 1.0, 'epsilon': 0.1},
            'rationale': 'FE pooled (alpha=0) + SVR incremental (run1 → run1+2 → ... → all). '
                         'MultiOutput SVR with RBF kernel, auto-scaled gamma.'
        }
    }
    return defaults.get(model_name, {'params': {}, 'rationale': 'unknown model'})


def aggregate_best_params(fold_results):
    """
    Aggregate hyperparameter selections across LORO folds.

    Args:
        fold_results: List of dicts, each with 'best_params' key

    Returns:
        dict with per-param selection counts and most-frequent choice
    """
    from collections import Counter

    if not fold_results or 'best_params' not in fold_results[0]:
        return {}

    # Collect all param keys
    all_params = {}
    for fold in fold_results:
        bp = fold.get('best_params', {})
        for k, v in bp.items():
            all_params.setdefault(k, []).append(str(v))

    summary = {}
    for param_name, values in all_params.items():
        counts = Counter(values)
        most_common = counts.most_common(1)[0] if counts else (None, 0)
        summary[param_name] = {
            'selection_counts': dict(counts),
            'most_frequent': most_common[0],
            'n_folds': len(values)
        }

    return summary


if __name__ == '__main__':
    # Test utilities
    print("Utility Functions Test")
    print("="*80)

    # Test circular math
    print("\nCircular Math:")
    print(f"  circular_diff_deg(10, 350) = {circular_diff_deg(10, 350)}°  (expected: 20°)")
    print(f"  circular_diff_deg(350, 10) = {circular_diff_deg(350, 10)}°  (expected: -20°)")

    # Test label/hue conversion
    print("\nLabel/Hue Conversion:")
    labels = np.array([0, 2, 4, 6])
    hues = labels_to_hue(labels)
    print(f"  labels_to_hue([0,2,4,6]) = {hues}")
    print(f"  hue_to_labels({hues}) = {hue_to_labels(hues)}")

    # Test chance levels
    print("\nChance Levels:")
    for metric in ['acc_exact', 'acc_45', 'acc_90', 'mae']:
        print(f"  {metric}: {get_chance_level(metric)}")

    # Test group classification
    print("\nGroup Classification:")
    print(f"  get_subject_group('01') = {get_subject_group('01')}")
    print(f"  get_subject_group('08') = {get_subject_group('08')}")

    print("\n" + "="*80)
