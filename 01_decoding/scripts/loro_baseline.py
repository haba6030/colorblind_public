#!/usr/bin/env python3
"""
LORO Baseline Decoder Comparison

Compares decoding models on full_dataset_C010 using LORO cross-validation:
1. LDA (Linear Discriminant Analysis)
2. Ridge Regression - Linear with circular encoding
3. Kernel Ridge - Non-linear with RBF
4. SVM - Support Vector Machine
5. MLP - Multi-Layer Perceptron
6. Forward Encoding - 6-channel model (pooled W)
7. FE_MLP - FE + MLP hybrid
8. FE_SVM - FE + SVM hybrid

Protocol:
- LORO (Leave-One-Run-Out) cross-validation
- Nested hyperparameter tuning
- Before/After Procrustes alignment comparison
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase3_decoder_comparing/model_comparison_validation/scripts/loro_baseline.py  (development repository, commit 53c81c2)
# Role      : Six-decoder LORO comparison (S5, tab:loro_decoders). Output: results/loro/{srm,procrustes}/sub-*_performance_raw.json
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

# ML libraries
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge as SKLearnKernelRidge
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import f_classif
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.linalg import orthogonal_procrustes

# Import from project utils (avoid shadowing by local utils.py)
project_root = Path(__file__).resolve().parents[4]
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "utils_color_decoding",
    project_root / "analysis" / "utils" / "utils_color_decoding.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
create_basis_functions = _mod.create_basis_functions
circular_diff_deg = _mod.circular_diff_deg

warnings.filterwarnings('ignore')

# Import logging helpers from local utils
from utils import (get_model_architecture, aggregate_best_params,
                   get_subject_group)


# ============================================================================
# Configuration
# ============================================================================

HC_SUBJECTS = [f"{i:02d}" for i in range(1, 8)]   # sub-01 ~ sub-07
CVD_SUBJECTS = [f"{i:02d}" for i in range(8, 11)]  # sub-08 ~ sub-10
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS

ROIS = ['V1', 'V2', 'V3', 'V4']

# Color mapping (8 colors, 45° spacing)
LABEL2HUE_DEG = {
    'color_1': 0,      # Red
    'color_2': 45,     # Orange
    'color_3': 90,     # Yellow
    'color_4': 135,    # Green
    'color_5': 180,    # Cyan
    'color_6': 225,    # Blue
    'color_7': 270,    # Purple
    'color_8': 315     # Magenta
}

HUE_ANGLES = [LABEL2HUE_DEG[f'color_{i+1}'] for i in range(8)]


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
        raise ValueError(f"Unknown alignment: {alignment}")

    if not amp_path.exists():
        raise FileNotFoundError(f"Amplitudes not found: {amp_path}")

    amplitudes = np.load(amp_path)
    return amplitudes


def _normalize_run(pattern):
    """Center and scale a single run pattern (n_colors, n_voxels)."""
    centered = pattern - pattern.mean()
    scale = np.std(centered)
    if scale > 1e-10:
        return centered / scale
    return centered


def procrustes_nested_fold(amplitudes_raw, train_indices, test_index):
    """Fit Procrustes on train runs only, apply to test run.

    1. Normalize each train run (center + scale by std)
    2. Compute train reference = mean of normalized train runs
    3. Align each train run to reference via orthogonal_procrustes
    4. Normalize test run (center + scale by its own stats)
    5. Align test run to train reference via orthogonal_procrustes

    Args:
        amplitudes_raw: (n_runs, n_colors, n_voxels) raw amplitudes
        train_indices: list of train run indices
        test_index: single test run index

    Returns:
        train_aligned: (len(train_indices), n_colors, n_voxels)
        test_aligned: (n_colors, n_voxels)
    """
    n_colors, n_voxels = amplitudes_raw.shape[1], amplitudes_raw.shape[2]

    # Normalize train runs
    train_normalized = np.zeros((len(train_indices), n_colors, n_voxels))
    for i, r in enumerate(train_indices):
        train_normalized[i] = _normalize_run(amplitudes_raw[r])

    # Train reference = mean of normalized train runs
    train_ref = train_normalized.mean(axis=0)  # (n_colors, n_voxels)

    # Align each train run to reference
    train_aligned = np.zeros_like(train_normalized)
    for i in range(len(train_indices)):
        R, _ = orthogonal_procrustes(train_normalized[i].T, train_ref.T)
        train_aligned[i] = (train_normalized[i].T @ R).T

    # Normalize and align test run
    test_normalized = _normalize_run(amplitudes_raw[test_index])
    R_test, _ = orthogonal_procrustes(test_normalized.T, train_ref.T)
    test_aligned = (test_normalized.T @ R_test).T

    return train_aligned, test_aligned


# ============================================================================
# Decoder Models
# ============================================================================

class LDADecoder:
    """Linear Discriminant Analysis - Current baseline"""

    def __init__(self, solver='lsqr', shrinkage='auto'):
        self.solver = solver
        self.shrinkage = shrinkage
        self.model = None

    def fit(self, X, y_labels):
        """
        Args:
            X: (n_samples, n_voxels) brain patterns
            y_labels: (n_samples,) color labels (0-7)
        """
        # Adjust shrinkage for solver
        if self.solver == 'svd':
            shrinkage = None
        else:
            shrinkage = self.shrinkage

        self.model = LinearDiscriminantAnalysis(
            solver=self.solver,
            shrinkage=shrinkage
        )
        self.model.fit(X, y_labels)

    def predict(self, X):
        """Returns: (n_samples,) predicted labels (0-7)"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        return self.model.predict(X)

    @staticmethod
    def get_param_grid():
        return {
            'solver': ['svd', 'lsqr'],
            'shrinkage': [None, 'auto', 0.5]
        }


class RidgeDecoder:
    """Ridge Regression with circular hue encoding"""

    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.model = None

    def fit(self, X, y_hue):
        """
        Args:
            X: (n_samples, n_voxels)
            y_hue: (n_samples,) hue angles in degrees
        """
        # Convert to sin/cos for continuity
        y_sin = np.sin(np.deg2rad(y_hue))
        y_cos = np.cos(np.deg2rad(y_hue))
        y_circular = np.column_stack([y_sin, y_cos])

        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y_circular)

    def predict(self, X):
        """Returns: (n_samples,) predicted hue angles"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")

        y_pred_circular = self.model.predict(X)
        y_pred_hue = np.rad2deg(np.arctan2(y_pred_circular[:, 0], y_pred_circular[:, 1]))
        y_pred_hue = np.mod(y_pred_hue, 360)

        return y_pred_hue

    @staticmethod
    def get_param_grid():
        return {'alpha': [0.01, 0.1, 1, 10, 100]}


class KernelRidgeDecoder:
    """Kernel Ridge with RBF kernel"""

    def __init__(self, alpha=1.0, gamma=0.01):
        self.alpha = alpha
        self.gamma = gamma
        self.model = None

    def fit(self, X, y_hue):
        """
        Args:
            X: (n_samples, n_voxels)
            y_hue: (n_samples,) hue angles
        """
        # Convert to sin/cos
        y_sin = np.sin(np.deg2rad(y_hue))
        y_cos = np.cos(np.deg2rad(y_hue))
        y_circular = np.column_stack([y_sin, y_cos])

        self.model = SKLearnKernelRidge(
            alpha=self.alpha,
            gamma=self.gamma,
            kernel='rbf'
        )
        self.model.fit(X, y_circular)

    def predict(self, X):
        """Returns: (n_samples,) predicted hue angles"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")

        y_pred_circular = self.model.predict(X)
        y_pred_hue = np.rad2deg(np.arctan2(y_pred_circular[:, 0], y_pred_circular[:, 1]))
        y_pred_hue = np.mod(y_pred_hue, 360)

        return y_pred_hue

    @staticmethod
    def get_param_grid():
        return {
            'alpha': [0.1, 1, 10],
            'gamma': [0.0001, 0.001, 0.01, 0.1]
        }


class SVMDecoder:
    """Support Vector Machine with RBF kernel"""

    def __init__(self, C=1.0, gamma=0.01):
        self.C = C
        self.gamma = gamma
        self.model = None

    def fit(self, X, y_labels):
        """
        Args:
            X: (n_samples, n_voxels)
            y_labels: (n_samples,) color labels (0-7)
        """
        self.model = SVC(
            C=self.C,
            gamma=self.gamma,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        self.model.fit(X, y_labels)

    def predict(self, X):
        """Returns: (n_samples,) predicted labels (0-7)"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        return self.model.predict(X)

    @staticmethod
    def get_param_grid():
        return {
            'C': [0.1, 1, 10],
            'gamma': [0.001, 0.01, 0.1]
        }


class MLPDecoder:
    """Multi-Layer Perceptron for classification"""

    def __init__(self, hidden_layer_sizes=(64,), alpha=0.1):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.model = None

    def fit(self, X, y_labels):
        """
        Args:
            X: (n_samples, n_voxels)
            y_labels: (n_samples,) color labels (0-7)
        """
        self.model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.alpha,
            activation='relu',
            solver='adam',
            learning_rate='adaptive',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            verbose=False
        )
        self.model.fit(X, y_labels)

    def predict(self, X):
        """Returns: (n_samples,) predicted labels (0-7)"""
        if self.model is None:
            raise RuntimeError("Model not fitted yet")
        return self.model.predict(X)

    @staticmethod
    def get_param_grid():
        return {
            'hidden_layer_sizes': [(64,), (64, 32)],
            'alpha': [0.01, 0.1]
        }


class ForwardEncodingDecoder:
    """6-channel forward encoding model"""

    def __init__(self, alpha=0, n_channels=6):
        self.alpha = alpha
        self.n_channels = n_channels
        self.weights = None
        self.basis_functions = None

    def fit(self, X, y_labels):
        """
        Fit encoding weights from POOLED trials (not run-averaged).

        Args:
            X: (n_samples, n_voxels) — e.g., (40, n_voxels) for 5 train runs × 8 colors
            y_labels: (n_samples,) color labels (0-7)
        """
        # Create basis functions: (360, n_channels) full
        basis_full = create_basis_functions(n_channels=self.n_channels)  # (360, n_channels)
        self.basis_functions = basis_full[HUE_ANGLES]  # (8, n_channels) — for predict

        # Get hue for EACH trial → basis function per trial (pooled)
        hues = np.array([HUE_ANGLES[int(l)] for l in y_labels])
        C = basis_full[hues]  # (n_samples, n_channels)
        B = X                 # (n_samples, n_voxels)

        # Encoding weights: W = (C^T C + αI)^-1 C^T B
        if self.alpha > 0:
            self.weights = np.linalg.solve(
                C.T @ C + self.alpha * np.eye(self.n_channels),
                C.T @ B
            )
        else:
            self.weights = np.linalg.pinv(C) @ B

    def predict(self, X):
        """Returns: (n_samples,) predicted labels (0-7)"""
        if self.weights is None:
            raise RuntimeError("Model not fitted yet")

        # Predict channel responses
        channel_responses = self.weights @ X.T  # (n_channels, n_samples)

        # Find best matching color
        n_samples = X.shape[0]
        y_pred_labels = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            predicted_response = channel_responses[:, i]  # (n_channels,)
            # Dot product with each color's channel template → (n_colors,)
            correlations = self.basis_functions @ predicted_response  # (8, n_channels) @ (n_channels,) = (8,)
            y_pred_labels[i] = np.argmax(correlations)

        return y_pred_labels

    @staticmethod
    def get_param_grid():
        return {'alpha': [0, 10, 50]}


class FEMLPHybridDecoder:
    """ForwardEncoding (6-channel) + MLP readout.

    Stage 1: ForwardEncoding extracts 6 channel responses (stable, interpretable)
    Stage 2: MLP classifies from 6-dim channel space (can capture nonlinearity)

    With only 6 features, MLP cannot overfit (sample/feature = 40/6 ≈ 6.7:1).
    """

    def __init__(self, fe_alpha=0, n_channels=6,
                 hidden_layer_sizes=(16,), mlp_alpha=0.01):
        self.fe_alpha = fe_alpha
        self.n_channels = n_channels
        self.hidden_layer_sizes = hidden_layer_sizes
        self.mlp_alpha = mlp_alpha
        self.fe_model = None
        self.mlp_model = None

    def fit(self, X, y_labels):
        """
        Args:
            X: (n_samples, n_voxels)
            y_labels: (n_samples,) color labels (0-7)
        """
        # Stage 1: Fit ForwardEncoding to get W
        self.fe_model = ForwardEncodingDecoder(
            alpha=self.fe_alpha, n_channels=self.n_channels)
        self.fe_model.fit(X, y_labels)

        # Extract channel responses for training data
        C_train = self.fe_model.weights @ X.T  # (n_channels, n_samples)

        # Stage 2: MLP on channel responses
        self.mlp_model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            alpha=self.mlp_alpha,
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2,
            random_state=42,
            verbose=False
        )
        self.mlp_model.fit(C_train.T, y_labels)

    def predict(self, X):
        """Returns: (n_samples,) predicted labels (0-7)"""
        if self.fe_model is None or self.mlp_model is None:
            raise RuntimeError("Model not fitted yet")

        C_test = self.fe_model.weights @ X.T  # (n_channels, n_samples)
        return self.mlp_model.predict(C_test.T)

    @staticmethod
    def get_param_grid():
        return {
            'fe_alpha': [0, 10],
            'hidden_layer_sizes': [(16,), (16, 8)],
            'mlp_alpha': [0.01, 0.1]
        }


class FESVMHybridDecoder:
    """ForwardEncoding (6-channel) + SVM readout.

    Stage 1: ForwardEncoding extracts 6 channel responses
    Stage 2: SVM with RBF kernel classifies from 6-dim channel space
    """

    def __init__(self, fe_alpha=0, n_channels=6, C=1.0, gamma='scale'):
        self.fe_alpha = fe_alpha
        self.n_channels = n_channels
        self.C_param = C
        self.gamma = gamma
        self.fe_model = None
        self.svm_model = None

    def fit(self, X, y_labels):
        """
        Args:
            X: (n_samples, n_voxels)
            y_labels: (n_samples,) color labels (0-7)
        """
        # Stage 1: Fit ForwardEncoding
        self.fe_model = ForwardEncodingDecoder(
            alpha=self.fe_alpha, n_channels=self.n_channels)
        self.fe_model.fit(X, y_labels)

        # Extract channel responses
        C_train = self.fe_model.weights @ X.T  # (n_channels, n_samples)

        # Stage 2: SVM on channel responses
        self.svm_model = SVC(
            C=self.C_param, gamma=self.gamma,
            kernel='rbf', random_state=42
        )
        self.svm_model.fit(C_train.T, y_labels)

    def predict(self, X):
        """Returns: (n_samples,) predicted labels (0-7)"""
        if self.fe_model is None or self.svm_model is None:
            raise RuntimeError("Model not fitted yet")

        C_test = self.fe_model.weights @ X.T
        return self.svm_model.predict(C_test.T)

    @staticmethod
    def get_param_grid():
        return {
            'fe_alpha': [0, 10],
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.1, 1.0]
        }


# ============================================================================
# Metrics
# ============================================================================

def labels_to_hue(labels):
    """Convert labels (0-7) to hue angles (0-360)"""
    return np.array([HUE_ANGLES[int(l)] for l in labels])


def hue_to_labels(hue_angles):
    """Convert hue angles to nearest labels (0-7)"""
    labels = []
    for hue in hue_angles:
        # Find nearest color
        diffs = [abs(circular_diff_deg(hue, target_hue)) for target_hue in HUE_ANGLES]
        labels.append(np.argmin(diffs))
    return np.array(labels)


def compute_classification_metrics(y_true_labels, y_pred_labels):
    """
    Compute classification accuracy at different thresholds

    Args:
        y_true_labels: (n_samples,) true labels (0-7)
        y_pred_labels: (n_samples,) predicted labels (0-7)

    Returns:
        metrics: Dict with acc_exact, acc_45, acc_90
    """
    # Convert to hue for circular distance
    y_true_hue = labels_to_hue(y_true_labels)
    y_pred_hue = labels_to_hue(y_pred_labels)

    errors = np.abs(circular_diff_deg(y_true_hue, y_pred_hue))

    metrics = {
        'acc_exact': float(np.mean(y_true_labels == y_pred_labels)),
        'acc_45': float(np.mean(errors <= 45)),
        'acc_90': float(np.mean(errors <= 90)),
        'mae': float(np.mean(errors)),
        'medae': float(np.median(errors))
    }

    return metrics


def compute_reconstruction_metrics(y_true_hue, y_pred_hue):
    """
    Compute circular reconstruction error metrics.

    Args:
        y_true_hue: (n_samples,) true hue angles
        y_pred_hue: (n_samples,) predicted hue angles

    Returns:
        metrics: Dict with mae, medae, errors
    """
    errors = np.abs(circular_diff_deg(y_true_hue, y_pred_hue))
    return {
        'mae': float(np.mean(errors)),
        'medae': float(np.median(errors)),
        'errors': errors.tolist()
    }


# ============================================================================
# LORO Cross-Validation
# ============================================================================

def _apply_dim_reduction_fit(X_train, y_train_labels, dim_reduction, dim_k, n_colors):
    """Fit dimensionality reduction on training data.

    Returns:
        X_train_reduced, reducer_state (tuple for applying to test)
    """
    if dim_reduction == 'anova' and dim_k is not None:
        labels_anova = y_train_labels if y_train_labels is not None else np.tile(np.arange(n_colors), len(X_train) // n_colors)
        f_scores, _ = f_classif(X_train, labels_anova)
        k_actual = min(dim_k, X_train.shape[1])  # guard: can't select more than available
        top_k_idx = np.sort(np.argsort(f_scores)[::-1][:k_actual])
        return X_train[:, top_k_idx], ('anova', top_k_idx)
    elif dim_reduction == 'pca' and dim_k is not None:
        k_actual = min(dim_k, X_train.shape[1], X_train.shape[0])
        pca = PCA(n_components=k_actual)
        X_reduced = pca.fit_transform(X_train)
        return X_reduced, ('pca', pca)
    return X_train, None


def _apply_dim_reduction_transform(X_test, reducer_state):
    """Apply fitted dim reduction to test data."""
    if reducer_state is None:
        return X_test
    method, obj = reducer_state
    if method == 'anova':
        return X_test[:, obj]
    elif method == 'pca':
        return obj.transform(X_test)
    return X_test


def loro_cv_generic(amplitudes, model_class, model_name, param_grid=None,
                    alignment='preloaded', dim_reduction=None, dim_k=None):
    """
    Generic LORO CV for any decoder model

    Args:
        amplitudes: (n_runs, n_colors, n_voxels) array
        model_class: Decoder class
        model_name: Model name string
        param_grid: Optional parameter grid for tuning
        alignment: 'preloaded' (use as-is) or 'nested_procrustes' (fit per fold)
        dim_reduction: None, 'anova', or 'pca'
        dim_k: Number of features/components for dim reduction

    Returns:
        fold_results: List of dicts with performance per fold
    """
    n_runs, n_colors, n_voxels = amplitudes.shape
    labels = np.arange(n_colors)
    hue_angles = labels_to_hue(labels)

    fold_results = []

    # Determine if model uses labels or hue
    uses_labels = model_name in ['LDA', 'SVM', 'MLP', 'ForwardEncoding',
                                  'FE_MLP', 'FE_SVM']
    # Models that output continuous hues (0-359°) instead of discrete labels
    outputs_continuous = False

    for test_run in range(n_runs):
        # Split train/test
        train_runs = [r for r in range(n_runs) if r != test_run]

        # --- RT-2: Nested Procrustes ---
        if alignment == 'nested_procrustes':
            train_aligned, test_aligned = procrustes_nested_fold(
                amplitudes, train_runs, test_run)
        else:
            train_aligned = amplitudes[train_runs]
            test_aligned = amplitudes[test_run]

        n_features = train_aligned.shape[-1]
        X_train = train_aligned.reshape(-1, n_features)
        X_test = test_aligned  # (n_colors, n_features)

        if uses_labels:
            y_train = np.tile(labels, len(train_runs))
            y_test = labels
        else:
            y_train = np.tile(hue_angles, len(train_runs))
            y_test = hue_angles

        # --- RT-3: Dimensionality reduction (within fold) ---
        y_train_labels = np.tile(labels, len(train_runs))
        X_train, reducer_state = _apply_dim_reduction_fit(
            X_train, y_train_labels, dim_reduction, dim_k, n_colors)
        X_test = _apply_dim_reduction_transform(X_test, reducer_state)

        # Hyperparameter tuning (simple grid search)
        best_params = {}
        best_score = -np.inf

        if param_grid is not None and len(train_runs) >= 3:
            # Simple inner CV: leave-one-out on train runs
            for params in _generate_param_combinations(param_grid):
                tune_scores = []

                # Inner loop uses train_aligned with relative indexing (0..4)
                for inner_test_idx in range(len(train_runs)):
                    inner_train_idx = [i for i in range(len(train_runs)) if i != inner_test_idx]

                    if alignment == 'nested_procrustes':
                        # Re-do nested Procrustes for inner fold
                        inner_train_abs = [train_runs[i] for i in inner_train_idx]
                        inner_test_abs = train_runs[inner_test_idx]
                        inner_train_al, inner_test_al = procrustes_nested_fold(
                            amplitudes, inner_train_abs, inner_test_abs)
                    else:
                        inner_train_al = train_aligned[inner_train_idx]
                        inner_test_al = train_aligned[inner_test_idx]

                    n_feat_inner = inner_train_al.shape[-1]
                    X_tune_train = inner_train_al.reshape(-1, n_feat_inner)
                    X_tune_test = inner_test_al

                    if uses_labels:
                        y_tune_train = np.tile(labels, len(inner_train_idx))
                        y_tune_test = labels
                    else:
                        y_tune_train = np.tile(hue_angles, len(inner_train_idx))
                        y_tune_test = hue_angles

                    # Apply dim reduction inside inner loop
                    y_tune_train_labels = np.tile(labels, len(inner_train_idx))
                    X_tune_train, inner_reducer = _apply_dim_reduction_fit(
                        X_tune_train, y_tune_train_labels, dim_reduction, dim_k, n_colors)
                    X_tune_test = _apply_dim_reduction_transform(X_tune_test, inner_reducer)

                    # Train and evaluate
                    model_tune = model_class(**params)
                    model_tune.fit(X_tune_train, y_tune_train)
                    y_tune_pred = model_tune.predict(X_tune_test)

                    # Score (accuracy for label-based, MAE for hue-based)
                    if uses_labels:
                        if outputs_continuous:
                            y_tune_pred_labels = hue_to_labels(y_tune_pred)
                            score = accuracy_score(y_tune_test, y_tune_pred_labels)
                        else:
                            score = accuracy_score(y_tune_test, y_tune_pred)
                    else:
                        y_tune_pred_labels = hue_to_labels(y_tune_pred)
                        y_tune_test_labels = hue_to_labels(y_tune_test)
                        score = accuracy_score(y_tune_test_labels, y_tune_pred_labels)

                    tune_scores.append(score)

                mean_score = np.mean(tune_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params

        # Train final model with best params
        model = model_class(**best_params) if best_params else model_class()
        model.fit(X_train, y_train)

        # Test on held-out run
        y_pred = model.predict(X_test)

        # Convert predictions to labels if needed
        if uses_labels:
            if outputs_continuous:
                y_pred_labels = hue_to_labels(y_pred)
            else:
                y_pred_labels = y_pred
        else:
            y_pred_labels = hue_to_labels(y_pred)

        # Compute metrics
        metrics = compute_classification_metrics(labels, y_pred_labels)

        fold_results.append({
            'test_run': test_run,
            'best_params': best_params,
            **metrics
        })

    return fold_results


def _generate_param_combinations(param_grid):
    """Generate all combinations of parameters from grid"""
    from itertools import product

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    for combo in product(*values):
        yield dict(zip(keys, combo))


# ============================================================================
# Main Comparison
# ============================================================================

def run_single_subject_roi(baseline_dir, subject, roi, alignment, models,
                           dim_reduction=None, dim_k=None):
    """
    Run model comparison for a single subject-ROI pair

    Args:
        baseline_dir: Path to full_dataset_C010
        subject: Subject ID (e.g., '01')
        roi: ROI name (e.g., 'V1')
        alignment: 'raw', 'procrustes', or 'nested_procrustes'
        models: List of model names to compare
        dim_reduction: None, 'anova', or 'pca'
        dim_k: Number of features/components for dim reduction

    Returns:
        results: Dict with results per model
    """
    print(f"\n{'='*60}")
    print(f"Subject: sub-{subject} | ROI: {roi} | Alignment: {alignment}")
    if dim_reduction:
        print(f"  Dim reduction: {dim_reduction} (k={dim_k})")
    print(f"{'='*60}")

    # Load amplitudes
    try:
        if alignment == 'nested_procrustes':
            # Load raw data — Procrustes will be applied per fold
            amplitudes = load_amplitudes(baseline_dir, subject, roi, 'raw')
            loro_alignment = 'nested_procrustes'
        else:
            amplitudes = load_amplitudes(baseline_dir, subject, roi, alignment)
            loro_alignment = 'preloaded'
        print(f"Loaded amplitudes: {amplitudes.shape}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return None

    results = {}

    # Model mapping
    model_map = {
        'LDA': (LDADecoder, LDADecoder.get_param_grid()),
        'Ridge': (RidgeDecoder, RidgeDecoder.get_param_grid()),
        'KernelRidge': (KernelRidgeDecoder, KernelRidgeDecoder.get_param_grid()),
        'SVM': (SVMDecoder, SVMDecoder.get_param_grid()),
        'MLP': (MLPDecoder, MLPDecoder.get_param_grid()),
        'ForwardEncoding': (ForwardEncodingDecoder, ForwardEncodingDecoder.get_param_grid()),
        'FE_MLP': (FEMLPHybridDecoder, FEMLPHybridDecoder.get_param_grid()),
        'FE_SVM': (FESVMHybridDecoder, FESVMHybridDecoder.get_param_grid())
    }

    # Run each model
    for model_name in models:
        if model_name not in model_map:
            print(f"Unknown model: {model_name}")
            continue

        print(f"\n--- Running {model_name} ---")

        try:
            model_class, param_grid = model_map[model_name]
            fold_results = loro_cv_generic(
                amplitudes, model_class, model_name, param_grid,
                alignment=loro_alignment,
                dim_reduction=dim_reduction,
                dim_k=dim_k
            )

            results[model_name] = fold_results

            # Print summary
            acc_45_values = [f['acc_45'] for f in fold_results]
            mae_values = [f['mae'] for f in fold_results]

            print(f"  Accuracy (45°): {np.mean(acc_45_values):.3f} ± {np.std(acc_45_values):.3f}")
            print(f"  MAE: {np.mean(mae_values):.2f}° ± {np.std(mae_values):.2f}°")

        except Exception as e:
            print(f"ERROR in {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extended decoder model comparison (6 models)"
    )
    parser.add_argument('--baseline_dir', type=str, required=True,
                       help='Path to full_dataset_C010')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--subject', type=str, required=True,
                       help='Subject ID (e.g., 01)')
    parser.add_argument('--rois', nargs='+', default=['V1', 'V2', 'V3', 'V4'],
                       help='ROI names')
    parser.add_argument('--models', nargs='+',
                       default=['LDA', 'Ridge', 'KernelRidge', 'SVM', 'MLP',
                                'ForwardEncoding', 'FE_MLP', 'FE_SVM'],
                       help='Models to compare')
    parser.add_argument('--alignment', type=str, default='both',
                       choices=['raw', 'procrustes', 'srm', 'nested_procrustes', 'both'],
                       help='Alignment condition')
    parser.add_argument('--dim_reduction', type=str, default=None,
                       choices=['anova', 'pca'],
                       help='Dimensionality reduction method (within fold)')
    parser.add_argument('--dim_k', type=int, default=None,
                       help='Number of features (ANOVA) or components (PCA)')

    args = parser.parse_args()

    # Create output directory (flat — no timestamp subdirs)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"Extended Decoder Model Comparison")
    print(f"{'='*80}")
    print(f"Baseline: {args.baseline_dir}")
    print(f"Subject: sub-{args.subject}")
    print(f"ROIs: {args.rois}")
    print(f"Models: {args.models}")
    print(f"Alignment: {args.alignment}")
    print(f"Output: {output_path}")
    print(f"{'='*80}\n")

    # Determine alignments to test
    alignments = ['raw', 'procrustes'] if args.alignment == 'both' else [args.alignment]

    # Run for each alignment and ROI
    all_results = {}

    for alignment in alignments:
        all_results[alignment] = {}

        for roi in args.rois:
            results = run_single_subject_roi(
                args.baseline_dir,
                args.subject,
                roi,
                alignment,
                args.models,
                dim_reduction=args.dim_reduction,
                dim_k=args.dim_k
            )

            if results is not None:
                all_results[alignment][roi] = results

    # Save results
    output_file = output_path / f"sub-{args.subject}_performance_raw.json"

    # Build enhanced logging: hyperparameters and architectures
    hyperparameters = {}
    model_architectures = {}
    for model_name in args.models:
        model_architectures[model_name] = get_model_architecture(model_name)

        # Aggregate HP selections from procrustes results (or first alignment)
        hp_alignment = 'procrustes' if 'procrustes' in all_results else alignments[0]
        model_hp = {}
        for roi in args.rois:
            roi_results = all_results.get(hp_alignment, {}).get(roi, {})
            fold_list = roi_results.get(model_name, [])
            if fold_list:
                model_hp[roi] = aggregate_best_params(fold_list)
        if model_hp:
            hyperparameters[model_name] = model_hp

    results_data = {
        'subject': args.subject,
        'subject_group': get_subject_group(args.subject),
        'rois': args.rois,
        'models': args.models,
        'alignments': alignments,
        'settings': {
            'baseline_dir': str(args.baseline_dir),
            'dataset_name': Path(args.baseline_dir).name,
            'n_runs': 6,
            'n_colors': 8,
            'cv_method': 'LORO (Leave-One-Run-Out)',
            'hp_tuning': True,
            'hp_tuning_method': 'nested CV (inner LORO on train runs)',
            'dim_reduction': args.dim_reduction,
            'dim_k': args.dim_k
        },
        'hyperparameters': hyperparameters,
        'model_architectures': model_architectures,
        'results': all_results,
        'timestamp': timestamp,
        'datetime': datetime.now().isoformat()
    }

    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Save config.json (one per output_dir, safe to overwrite — identical across subjects)
    config_file = output_path / 'config.json'
    config_data = {
        'description': 'LORO model comparison',
        'baseline_dir': str(args.baseline_dir),
        'dataset_name': Path(args.baseline_dir).name,
        'alignments': alignments,
        'models': args.models,
        'rois': args.rois,
        'dim_reduction': args.dim_reduction,
        'dim_k': args.dim_k,
        'cv_method': 'LORO (Leave-One-Run-Out)',
        'hp_tuning': True,
        'n_runs': 6,
        'n_colors': 8,
        'created': datetime.now().isoformat()
    }
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved: {output_file}")
    print(f"Config saved: {config_file}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
