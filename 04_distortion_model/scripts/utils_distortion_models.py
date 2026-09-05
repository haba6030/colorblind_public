#!/usr/bin/env python3
"""
utils_distortion_models.py — Unified interface for Gen-4 hue distortion models.

Gen-4 active models:
  machado_1way       (df=1): Machado 2009 Eq 5/6, α coupled to Δλ
  machado_alpha_free (df=2): Machado 2009 Eq 5/6, independent Δλ and α
  cone_3way          (df=3): Independent L, M, S cone shifts in nm (ablation)

Legacy Fourier / per-color distortions that depend on `utils_filter` have been
removed — Gen-4 does not use them and the module is not shipped with the
cone_shift_pipeline tree on the server.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/utils_distortion_models.py  (development repository, commit 53c81c2)
# Role      : Distortion application helpers shared by both model classes.
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
import sys
from pathlib import Path

# Add forward model scripts to path
_PHASE2_DIR = Path(__file__).resolve().parent.parent
for _base in [_PHASE2_DIR.parent, _PHASE2_DIR.parent.parent]:
    _fwd = _base / 'phase4_forward_model' / 'scripts'
    if _fwd.exists() and str(_fwd) not in sys.path:
        sys.path.insert(0, str(_fwd))
        break

from utils_forward_model import create_basis_full, HUE_ANGLES, N_CHANNELS
try:
    from utils_cone_3way import compute_shifted_hue_3way
except ModuleNotFoundError:
    def compute_shifted_hue_3way(*args, **kwargs):  # pragma: no cover
        raise ModuleNotFoundError(
            "utils_cone_3way not installed; 'cone_3way' model unavailable. "
            "Use machado_1way / 2component / 3component / rc_opponent instead.")

# Gen-4 Machado simulator (physiologically anchored cone shift)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from machado_simulator import machado_shifted_hue, DELTA_LAMBDA_MAX

# ============================================================================
# Constants
# ============================================================================

HUE_ANGLES_FLOAT = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)

# Sign convention:
#   deutan: M-cone shifted +delta_lambda (toward L, longer wavelengths)
#   protan: L-cone shifted -delta_lambda (toward M, shorter wavelengths)
# Direction is handled by cvd_type inside the shift functions.

MODELS = {
    # --- Gen-4 (Machado-anchored, primary) ---
    'machado_1way': {
        'df': 1,
        'bounds': [(0.0, DELTA_LAMBDA_MAX)],
        'x0': [5.0],
        'description': 'Machado 2009 Eq 5/6 — Δλ only, α coupled',
    },
    'machado_alpha_free': {
        'df': 2,
        'bounds': [(0.0, DELTA_LAMBDA_MAX), (0.0, 1.0)],
        'x0': [5.0, 0.75],
        'description': 'Machado 2009 Eq 5/6 — independent Δλ and α (2-DOF)',
    },
    # --- Gen-3 ablation only (kept to reproduce the Gen-3 failure mode) ---
    'cone_3way': {
        'df': 3,
        'bounds': [(-60, 60), (-60, 60), (-60, 60)],
        'x0': [0.0, 10.0, 0.0],  # default: M-cone shift for deutan
        'description': 'Independent L, M, S cone shifts (nm) [Gen-3 ablation]',
    },
}


# ============================================================================
# Core Functions
# ============================================================================

def apply_distortion(model_name, params, hue_angles=None, cvd_type='deutan'):
    """Apply distortion model to hue angles.

    Args:
        model_name: one of MODELS keys
        params: parameter array matching model df
        hue_angles: (8,) base hue angles in degrees (default: standard 8 colors)
        cvd_type: 'deutan' or 'protan' (for cone models)

    Returns:
        shifted_angles: (8,) shifted hue angles in degrees [0, 360)
    """
    if hue_angles is None:
        hue_angles = HUE_ANGLES_FLOAT.copy()
    params = np.asarray(params, dtype=float)

    if model_name == 'machado_1way':
        delta_lambda = float(params[0])
        _, hue_shifted, _ = machado_shifted_hue(delta_lambda, cvd_type)
        return hue_shifted % 360

    elif model_name == 'machado_alpha_free':
        delta_lambda = float(params[0])
        alpha = float(params[1])
        _, hue_shifted, _ = machado_shifted_hue(
            delta_lambda, cvd_type, alpha=alpha)
        return hue_shifted % 360

    elif model_name == 'cone_3way':
        delta_L, delta_M, delta_S = params
        _, hue_shifted, _ = compute_shifted_hue_3way(delta_L, delta_M, delta_S)
        return hue_shifted % 360

    else:
        raise ValueError(f'Unknown model: {model_name}')


def get_design_matrix(model_name, params, n_channels=N_CHANNELS, basis_type='fe',
                      cvd_type='deutan'):
    """Get C(theta + delta_theta) design matrix for shifted angles.

    Args:
        model_name: distortion model name
        params: model parameters
        n_channels: number of basis channels (default: 6)
        basis_type: 'fe' or 'lf'
        cvd_type: CVD type for cone models

    Returns:
        C_shifted: (8, K) design matrix at shifted hue angles
    """
    shifted_angles = apply_distortion(model_name, params, cvd_type=cvd_type)
    basis_full = create_basis_full(n_channels, basis_type=basis_type)
    idx = np.round(shifted_angles).astype(int) % 360
    return basis_full[idx]


def get_delta_theta(model_name, params, cvd_type='deutan'):
    """Get per-color hue shift delta_theta (degrees).

    Args:
        model_name: distortion model name
        params: model parameters
        cvd_type: CVD type

    Returns:
        delta_theta: (8,) hue shifts in degrees, wrapped to [-180, 180]
    """
    shifted = apply_distortion(model_name, params, cvd_type=cvd_type)
    delta = shifted - HUE_ANGLES_FLOAT
    return (delta + 180) % 360 - 180


def get_initial_params(model_name, cvd_type='deutan'):
    """Get sensible initial parameters for optimization.

    Args:
        model_name: model name
        cvd_type: CVD type (affects cone_3way init)

    Returns:
        x0: initial parameter array
    """
    x0 = np.array(MODELS[model_name]['x0'], dtype=float)
    if model_name == 'machado_1way':
        x0 = np.array([5.0])
    elif model_name == 'machado_alpha_free':
        x0 = np.array([5.0, 0.75])
    elif model_name == 'cone_3way':
        if cvd_type == 'deutan':
            x0 = np.array([0.0, 10.0, 0.0])  # M-cone shift
        elif cvd_type == 'protan':
            x0 = np.array([-10.0, 0.0, 0.0])  # L-cone shift
        else:
            x0 = np.array([0.0, 0.0, 0.0])
    return x0


def compute_aicc(rss, df, n_obs=28):
    """Compute corrected AIC for model comparison.

    Args:
        rss: residual sum of squares
        df: number of free parameters
        n_obs: number of observations (28 for RDM upper triangle)

    Returns:
        aicc: corrected AIC value (lower = better)
    """
    if rss <= 0 or n_obs - df - 1 <= 0:
        return np.nan
    aic = n_obs * np.log(rss / n_obs) + 2 * df
    correction = 2 * df * (df + 1) / (n_obs - df - 1)
    return aic + correction
