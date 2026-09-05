#!/usr/bin/env python3
"""
stockman_cone_shift.py — Approach A: Physics-based cone shift quantification.

Uses Stockman & Sharpe (2000) cone fundamentals to model CVD hue distortion
as a cone peak wavelength shift (Δλ, nm).

For each CVD subject:
  1. Shift M-cone (deutan) or L-cone (protan) sensitivity by Δλ
  2. Compute predicted hue angles for 8 experimental colors under shifted cones
  3. Fit Δλ* via grid search against observed LOCO per-color voxel_corr gap
  4. Extract Fourier coefficients (a₁,b₁,a₂,b₂) from δθ(θ) for T_ψ₀ initialization

Runs locally (conda env: srm).
Prereq: pip install colour-science   (optional; CVRL CSV fallback available)

Usage:
    python scripts/stockman_cone_shift.py \
        --loco_dir results/validation \
        --output_dir results/cone_shift
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase4_forward_model/scripts/stockman_cone_shift.py  (development repository, commit 53c81c2)
# Role      : Stockman & Sharpe cone fundamentals and the spectral-shift computation used by the Machado simulator.
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
from datetime import datetime
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

# ============================================================================
# Constants
# ============================================================================

HC_SUBJECTS = [f'{i:02d}' for i in range(1, 8)]
CVD_SUBJECTS = ['08', '09', '10']

# CVD type assignment (from behavioral data)
CVD_TYPE = {
    '08': 'deutan',   # M-cone shift
    '09': 'protan',   # L-cone shift
    '10': 'normal',   # compensated, minimal shift expected
}

# 8 experimental colors in CIELab
# L*=75, chroma=40 matches actual experiment (colorBlind_test.py)
HUE_ANGLES_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)
L_STAR = 75.0
CHROMA = 40.0

# CIELab coordinates
COLOR_LAB = np.zeros((8, 3))
for i, h in enumerate(HUE_ANGLES_DEG):
    h_rad = np.deg2rad(h)
    COLOR_LAB[i] = [L_STAR, CHROMA * np.cos(h_rad), CHROMA * np.sin(h_rad)]

# D65 white point (CIE 1931 2-degree)
D65_XYZ = np.array([0.95047, 1.00000, 1.08883])

COLOR_NAMES = ['red', 'orange', 'yellow', 'green',
               'cyan', 'blue', 'purple', 'magenta']


# ============================================================================
# Color Space Conversions
# ============================================================================

def _lab_f_inv(t):
    """Inverse of CIELab f function."""
    delta = 6.0 / 29.0
    mask = t > delta
    result = np.where(mask, t ** 3, 3 * delta ** 2 * (t - 4.0 / 29.0))
    return result


def lab_to_xyz(lab, white=D65_XYZ):
    """CIELab → XYZ (D65 illuminant)."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = a / 500.0 + fy
    fz = fy - b / 200.0
    X = white[0] * _lab_f_inv(fx)
    Y = white[1] * _lab_f_inv(fy)
    Z = white[2] * _lab_f_inv(fz)
    return np.stack([X, Y, Z], axis=-1)


# ============================================================================
# Stockman Cone Fundamentals
# ============================================================================

def load_stockman_fundamentals():
    """Load Stockman & Sharpe (2000) 2-degree cone fundamentals.

    Returns:
        wavelengths: (N,) array of wavelengths in nm
        L, M, S: (N,) arrays of cone sensitivities (linear, energy basis)
    """
    try:
        import colour
        cmfs = colour.colorimetry.MSDS_CMFS_LMS[
            'Stockman & Sharpe 2 Degree Cone Fundamentals'
        ]
        wavelengths = cmfs.wavelengths
        L = cmfs.values[:, 0]
        M = cmfs.values[:, 1]
        S = cmfs.values[:, 2]
        return wavelengths, L, M, S
    except (ImportError, KeyError):
        pass

    # Fallback: generate approximate fundamentals from CIE 2006 LMS
    # using the standard peak wavelengths and bandwidth model
    print("[WARN] colour-science not found. Using approximate cone fundamentals.")
    wavelengths = np.arange(390, 781, 1.0)
    # Approximate log-Gaussian cone fundamentals (Stockman peaks)
    L = _approx_cone(wavelengths, peak=566.0, sigma_l=32.0, sigma_r=65.0)
    M = _approx_cone(wavelengths, peak=541.0, sigma_l=27.0, sigma_r=55.0)
    S = _approx_cone(wavelengths, peak=441.0, sigma_l=22.0, sigma_r=30.0)
    return wavelengths, L, M, S


def _approx_cone(wl, peak, sigma_l, sigma_r):
    """Approximate cone sensitivity with asymmetric Gaussian."""
    sigma = np.where(wl < peak, sigma_l, sigma_r)
    return np.exp(-0.5 * ((wl - peak) / sigma) ** 2)


def compute_xyz_to_lms_matrix(wavelengths, L, M, S):
    """Compute XYZ → LMS transformation matrix.

    Uses numerical integration: LMS = M_xyz2lms @ XYZ
    where M_xyz2lms is derived from CIE 1931 color matching functions.
    """
    try:
        import colour
        # Use the standard Hunt-Pointer-Estevez or CIECAM02 matrix
        M_cat = colour.adaptation.matrix_chromatic_adaptation_VonKries(
            colour.xy_to_XYZ(colour.CCS_ILLUMINANTS[
                'CIE 1931 2 Degree Standard Observer']['D65'][:2]),
            colour.xy_to_XYZ(colour.CCS_ILLUMINANTS[
                'CIE 1931 2 Degree Standard Observer']['D65'][:2]),
            transform='CAT02'
        )
        # For cone fundamentals: use Stockman-Sharpe directly
        # The XYZ→LMS matrix from Stockman & Sharpe (2000)
        M_lms = np.array([
            [ 0.38971, 0.68898, -0.07868],
            [-0.22981, 1.18340,  0.04641],
            [ 0.00000, 0.00000,  1.00000],
        ])
        return M_lms
    except ImportError:
        # Standard Stockman & Sharpe (2000) XYZ→LMS matrix
        M_lms = np.array([
            [ 0.38971, 0.68898, -0.07868],
            [-0.22981, 1.18340,  0.04641],
            [ 0.00000, 0.00000,  1.00000],
        ])
        return M_lms


def shift_cone_sensitivity(wavelengths, cone_vals, delta_lambda):
    """Shift cone sensitivity by Δλ nm along wavelength axis.

    Args:
        wavelengths: (N,) original wavelengths
        cone_vals: (N,) original cone sensitivity values
        delta_lambda: shift in nm (positive = shift to longer wavelengths)

    Returns:
        shifted: (N,) shifted cone sensitivity (cubic spline interpolation)
    """
    cs = CubicSpline(wavelengths, cone_vals, extrapolate=True)
    shifted = cs(wavelengths - delta_lambda)
    shifted = np.maximum(shifted, 0.0)
    return shifted


# ============================================================================
# Hue Angle Computation
# ============================================================================

def lms_to_opponent(lms):
    """LMS → opponent channels.

    Args:
        lms: (..., 3) LMS values

    Returns:
        rg: L-M (red-green)
        by: S - (L+M)/2 (blue-yellow)
    """
    L, M, S = lms[..., 0], lms[..., 1], lms[..., 2]
    rg = L - M
    by = S - (L + M) / 2.0
    return rg, by


def opponent_to_hue_angle(rg, by):
    """Opponent channels → hue angle in degrees [0, 360)."""
    theta = np.rad2deg(np.arctan2(by, rg))
    theta = theta % 360.0
    return theta


def compute_hue_angles_for_colors(M_xyz2lms, colors_lab=COLOR_LAB):
    """Compute hue angles for experimental colors given XYZ→LMS matrix.

    Args:
        M_xyz2lms: (3,3) XYZ→LMS matrix
        colors_lab: (8,3) CIELab coordinates

    Returns:
        hue_angles: (8,) hue angles in degrees
    """
    xyz = lab_to_xyz(colors_lab)  # (8, 3)
    lms = (M_xyz2lms @ xyz.T).T  # (8, 3)
    rg, by = lms_to_opponent(lms)
    return opponent_to_hue_angle(rg, by)


def compute_shifted_hue_angles(wavelengths, L, M, S, delta_lambda,
                               cvd_type='deutan'):
    """Compute hue angles under shifted cone sensitivity.

    Both normal and shifted use the same spectral-level computation
    (_compute_spectral_shift_lms) to ensure Δλ=0 → δθ=0 exactly.

    Args:
        wavelengths, L, M, S: cone fundamentals
        delta_lambda: shift magnitude in nm
        cvd_type: 'deutan' (shift M) or 'protan' (shift L)

    Returns:
        hue_normal: (8,) normal hue angles
        hue_shifted: (8,) shifted hue angles
        delta_theta: (8,) hue angle differences (shifted - normal)
    """
    M_xyz2lms = compute_xyz_to_lms_matrix(wavelengths, L, M, S)
    xyz_colors = lab_to_xyz(COLOR_LAB)  # (8, 3)

    # Normal: use spectral-level computation with UNSHIFTED cones
    lms_normal = _compute_spectral_shift_lms(
        wavelengths, L, M, S, M_xyz2lms, xyz_colors
    )
    rg_n, by_n = lms_to_opponent(lms_normal)
    hue_normal = opponent_to_hue_angle(rg_n, by_n)

    # Shifted: apply cone shift then same computation
    if cvd_type == 'deutan':
        M_shifted = shift_cone_sensitivity(wavelengths, M, delta_lambda)
        lms_shifted = _compute_spectral_shift_lms(
            wavelengths, L, M_shifted, S, M_xyz2lms, xyz_colors
        )
    elif cvd_type == 'protan':
        L_shifted = shift_cone_sensitivity(wavelengths, L, -delta_lambda)
        lms_shifted = _compute_spectral_shift_lms(
            wavelengths, L_shifted, M, S, M_xyz2lms, xyz_colors
        )
    else:
        lms_shifted = lms_normal.copy()

    rg_s, by_s = lms_to_opponent(lms_shifted)
    hue_shifted = opponent_to_hue_angle(rg_s, by_s)

    # Circular difference
    delta_theta = hue_shifted - hue_normal
    delta_theta = (delta_theta + 180) % 360 - 180  # wrap to [-180, 180]

    return hue_normal, hue_shifted, delta_theta


def _compute_spectral_shift_lms(wavelengths, L_vals, M_vals, S_vals,
                                M_xyz2lms_orig, xyz_colors):
    """Compute LMS responses with spectrally shifted cone fundamentals.

    Uses the relationship between XYZ→LMS and cone fundamentals to
    recompute the mapping with shifted cones.

    For display-rendered stimuli (CIELab→XYZ), we need to go through
    the spectral domain to properly account for cone shifts.

    Simplified approach: recompute XYZ→LMS matrix from shifted fundamentals.
    """
    try:
        import colour
        cmfs_xyz = colour.colorimetry.MSDS_CMFS[
            'CIE 1931 2 Degree Standard Observer'
        ]
        # Interpolate XYZ CMFs to match our wavelength grid
        x_bar = np.interp(wavelengths, cmfs_xyz.wavelengths, cmfs_xyz.values[:, 0])
        y_bar = np.interp(wavelengths, cmfs_xyz.wavelengths, cmfs_xyz.values[:, 1])
        z_bar = np.interp(wavelengths, cmfs_xyz.wavelengths, cmfs_xyz.values[:, 2])
    except ImportError:
        # Approximate CIE 1931 CMFs using Gaussian model
        x_bar = (1.056 * _approx_cone(wavelengths, 599.8, 37.9, 31.0) +
                 0.362 * _approx_cone(wavelengths, 442.0, 16.0, 26.7) -
                 0.065 * _approx_cone(wavelengths, 501.1, 20.4, 26.2))
        y_bar = (0.821 * _approx_cone(wavelengths, 568.8, 46.9, 40.5) +
                 0.286 * _approx_cone(wavelengths, 530.9, 16.3, 31.1))
        z_bar = (1.217 * _approx_cone(wavelengths, 437.0, 11.8, 36.0) +
                 0.681 * _approx_cone(wavelengths, 459.0, 26.0, 13.8))

    # Build XYZ CMF matrix: (N_wl, 3)
    XYZ_cmf = np.column_stack([x_bar, y_bar, z_bar])

    # Build LMS CMF matrix with shifted cones: (N_wl, 3)
    LMS_cmf = np.column_stack([L_vals, M_vals, S_vals])

    # Compute XYZ→LMS via least-squares: LMS_cmf ≈ XYZ_cmf @ M.T
    # M_new = (XYZ_cmf.T @ XYZ_cmf)^{-1} @ XYZ_cmf.T @ LMS_cmf
    M_new = np.linalg.lstsq(XYZ_cmf, LMS_cmf, rcond=None)[0].T  # (3, 3)

    # Apply to stimulus XYZ
    lms = (M_new @ xyz_colors.T).T  # (8, 3)
    return lms


# ============================================================================
# Grid Search
# ============================================================================

def load_loco_per_color(loco_dir, subject, model='ridge_gcv', roi='V4'):
    """Load per-color LOCO voxel_corr for a subject.

    Args:
        loco_dir: directory with sub-XX_loco.json files
        subject: subject ID (e.g., '08')
        model: encoder model name
        roi: ROI key in JSON

    Returns:
        per_color_corr: (8,) voxel_corr per color
    """
    path = Path(loco_dir) / f'sub-{subject}_loco.json'
    with open(path) as f:
        data = json.load(f)

    folds = data[roi][model]['folds']
    per_color = np.zeros(8)
    for fold in folds:
        idx = fold['test_color']
        per_color[idx] = fold['voxel_corr']
    return per_color


def compute_hc_mean_loco_per_color(loco_dir, model='ridge_gcv', roi='V4'):
    """Compute HC mean per-color LOCO voxel_corr."""
    all_corrs = []
    for subj in HC_SUBJECTS:
        path = Path(loco_dir) / f'sub-{subj}_loco.json'
        if not path.exists():
            continue
        corr = load_loco_per_color(loco_dir, subj, model, roi)
        all_corrs.append(corr)
    return np.mean(all_corrs, axis=0)  # (8,)


def grid_search_delta_lambda(wavelengths, L, M, S, delta_obs, cvd_type,
                             max_shift=40, step=1):
    """Grid search for optimal Δλ.

    Args:
        wavelengths, L, M, S: cone fundamentals
        delta_obs: (8,) observed hue distortion metric (per-color)
        cvd_type: 'deutan' or 'protan'
        max_shift: maximum shift in nm
        step: grid step size in nm

    Returns:
        best_delta: optimal Δλ in nm
        best_delta_theta: (8,) predicted hue shifts at optimal Δλ
        grid_results: dict of Δλ → {mse, spearman_r, delta_theta}
    """
    from scipy.stats import spearmanr

    grid = np.arange(0, max_shift + step, step)
    grid_results = {}
    best_mse = np.inf
    best_delta = 0
    best_delta_theta = np.zeros(8)

    for dl in grid:
        _, _, dt = compute_shifted_hue_angles(
            wavelengths, L, M, S, dl, cvd_type
        )
        # Normalize both to compare shapes (not magnitudes)
        # Use rank correlation for robustness
        if np.std(dt) > 0 and np.std(delta_obs) > 0:
            rho, p_rho = spearmanr(dt, delta_obs)
        else:
            rho, p_rho = 0.0, 1.0

        # Also compute MSE on normalized profiles
        if np.std(dt) > 0:
            dt_norm = (dt - dt.mean()) / dt.std()
            obs_norm = (delta_obs - delta_obs.mean()) / delta_obs.std()
            mse = float(np.mean((dt_norm - obs_norm) ** 2))
        else:
            mse = np.inf

        grid_results[float(dl)] = {
            'mse': mse,
            'spearman_r': float(rho) if np.isfinite(rho) else 0.0,
            'spearman_p': float(p_rho) if np.isfinite(p_rho) else 1.0,
            'delta_theta': dt.tolist(),
        }

        if mse < best_mse:
            best_mse = mse
            best_delta = float(dl)
            best_delta_theta = dt.copy()

    return best_delta, best_delta_theta, grid_results


# ============================================================================
# Fourier Fit for T_ψ₀
# ============================================================================

def fourier_fit_delta_theta(hue_angles_rad, delta_theta_rad):
    """Fit Fourier k=1,2 to δθ(θ) for T_ψ₀ initialization.

    T_ψ(θ) = θ + a₁cos(θ) + b₁sin(θ) + a₂cos(2θ) + b₂sin(2θ)
    δθ(θ) ≈ a₁cos(θ) + b₁sin(θ) + a₂cos(2θ) + b₂sin(2θ)

    Args:
        hue_angles_rad: (8,) hue angles in radians
        delta_theta_rad: (8,) hue shifts in radians

    Returns:
        coeffs: dict {a1, b1, a2, b2} in radians
    """
    # Design matrix: [cos(θ), sin(θ), cos(2θ), sin(2θ)]
    X = np.column_stack([
        np.cos(hue_angles_rad),
        np.sin(hue_angles_rad),
        np.cos(2 * hue_angles_rad),
        np.sin(2 * hue_angles_rad),
    ])

    # OLS fit
    beta = np.linalg.lstsq(X, delta_theta_rad, rcond=None)[0]

    return {
        'a1': float(beta[0]),
        'b1': float(beta[1]),
        'a2': float(beta[2]),
        'b2': float(beta[3]),
    }


def apply_T_psi(theta_deg, coeffs):
    """Apply T_ψ transform: θ' = θ + ψ(θ).

    Args:
        theta_deg: angle(s) in degrees
        coeffs: {a1, b1, a2, b2} in radians

    Returns:
        theta_corrected_deg: corrected angle(s) in degrees
    """
    theta_rad = np.deg2rad(theta_deg)
    psi = (coeffs['a1'] * np.cos(theta_rad) +
           coeffs['b1'] * np.sin(theta_rad) +
           coeffs['a2'] * np.cos(2 * theta_rad) +
           coeffs['b2'] * np.sin(2 * theta_rad))
    return (theta_deg + np.rad2deg(psi)) % 360


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Stockman cone shift analysis (Approach A)'
    )
    parser.add_argument('--loco_dir', type=str,
                        default='results/validation',
                        help='Directory with per-subject LOCO JSON files')
    parser.add_argument('--output_dir', type=str,
                        default='results/cone_shift',
                        help='Output directory')
    parser.add_argument('--roi', type=str, default='V4',
                        help='ROI key in LOCO JSON (V4 = hV4)')
    parser.add_argument('--model', type=str, default='ridge_gcv',
                        help='Encoder model name')
    parser.add_argument('--max_shift', type=int, default=40,
                        help='Maximum Δλ in nm for grid search')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load cone fundamentals
    print("[1] Loading Stockman & Sharpe (2000) cone fundamentals...")
    wavelengths, L, M, S = load_stockman_fundamentals()
    print(f"    Wavelength range: {wavelengths[0]}-{wavelengths[-1]} nm "
          f"({len(wavelengths)} points)")

    # Compute normal hue angles
    M_xyz2lms = compute_xyz_to_lms_matrix(wavelengths, L, M, S)
    hue_normal = compute_hue_angles_for_colors(M_xyz2lms)
    print(f"[2] Normal opponent-space hue angles:")
    for i, (name, h) in enumerate(zip(COLOR_NAMES, hue_normal)):
        print(f"    {name}: {h:.1f}°")

    # Load HC mean LOCO per-color
    loco_dir = Path(args.loco_dir)
    print(f"\n[3] Loading LOCO data from {loco_dir}...")
    hc_mean = compute_hc_mean_loco_per_color(loco_dir, args.model, args.roi)
    print(f"    HC mean per-color voxel_corr: {hc_mean}")

    results = {}

    for cvd_subj in CVD_SUBJECTS:
        cvd_type = CVD_TYPE[cvd_subj]
        print(f"\n{'='*60}")
        print(f"[4] Processing sub-{cvd_subj} (type={cvd_type})")

        if cvd_type == 'normal':
            print("    Compensated CVD → minimal shift expected")

        # Load CVD per-color LOCO
        cvd_path = loco_dir / f'sub-{cvd_subj}_loco.json'
        if not cvd_path.exists():
            print(f"    [SKIP] {cvd_path} not found")
            continue
        cvd_corr = load_loco_per_color(loco_dir, cvd_subj, args.model, args.roi)
        print(f"    CVD per-color voxel_corr: {cvd_corr}")

        # Observed distortion: HC - CVD per-color gap
        delta_obs = hc_mean - cvd_corr
        print(f"    Delta (HC-CVD): {delta_obs}")

        # Grid search
        effective_type = cvd_type if cvd_type in ('deutan', 'protan') else 'deutan'
        best_dl, best_dt, grid = grid_search_delta_lambda(
            wavelengths, L, M, S, delta_obs, effective_type,
            max_shift=args.max_shift
        )
        print(f"    Best Δλ = {best_dl} nm")
        print(f"    Predicted δθ: {best_dt}")
        best_info = grid[best_dl]
        print(f"    Spearman r = {best_info['spearman_r']:.3f} "
              f"(p = {best_info['spearman_p']:.3f})")

        # Fourier fit of predicted delta_theta → T_ψ₀ init
        hue_rad = np.deg2rad(HUE_ANGLES_DEG)
        dt_rad = np.deg2rad(best_dt)
        fourier_init = fourier_fit_delta_theta(hue_rad, dt_rad)
        print(f"    Fourier init: {fourier_init}")

        # Verify: apply T_ψ₀ to hue angles
        corrected = apply_T_psi(HUE_ANGLES_DEG, fourier_init)
        print(f"    Corrected hue angles: {corrected}")

        # Store results
        results[f'sub-{cvd_subj}'] = {
            'cvd_type': cvd_type,
            'best_delta_lambda_nm': best_dl,
            'spearman_r': best_info['spearman_r'],
            'spearman_p': best_info['spearman_p'],
            'mse_normalized': best_info['mse'],
            'delta_theta_pred_deg': best_dt.tolist(),
            'delta_obs_voxel_corr': delta_obs.tolist(),
            'cvd_per_color_corr': cvd_corr.tolist(),
            'hc_mean_per_color_corr': hc_mean.tolist(),
            'fourier_init_rad': fourier_init,
            'hue_normal_deg': hue_normal.tolist(),
            'corrected_hue_deg': corrected.tolist(),
            'grid_search_top5': _top_n_grid(grid, n=5),
        }

    # Save results
    out_path = output_dir / 'cone_shift_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[5] Results saved to {out_path}")

    # Save config
    config = {
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'script': 'stockman_cone_shift.py',
        'parameters': {
            'loco_dir': str(args.loco_dir),
            'roi': args.roi,
            'model': args.model,
            'max_shift_nm': args.max_shift,
            'n_colors': 8,
            'L_star': L_STAR,
            'chroma': CHROMA,
            'wavelength_range': f'{wavelengths[0]}-{wavelengths[-1]}',
        },
        'cvd_types': CVD_TYPE,
    }
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"    Config saved to {config_path}")


def _top_n_grid(grid_results, n=5):
    """Extract top-N grid search results by MSE."""
    sorted_items = sorted(grid_results.items(), key=lambda x: x[1]['mse'])
    return {str(k): v for k, v in sorted_items[:n]}


if __name__ == '__main__':
    main()
