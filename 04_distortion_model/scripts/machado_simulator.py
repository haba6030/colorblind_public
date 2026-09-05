#!/usr/bin/env python3
"""
machado_simulator.py — Machado, Oliveira & Fernandes (2009) CVD simulator.

Implements Equations 5 and 6 of the paper in LMS (Stockman & Sharpe 2000)
cone space, with area preservation (0.96 constant) so that the D65
achromatic point is invariant across severities.

Protanomaly (Eq. 5):
    L_a(λ) = α · L(λ + Δλ) + (1 − α) · k_L · M(λ)
    M_a(λ) = M(λ)
    S_a(λ) = S(λ)
    k_L    = 0.96 · (Area_M / Area_L)

Deuteranomaly (Eq. 6):
    L_a(λ) = L(λ)
    M_a(λ) = α · M(λ + Δλ) + (1 − α) · k_M · L(λ)
    S_a(λ) = S(λ)
    k_M    = 0.96 · (Area_L / Area_M)

Severity coupling (both equations):
    α = clip((Δλ_max − Δλ) / Δλ_max, 0, 1),  Δλ_max = 20 nm
        α = 1 → normal trichromat
        α = 0 → dichromat (fully replaced by the other cone, area-preserved)

The public API provides:
  - machado_mixed_fundamentals(...)      — Machado-1way, α coupled to Δλ
  - machado_alpha_free_fundamentals(...) — Machado-α-free (2-DOF)
  - machado_shifted_hue(...)             — end-to-end Δλ → shifted hue angles (8,)
  - verify_machado_gray_point(...)       — D65 preservation sanity check

All spectral functions are computed on the Stockman 1 nm wavelength grid using
the existing ``stockman_cone_shift`` utilities.
"""

from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/machado_simulator.py  (development repository, commit 53c81c2)
# Role      : Machado (2009) cone-shift hue displacement on Stockman-Sharpe fundamentals.
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


import sys
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np

_PHASE2_DIR = Path(__file__).resolve().parent.parent
for _base in [_PHASE2_DIR.parent, _PHASE2_DIR.parent.parent]:
    _fwd = _base / 'phase4_forward_model' / 'scripts'
    if _fwd.exists() and str(_fwd) not in sys.path:
        sys.path.insert(0, str(_fwd))
        break

from stockman_cone_shift import (  # noqa: E402
    COLOR_LAB,
    D65_XYZ,
    HUE_ANGLES_DEG,
    _compute_spectral_shift_lms,
    compute_xyz_to_lms_matrix,
    lab_to_xyz,
    lms_to_opponent,
    load_stockman_fundamentals,
    opponent_to_hue_angle,
    shift_cone_sensitivity,
)

# ============================================================================
# Constants
# ============================================================================

DELTA_LAMBDA_MAX: float = 20.0  # Machado Δλ_max (nm)
MACHADO_AREA_CONSTANT: float = 0.96  # Machado's 0.96 D65 calibration constant

CvdType = Literal['protan', 'deutan', 'normal']

# Cache (computed once)
_GRID_CACHE: dict = {}
_D65_LMS_NORMAL: Optional[np.ndarray] = None


# ============================================================================
# Grid / baseline helpers
# ============================================================================

def _load_stockman_grid() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                    float, float, np.ndarray]:
    """Load Stockman cone fundamentals + compute area terms.

    Returns:
        wl, L, M, S, Area_L, Area_M, M_xyz2lms
    """
    if 'data' not in _GRID_CACHE:
        wl, L, M, S = load_stockman_fundamentals()
        wl = np.asarray(wl, dtype=float)
        L = np.asarray(L, dtype=float)
        M = np.asarray(M, dtype=float)
        S = np.asarray(S, dtype=float)
        area_L = float(np.trapz(L, wl))
        area_M = float(np.trapz(M, wl))
        M_xyz2lms = compute_xyz_to_lms_matrix(wl, L, M, S)
        _GRID_CACHE['data'] = (wl, L, M, S, area_L, area_M, M_xyz2lms)
    return _GRID_CACHE['data']


def _compute_alpha(delta_lambda: float,
                   delta_lambda_max: float = DELTA_LAMBDA_MAX) -> float:
    """Severity mapping α = clip((Δλ_max − Δλ)/Δλ_max, 0, 1)."""
    if delta_lambda_max <= 0:
        raise ValueError('delta_lambda_max must be positive')
    alpha = (delta_lambda_max - float(delta_lambda)) / delta_lambda_max
    return float(np.clip(alpha, 0.0, 1.0))


def _load_d65_spd(wl: np.ndarray) -> np.ndarray:
    """Load D65 SPD on the Stockman wavelength grid (cached, area-normalised).

    The colour-science ``SDS_ILLUMINANTS['D65']`` is interpolated onto the
    caller's wavelength grid. A smooth Gaussian fallback is used if the
    library is unavailable, purely so the module can still be imported in
    stripped-down environments; the Gaussian path does NOT reproduce
    Machado's calibration and is flagged via a warning.
    """
    key = ('d65_spd', float(wl[0]), float(wl[-1]), int(wl.size))
    if key not in _GRID_CACHE:
        try:
            import colour  # type: ignore
            sd = colour.SDS_ILLUMINANTS['D65']
            spd = np.interp(wl, np.asarray(sd.wavelengths, dtype=float),
                            np.asarray(sd.values, dtype=float))
        except Exception as exc:  # pragma: no cover — dev fallback
            import warnings
            warnings.warn(
                f'colour-science D65 unavailable ({exc}); using Gaussian '
                'fallback — Machado calibration will drift.')
            spd = 100.0 * np.exp(-0.5 * ((wl - 555.0) / 150.0) ** 2)
        norm = float(np.trapz(spd, wl))
        if norm > 0:
            spd = spd / norm
        _GRID_CACHE[key] = spd
    return _GRID_CACHE[key]


def _compute_k_factors() -> Tuple[float, float]:
    """D65-preserving k_L (protan) and k_M (deutan) on the Stockman grid.

    At α = 0 (full dichromat), Machado Eq 5 reduces to ``L_a = k_L · M``
    and Eq 6 to ``M_a = k_M · L``. For the D65 achromatic point to be
    preserved under the same XYZ→LMS LSQ projection the simulator uses
    downstream, the integrated response of the replacement cone against
    D65 must equal the original cone's integrated D65 response:

        ∫ L(λ) D65(λ) dλ = k_L · ∫ M(λ) D65(λ) dλ
        ∫ M(λ) D65(λ) dλ = k_M · ∫ L(λ) D65(λ) dλ

    which gives closed-form k values below. These replace Machado's
    published 0.96·(Area_M/Area_L) constant when the caller's Stockman
    normalisation differs from the reference used in the paper (true
    for colour-science's 2° fundamentals).
    """
    if 'k_factors' not in _GRID_CACHE:
        wl, L, M, S, _, _, _ = _load_stockman_grid()
        spd = _load_d65_spd(wl)
        L_d65 = float(np.trapz(L * spd, wl))
        M_d65 = float(np.trapz(M * spd, wl))
        k_L = L_d65 / max(M_d65, 1e-12)  # protan: L_a = k_L · M ⇒ equal D65 response
        k_M = M_d65 / max(L_d65, 1e-12)  # deutan: M_a = k_M · L ⇒ equal D65 response
        _GRID_CACHE['k_factors'] = (float(k_L), float(k_M))
    return _GRID_CACHE['k_factors']


# ============================================================================
# Core Machado cone mixtures
# ============================================================================

def machado_mixed_fundamentals(
    delta_lambda: float,
    cvd_type: CvdType,
    wl: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    M: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
    delta_lambda_max: float = DELTA_LAMBDA_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Machado 1-way (α coupled to Δλ) area-preserving fundamentals.

    Implements Eq. 5 (protanomaly) or Eq. 6 (deuteranomaly) of
    Machado, Oliveira & Fernandes (2009) in Stockman LMS space.

    Args:
        delta_lambda: shift magnitude in nm (>= 0, capped at Δλ_max for α coupling)
        cvd_type: 'protan' (Eq. 5, L shifted) or 'deutan' (Eq. 6, M shifted)
        wl, L, M, S: optional cone fundamentals (use cached Stockman by default)
        delta_lambda_max: severity normalization constant (default 20 nm)

    Returns:
        L_a, M_a, S_a: (N,) arrays on the same wavelength grid
    """
    if wl is None or L is None or M is None or S is None:
        wl, L, M, S, *_ = _load_stockman_grid()

    alpha = _compute_alpha(delta_lambda, delta_lambda_max)
    k_L, k_M = _compute_k_factors()

    if cvd_type == 'protan':
        # Protanomaly shifts L toward M (shorter wavelengths). Matches
        # stockman_cone_shift.compute_shifted_hue_angles: for protan the
        # shift is -Δλ so that the L peak moves toward the M peak.
        L_shift = shift_cone_sensitivity(wl, L, -float(delta_lambda))
        L_a = alpha * L_shift + (1.0 - alpha) * k_L * M
        M_a = M.copy()
        S_a = S.copy()
    elif cvd_type == 'deutan':
        # Deuteranomaly shifts M toward L (longer wavelengths): +Δλ.
        M_shift = shift_cone_sensitivity(wl, M, float(delta_lambda))
        L_a = L.copy()
        M_a = alpha * M_shift + (1.0 - alpha) * k_M * L
        S_a = S.copy()
    elif cvd_type == 'normal':
        L_a, M_a, S_a = L.copy(), M.copy(), S.copy()
    else:
        raise ValueError(f'Unknown cvd_type: {cvd_type}')

    return L_a, M_a, S_a


def machado_alpha_free_fundamentals(
    delta_lambda: float,
    alpha: float,
    cvd_type: CvdType,
    wl: Optional[np.ndarray] = None,
    L: Optional[np.ndarray] = None,
    M: Optional[np.ndarray] = None,
    S: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Machado 2-DOF variant: independent Δλ and α.

    Same equations as ``machado_mixed_fundamentals`` but α is supplied explicitly
    rather than coupled to Δλ. Useful for exploring trichromat↔dichromat spectra.
    """
    if wl is None or L is None or M is None or S is None:
        wl, L, M, S, *_ = _load_stockman_grid()

    alpha_c = float(np.clip(alpha, 0.0, 1.0))
    k_L, k_M = _compute_k_factors()

    if cvd_type == 'protan':
        L_shift = shift_cone_sensitivity(wl, L, -float(delta_lambda))
        L_a = alpha_c * L_shift + (1.0 - alpha_c) * k_L * M
        return L_a, M.copy(), S.copy()
    if cvd_type == 'deutan':
        M_shift = shift_cone_sensitivity(wl, M, float(delta_lambda))
        M_a = alpha_c * M_shift + (1.0 - alpha_c) * k_M * L
        return L.copy(), M_a, S.copy()
    if cvd_type == 'normal':
        return L.copy(), M.copy(), S.copy()
    raise ValueError(f'Unknown cvd_type: {cvd_type}')


# ============================================================================
# End-to-end: Δλ → shifted hue angles (8,)
# ============================================================================

def _hue_from_fundamentals(L_a: np.ndarray, M_a: np.ndarray, S_a: np.ndarray,
                           wl: np.ndarray, M_xyz2lms_orig: np.ndarray,
                           xyz_colors: np.ndarray) -> np.ndarray:
    """Go from (L_a, M_a, S_a) to 8 hue angles.

    Uses the same least-squares XYZ→LMS rebuild that
    ``_compute_spectral_shift_lms`` performs when cones are modified.
    """
    lms = _compute_spectral_shift_lms(wl, L_a, M_a, S_a, M_xyz2lms_orig,
                                      xyz_colors)
    rg, by = lms_to_opponent(lms)
    return opponent_to_hue_angle(rg, by)


def _baseline_hue_normal() -> np.ndarray:
    """Hue angles of the 8 stimuli under unshifted Stockman cones (cached)."""
    if 'hue_normal' not in _GRID_CACHE:
        wl, L, M, S, _, _, M_xyz2lms = _load_stockman_grid()
        xyz = lab_to_xyz(COLOR_LAB)
        _GRID_CACHE['hue_normal'] = _hue_from_fundamentals(
            L, M, S, wl, M_xyz2lms, xyz)
    return _GRID_CACHE['hue_normal']


def machado_shifted_hue_at(
    delta_lambda: float,
    cvd_type: CvdType,
    theta_deg: float | np.ndarray,
    alpha: Optional[float] = None,
    delta_lambda_max: float = DELTA_LAMBDA_MAX,
    L_star: float = 75.0,
    chroma: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Δλ → shifted hue angle(s) for **arbitrary** CIELab input angle(s).

    Unlike ``machado_shifted_hue`` which uses the 8 hardcoded stimuli,
    this function constructs CIELab at (L_star, chroma, θ_input) and
    evaluates the Machado model at those custom inputs.

    Args:
        delta_lambda: shift in nm (>= 0). 0 ↔ trichromatic baseline.
        cvd_type: 'protan', 'deutan', or 'normal'.
        theta_deg: hue angle(s) in degrees [0, 360). Scalar or 1-D array.
        alpha: if provided, use the 2-DOF variant; else α coupled to Δλ.
        delta_lambda_max: severity normalization constant.
        L_star: CIELab lightness (default 75.0, matching experiment).
        chroma: CIELab chroma (default 40.0, matching experiment).

    Returns:
        hue_normal: (N,) baseline hue angles under unshifted cones
        hue_shifted: (N,) shifted hue angles under CVD cones
        delta_theta: (N,) wrapped difference in [-180, 180]
    """
    theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=float))
    theta_rad = np.deg2rad(theta_deg)

    # Construct CIELab at requested angles
    custom_lab = np.column_stack([
        np.full_like(theta_deg, L_star),
        chroma * np.cos(theta_rad),
        chroma * np.sin(theta_rad),
    ])  # (N, 3)

    wl, L, M, S, _, _, M_xyz2lms = _load_stockman_grid()
    xyz_custom = lab_to_xyz(custom_lab)

    # Normal hue at custom inputs
    hue_normal = _hue_from_fundamentals(L, M, S, wl, M_xyz2lms, xyz_custom)

    if cvd_type == 'normal' or delta_lambda == 0.0:
        hue_shifted = hue_normal.copy()
    else:
        if alpha is None:
            L_a, M_a, S_a = machado_mixed_fundamentals(
                delta_lambda, cvd_type, wl, L, M, S,
                delta_lambda_max=delta_lambda_max)
        else:
            L_a, M_a, S_a = machado_alpha_free_fundamentals(
                delta_lambda, alpha, cvd_type, wl, L, M, S)
        hue_shifted = _hue_from_fundamentals(
            L_a, M_a, S_a, wl, M_xyz2lms, xyz_custom)

    delta_theta = (hue_shifted - hue_normal + 180.0) % 360.0 - 180.0
    return hue_normal, hue_shifted, delta_theta


def machado_shifted_hue(
    delta_lambda: float,
    cvd_type: CvdType,
    alpha: Optional[float] = None,
    delta_lambda_max: float = DELTA_LAMBDA_MAX,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Δλ → shifted hue angles for the 8 experimental stimuli.

    Args:
        delta_lambda: shift in nm (>= 0). 0 ↔ trichromatic baseline.
        cvd_type: 'protan', 'deutan', or 'normal'.
        alpha: if provided, use the 2-DOF ``machado_alpha_free`` variant;
               otherwise α is coupled to Δλ per Machado Eq 5/6.
        delta_lambda_max: severity normalization constant.

    Returns:
        hue_normal: (8,) baseline hue angles (Δλ = 0)
        hue_shifted: (8,) shifted hue angles
        delta_theta: (8,) wrapped difference in [-180, 180]
    """
    wl, L, M, S, _, _, M_xyz2lms = _load_stockman_grid()
    xyz_colors = lab_to_xyz(COLOR_LAB)
    hue_normal = _baseline_hue_normal()

    if cvd_type == 'normal' or delta_lambda == 0.0:
        hue_shifted = hue_normal.copy()
    else:
        if alpha is None:
            L_a, M_a, S_a = machado_mixed_fundamentals(
                delta_lambda, cvd_type, wl, L, M, S,
                delta_lambda_max=delta_lambda_max)
        else:
            L_a, M_a, S_a = machado_alpha_free_fundamentals(
                delta_lambda, alpha, cvd_type, wl, L, M, S)
        hue_shifted = _hue_from_fundamentals(
            L_a, M_a, S_a, wl, M_xyz2lms, xyz_colors)

    delta_theta = (hue_shifted - hue_normal + 180.0) % 360.0 - 180.0
    return hue_normal, hue_shifted, delta_theta


# ============================================================================
# Sanity check: D65 achromatic point preservation
# ============================================================================
#
# NOTE (numerical stability):
#   D65 lives at the origin of the (rg, by) opponent plane (rg ≈ 0, by ≈ 0
#   by construction of the Stockman LMS whitepoint). An arctan2-based polar
#   angle of a vector that close to the origin is meaningless: a 10⁻⁵
#   sign flip of rg_s swings the reported angle by ≳ 100°. The Machado
#   0.96 calibration is designed to keep the achromatic point invariant in
#   a *displacement* sense, not a polar-angle sense. We therefore measure
#   D65 preservation via the opponent-space displacement of the shifted
#   whitepoint relative to the mean chromatic magnitude of the 8
#   experimental stimuli, and convert it to an equivalent angular metric
#   via arcsin so that the "tolerance_deg" interface of
#   verify_machado_gray_point() still reads in degrees.
#
#   drift_frac = ||(rg_s, by_s) − (rg_n, by_n)|| / mean_chromatic_magnitude
#   drift_deg  = rad2deg(arcsin(clip(drift_frac, 0, 1)))
#
#   "drift_deg = 2°" therefore means "D65 moved by sin(2°) ≈ 3.5 % of a
#   typical stimulus saturation" — a physically sensible Machado check.


def _chromatic_reference_magnitude() -> float:
    """Mean |opponent vector| of the 8 experimental stimuli under normal cones.

    Serves as the denominator for the D65 drift check so that the result
    has a meaningful chromatic-saturation scale. Cached on first call.
    """
    if 'chromatic_ref' not in _GRID_CACHE:
        wl, L, M, S, _, _, M_xyz2lms = _load_stockman_grid()
        xyz = lab_to_xyz(COLOR_LAB)
        lms = _compute_spectral_shift_lms(wl, L, M, S, M_xyz2lms, xyz)
        rg, by = lms_to_opponent(lms)
        mags = np.sqrt(np.asarray(rg) ** 2 + np.asarray(by) ** 2)
        _GRID_CACHE['chromatic_ref'] = float(np.mean(mags))
    return float(_GRID_CACHE['chromatic_ref'])


def _d65_drift_deg(L_a: np.ndarray, M_a: np.ndarray, S_a: np.ndarray) -> float:
    """D65 achromatic-point drift as an equivalent chromatic angle.

    Numerically stable replacement for the deprecated arctan2-based
    `_d65_hue_angle`. Measures the displacement of the D65 opponent-space
    response under shifted cones relative to the mean chromatic magnitude
    of the 8 experimental stimuli, then converts to degrees via arcsin.

    Returns 0.0 at Δλ = 0 (by construction, since L_a = L, M_a = M, S_a = S)
    and grows monotonically with Δλ. The output is directly comparable to
    the ``tolerance_deg`` argument of ``verify_machado_gray_point``.
    """
    wl, L_n, M_n, S_n, _, _, M_xyz2lms = _load_stockman_grid()
    D65 = np.asarray(D65_XYZ, dtype=float)[np.newaxis, :]
    lms_n = _compute_spectral_shift_lms(wl, L_n, M_n, S_n, M_xyz2lms, D65)
    lms_s = _compute_spectral_shift_lms(wl, L_a, M_a, S_a, M_xyz2lms, D65)
    rg_n, by_n = lms_to_opponent(lms_n)
    rg_s, by_s = lms_to_opponent(lms_s)
    d_rg = float(np.ravel(np.asarray(rg_s) - np.asarray(rg_n))[0])
    d_by = float(np.ravel(np.asarray(by_s) - np.asarray(by_n))[0])
    drift = float(np.sqrt(d_rg * d_rg + d_by * d_by))
    ref = _chromatic_reference_magnitude()
    frac = float(np.clip(drift / max(ref, 1e-12), 0.0, 1.0))
    return float(np.rad2deg(np.arcsin(frac)))


def verify_machado_gray_point(
    tolerance_deg: float = 2.5,
    delta_grid: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> bool:
    """Check that the D65 achromatic point is preserved across Δλ ∈ [0, 20].

    For each severity Δλ in ``delta_grid``, builds Machado-mixed fundamentals
    and measures the chromatic drift of the D65 point (displacement /
    mean-stimulus chromatic magnitude, converted to degrees via arcsin —
    see `_d65_drift_deg` for rationale). The check passes when every
    drift is <= ``tolerance_deg``.

    Tolerance rationale:
        - k_L, k_M are computed via direct ∫cone·D65 integration
          (`_compute_k_factors`) so drift is exactly 0° at Δλ=0 (α=1)
          and ≤0.3° at Δλ=20 (α=0, full dichromat).
        - At intermediate α the mixed cone spectrum α·cone_shift +
          (1−α)·k·other_cone has more spectral curvature, increasing the
          LSQ-projection residual inside `_compute_spectral_shift_lms`.
          Empirically this produces ~2° peak drift at α≈0.5 for deutan
          and ~1.9° for protan — this is the LSQ-residual floor, not a
          k-factor calibration error. Dynamic-k variants were tested and
          made deutan drift *worse* (up to 4.6°), confirming that the
          residual is downstream of the k choice.
        - 2.5° drift = sin(2.5°) ≈ 4.4 % of a stimulus's chromatic
          saturation — ≤1 JND for saturated colours, i.e. the D65 point
          is preserved in every perceptually meaningful sense.

    Args:
        tolerance_deg: maximum allowed chromatic drift in degrees (default
            2.5, empirically chosen as the LSQ-residual floor).
        delta_grid: iterable of Δλ values to probe (default: [0, 5, 10, 15, 20]).
        verbose: if True, print per-Δλ diagnostics.

    Returns:
        True if all Δλ pass, False otherwise.
    """
    if delta_grid is None:
        delta_grid = np.array([0.0, 5.0, 10.0, 15.0, 20.0])

    wl, L, M, S, _, _, _ = _load_stockman_grid()
    all_ok = True
    details = []
    for cvd_type in ('protan', 'deutan'):
        for dl in delta_grid:
            L_a, M_a, S_a = machado_mixed_fundamentals(
                float(dl), cvd_type, wl, L, M, S)
            deviation = _d65_drift_deg(L_a, M_a, S_a)
            details.append((cvd_type, float(dl), deviation))
            if deviation > tolerance_deg:
                all_ok = False
            if verbose:
                status = 'OK' if deviation <= tolerance_deg else 'FAIL'
                print(f'  {cvd_type:>6s} Δλ={dl:5.1f} nm '
                      f'D65 drift={deviation:.3f}° [{status}]')
    _GRID_CACHE['gray_check_details'] = details
    return bool(all_ok)


def gray_point_details() -> list:
    """Return the (cvd_type, Δλ, deviation_deg) list from the last check."""
    return _GRID_CACHE.get('gray_check_details', [])


# ============================================================================
# Module self-test
# ============================================================================

if __name__ == '__main__':
    print('Machado simulator self-test')
    print('=' * 48)

    # 1. Gray-point preservation
    ok = verify_machado_gray_point(tolerance_deg=2.5, verbose=True)
    print(f'\nGray-point check: {"PASS" if ok else "FAIL"}')

    # 2. End-to-end hue shift at a few representative Δλ values
    print('\nExample hue shifts (deutan):')
    for dl in (0.0, 5.0, 10.0, 15.0, 20.0):
        hue_n, hue_s, dtheta = machado_shifted_hue(dl, 'deutan')
        print(f'  Δλ={dl:5.1f} nm | max |Δθ|={np.max(np.abs(dtheta)):6.2f}°')

    print('\nExample hue shifts (protan):')
    for dl in (0.0, 5.0, 10.0, 15.0, 20.0):
        hue_n, hue_s, dtheta = machado_shifted_hue(dl, 'protan')
        print(f'  Δλ={dl:5.1f} nm | max |Δθ|={np.max(np.abs(dtheta)):6.2f}°')
