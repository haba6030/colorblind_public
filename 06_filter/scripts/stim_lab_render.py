"""STIM_LAB-based rendering for filter visualization.

Source: utils_color_decoding.py COLOR_LAB — screenshot-derived CIElab values
representing what subjects actually saw on the MRI projector.

Provides:
- STIM_LAB / STIM_L / STIM_HUE_DEG / STIM_CHROMA constants
- lab2rgb: standard CIElab → sRGB
- stim_lab_at_hue(theta): interpolate L*/chroma from STIM_LAB samples
- render_at_hue(theta, dL=0): full Lab → sRGB at any hue, optional ΔL*
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 06_filter
# Manuscript: Results section 6; Figure 6; Methods 'Stimulus-space filter and its evaluation'
# Original  : analysis/phase5_filter_optimization/scripts/stim_lab_render.py  (development repository, commit 53c81c2)
# Role      : CIELab rendering of the original and filtered discs.
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

# Screenshot-derived CIElab from utils_color_decoding.py:46-55
# Comment: "These values were extracted from actual experiment screenshots
#           and represent what subjects saw."
STIM_LAB = {
    'color_1': [59.90, 62.69, 3.78],     # 0°  Red
    'color_2': [64.20, 49.20, 45.58],    # 45° Orange
    'color_3': [57.27, 13.06, 41.69],    # 90° Yellow
    'color_4': [69.08, -55.02, 47.38],   # 135° Green
    'color_5': [74.61, -41.33, -4.89],   # 180° Cyan
    'color_6': [69.14, -11.45, -40.91],  # 225° Blue
    'color_7': [60.68, 19.18, -54.13],   # 270° Violet
    'color_8': [60.17, 46.82, -40.31],   # 315° Magenta
}

STIM_LAB_ARR = np.array([STIM_LAB[f'color_{i+1}'] for i in range(8)])
STIM_L = STIM_LAB_ARR[:, 0]
STIM_A = STIM_LAB_ARR[:, 1]
STIM_B = STIM_LAB_ARR[:, 2]
STIM_CHROMA = np.sqrt(STIM_A**2 + STIM_B**2)
STIM_HUE_DEG = np.rad2deg(np.arctan2(STIM_B, STIM_A)) % 360
IDEAL_HUE_DEG = np.array([0, 45, 90, 135, 180, 225, 270, 315], dtype=float)

_M_XYZ_TO_SRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570],
])


def lab2rgb(L, a, b, clip=True):
    """CIElab → sRGB (D65). Returns shape (..., 3) in [0, 1]."""
    L = np.asarray(L, float); a = np.asarray(a, float); b = np.asarray(b, float)
    y = (L + 16) / 116
    x = a / 500 + y
    z = y - b / 200
    xyz = np.stack([x, y, z], axis=-1)
    xyz = np.where(xyz > 0.206893, xyz**3, (xyz - 16/116) / 7.787)
    xyz *= np.array([0.95047, 1.0, 1.08883])
    rgb = xyz @ _M_XYZ_TO_SRGB.T
    rgb = np.where(rgb <= 0.0031308, 12.92 * rgb,
                   1.055 * np.power(np.maximum(rgb, 0), 1/2.4) - 0.055)
    if clip:
        rgb = np.clip(rgb, 0, 1)
    return rgb


def _wrap_circular(hue_deg):
    return float(hue_deg) % 360.0


def _interp_circular(target_hue, sample_hues, sample_values):
    """Linear interpolation of sample_values at target_hue, treating hue as periodic."""
    target = _wrap_circular(target_hue)
    # Build extended arrays for wrap-around
    sample_hues = np.asarray(sample_hues, float) % 360
    order = np.argsort(sample_hues)
    h = sample_hues[order]
    v = np.asarray(sample_values, float)[order]
    h_ext = np.concatenate([h - 360, h, h + 360])
    v_ext = np.concatenate([v, v, v])
    return float(np.interp(target, h_ext, v_ext))


def stim_lab_at_hue(theta_deg, use_ideal_index=True):
    """Return (L*, chroma) at arbitrary hue θ by interpolating STIM_LAB.

    Args:
        theta_deg: target CIElab hue in degrees
        use_ideal_index: if True, interpolate over IDEAL_HUE_DEG (0,45,...,315);
                         if False, use screenshot-derived STIM_HUE_DEG.
                         IDEAL is preferred for model consistency.

    Returns:
        (L_star, chroma) tuple of floats.
    """
    hues = IDEAL_HUE_DEG if use_ideal_index else STIM_HUE_DEG
    L = _interp_circular(theta_deg, hues, STIM_L)
    C = _interp_circular(theta_deg, hues, STIM_CHROMA)
    return L, C


def render_at_hue(theta_deg, dL=0.0):
    """Render sRGB at any hue using STIM_LAB-interpolated L*/chroma.

    Args:
        theta_deg: CIElab hue in degrees
        dL: optional ΔL* (for Machado/R+C cone-shift); 0 for 2-comp.

    Returns:
        sRGB array shape (3,) in [0, 1].
    """
    L_base, C = stim_lab_at_hue(theta_deg)
    L = L_base + float(dL)
    rad = np.deg2rad(float(theta_deg))
    a = C * np.cos(rad)
    b = C * np.sin(rad)
    return lab2rgb(L, a, b)
