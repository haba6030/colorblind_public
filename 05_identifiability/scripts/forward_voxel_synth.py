"""forward_voxel_synth.py — Voxel-level synthesis for Phase B v6 verification.

Foundational module (NOT runnable directly) implementing voxel-space forward
simulation per Wilson & Collins 2019 / Schütt 2021. Replaces the
loss-function-level injection (Method C) used by exp14/15/18/21/22, which
those works deem insufficient because the noise structure and encoder
fidelity are absent.

Forward map:
    Y_synth[run, c, :] = W_k @ C(θ_c + δθ(c; β_s, β_c, family)) + ε[run, c, :]

where
    W_k        : HC_k's ridge_gcv encoder (channel-space → HC_k's voxel-space)
    C(·)       : FE-K basis at perceived hue
    δθ(c)      : 2-component cortical rotation (matches scripts/two_comp.py
                  forward; closure-active per project CLAUDE.md A13)
    ε          : noise with HC_k's spatial covariance (top-20 PCs) AND
                  temporal AR(1) across runs (ρ = 0.3 default)

The voxel-space synth uses the *donor HC's* W and noise structure — never the
CVD subject's own W (that would be circular). This is the property that
Method C lacks: Method C swaps RDM rows wholesale, ignoring W and ε; voxel
synth re-projects through W and re-adds realistic ε.

Used by:
  - null_within_hc_loo.py   (Source B variant B2 — synth GT=0 null)
  - param_recovery_voxel.py (Stage 1 — GT magnitude sweep)

NOT a runnable script. Importable functions only.
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 05_identifiability
# Manuscript: Methods 'Identifiability and recovery'; Supplementary S12 (tab:identifiability); S13 control leave-one-out magnitude anchor
# Original  : analysis/phase5_filter_optimization/scripts/forward_voxel_synth.py  (development repository, commit 53c81c2)
# Role      : Synthetic voxel responses at a known distortion with donor-matched noise (PCA(20) spatial covariance, AR(1) = 0.3).
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
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# project utilities
from neural_loss import HUE_ANGLES, ROI_K  # noqa: E402  (after sys.path mutation)
from two_comp import forward_2comp, THETA_CONF  # noqa: E402

# encoder
_PHASE1_FWD = SCRIPT_DIR.parents[1] / "phase4_forward_model" / "scripts"
sys.path.insert(0, str(_PHASE1_FWD))
from utils_forward_model import (  # noqa: E402
    create_basis_full,
    fit_W_ridge,
    gcv_select_alpha,
    N_RUNS,
    N_COLORS,
)

# ---------------------------------------------------------------------------
# Module-level constants (audit trail; written into output JSON 'config')
# ---------------------------------------------------------------------------
SPATIAL_COV_RANK: int = 20        # top-K PCs retained for spatial covariance
AR1_RHO: float = 0.3              # temporal AR(1) ρ across runs
EPS_RIDGE_FLOOR: float = 1e-8     # safety floor for ridge diag
DEFAULT_BASIS: str = "fe"


# ---------------------------------------------------------------------------
# δθ helper (mirrors two_comp.forward_2comp; explicit so callers can reason
# about the family dispatch without importing the closure module directly)
# ---------------------------------------------------------------------------
def delta_theta_2comp(theta_deg: np.ndarray, beta_s: float, beta_c: float,
                      family: str) -> np.ndarray:
    """Per-color δθ for the closure 2-component forward.

    δθ(c) = β_s · cos(θ_c − 90°) + β_c · cos(θ_c − θ_conf)
    θ_conf = 16° (protan) / 150° (deutan).

    Args:
        theta_deg: (8,) canonical hue angles in degrees.
        beta_s:    S-(L+M) amplitude.
        beta_c:    Confusion-axis amplitude.
        family:    'protan' | 'deutan'.

    Returns:
        δθ (8,) in degrees.
    """
    if family not in THETA_CONF:
        raise ValueError(f"Unknown family {family!r}; expected protan|deutan.")
    return forward_2comp(beta_s, beta_c, family, hues=np.asarray(theta_deg))


# ---------------------------------------------------------------------------
# Encoder W_k (per HC, per ROI)
# ---------------------------------------------------------------------------
def _fit_W_ridge_gcv(C: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ridge encoder with GCV-selected α. Returns W (K, V_s).

    Mirrors the encoder choice used by `neural_loss.precompute_loco_W_within`
    and `s10b_v6_pca_rdm` (the production fit path).
    """
    alpha, _ = gcv_select_alpha(C, Y)
    return fit_W_ridge(C, Y, max(float(alpha), EPS_RIDGE_FLOOR))


def precompute_per_hc_W(hc_amps_dict: dict, C_baseline: np.ndarray) -> dict:
    """Per-HC ridge_gcv encoder W mapping channel-space → voxel-space.

    For each HC_k:
        Stack (N_RUNS·N_COLORS, K) repeated baseline design against the
        HC_k amplitude reshape (N_RUNS·N_COLORS, V_k); fit ridge with GCV α.

    Args:
        hc_amps_dict: {hc_id: (N_RUNS, N_COLORS, V_k) ndarray}.
        C_baseline:   (8, K) design matrix at canonical hue.

    Returns:
        {hc_id: W_k of shape (K, V_k)}.
    """
    K = C_baseline.shape[1]
    out: dict = {}
    for hc_id, amp in hc_amps_dict.items():
        if amp.ndim != 3 or amp.shape[1] != N_COLORS:
            continue
        n_runs, n_col, V_k = amp.shape
        C_stack = np.tile(C_baseline, (n_runs, 1))         # (n_runs*8, K)
        Y_stack = amp.reshape(n_runs * n_col, V_k)          # (n_runs*8, V_k)
        try:
            W_k = _fit_W_ridge_gcv(C_stack, Y_stack)        # (K, V_k)
        except np.linalg.LinAlgError:
            continue
        assert W_k.shape == (K, V_k)
        out[hc_id] = W_k
    return out


# ---------------------------------------------------------------------------
# Noise estimation (Schütt 2021 style)
# ---------------------------------------------------------------------------
def estimate_noise_per_hc(hc_amp: np.ndarray, W_k: np.ndarray,
                          C_baseline: np.ndarray) -> dict:
    """Estimate spatial-temporal noise structure from one HC's residuals.

    Residual model:
        ε[run, c, v] = hc_amp[run, c, v] − (C_baseline @ W_k)[c, v]

    Spatial covariance: top-`SPATIAL_COV_RANK` PCs of the
    (n_runs · n_colors, V_k) residual matrix. Stored as (V_k, R) loading L so
    a sample ~ L @ z, z ~ N(0, I_R).

    Temporal correlation: AR(1) over runs with ρ = `AR1_RHO`.

    Args:
        hc_amp:     (N_RUNS, N_COLORS, V_k) ndarray.
        W_k:        (K, V_k) encoder.
        C_baseline: (8, K) design matrix.

    Returns:
        dict with keys:
          - 'spatial_loading'        : (V_k, R) loading matrix L
          - 'spatial_rank'           : R (≤ SPATIAL_COV_RANK)
          - 'temporal_ar1_rho'       : float
          - 'residual_std_per_voxel' : (V_k,)
          - 'V_k'                    : int
          - 'n_residual_samples'     : int (n_runs*n_colors)
    """
    n_runs, n_col, V_k = hc_amp.shape
    Y_baseline = C_baseline @ W_k                            # (8, V_k)
    residual = hc_amp - Y_baseline[None, :, :]               # (n_runs, 8, V_k)
    R_mat = residual.reshape(n_runs * n_col, V_k)            # (N, V_k)
    R_mat = R_mat - R_mat.mean(axis=0, keepdims=True)        # center per voxel

    # Truncated SVD: R = U S Vt; spatial loading L = Vt.T * (S / sqrt(N-1)).
    n_samp = R_mat.shape[0]
    try:
        U, S, Vt = np.linalg.svd(R_mat, full_matrices=False)
        rank = int(min(SPATIAL_COV_RANK, S.size, n_samp - 1))
        rank = max(rank, 1)
        scale = (S[:rank] / np.sqrt(max(n_samp - 1, 1)))
        L = (Vt[:rank].T) * scale[None, :]                    # (V_k, R)
    except np.linalg.LinAlgError:
        rank = 1
        L = np.zeros((V_k, 1))

    std_per_v = R_mat.std(axis=0, ddof=1)                     # (V_k,)
    return {
        "spatial_loading": L,
        "spatial_rank": rank,
        "temporal_ar1_rho": float(AR1_RHO),
        "residual_std_per_voxel": std_per_v,
        "V_k": int(V_k),
        "n_residual_samples": int(n_samp),
    }


def _sample_spatially_correlated(L: np.ndarray, n_samples: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """Draw n samples ~ N(0, L L^T). Returns (n_samples, V_k)."""
    R = L.shape[1]
    Z = rng.standard_normal((n_samples, R))
    return Z @ L.T


def _apply_ar1(samples_runs: np.ndarray, rho: float,
               rng: np.random.Generator) -> np.ndarray:
    """Inject AR(1) temporal correlation across runs.

    samples_runs: (n_runs, V_k) i.i.d. spatial draws.

    Output[r] = ρ · Output[r-1] + sqrt(1-ρ²) · samples_runs[r],
    matching marginal voxel variance (stationary AR(1)).
    """
    n_runs, V_k = samples_runs.shape
    out = np.empty_like(samples_runs)
    out[0] = samples_runs[0]
    scale = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    for r in range(1, n_runs):
        out[r] = rho * out[r - 1] + scale * samples_runs[r]
    return out


# ---------------------------------------------------------------------------
# Top-level voxel synthesis
# ---------------------------------------------------------------------------
def synthesize_voxel_response(W_k: np.ndarray, beta_s: float, beta_c: float,
                              family: str, theta_canonical: np.ndarray,
                              noise_params: dict, n_runs: int = N_RUNS,
                              rng: Optional[np.random.Generator] = None,
                              basis_type: str = DEFAULT_BASIS) -> np.ndarray:
    """Generate (n_runs, 8, V_k) voxel response under GT (β_s, β_c).

    For each (run, c):
        Y[run, c, :] = W_k @ C(θ_c + δθ(c; β_s, β_c)) + ε[run, c, :]
    where ε is spatially correlated (top-R PCs of HC residual) and AR(1)
    across runs.

    If (β_s, β_c) == (0, 0): output is the null carrier — HC_k's clean
    baseline + noise (Source A semantics).

    Args:
        W_k:              (K, V_k) HC encoder.
        beta_s, beta_c:   GT 2-component parameters in degrees.
        family:           'protan' | 'deutan'.
        theta_canonical:  (8,) canonical hue degrees (typically HUE_ANGLES).
        noise_params:     output of `estimate_noise_per_hc`.
        n_runs:           number of runs (default 6, matches production).
        rng:              numpy Generator; if None, fresh `default_rng()`.
        basis_type:       basis for `create_basis_full` (default 'fe').

    Returns:
        Y_synth: (n_runs, 8, V_k).
    """
    if rng is None:
        rng = np.random.default_rng()
    K, V_k = W_k.shape
    assert noise_params["V_k"] == V_k, "noise_params V_k mismatch"
    delta = delta_theta_2comp(theta_canonical, beta_s, beta_c, family)
    shifted = (theta_canonical + delta) % 360.0
    basis_full = create_basis_full(K, basis_type=basis_type)
    idx = np.round(shifted).astype(int) % 360
    C_shift = basis_full[idx]                              # (8, K)
    mean_pattern = C_shift @ W_k                           # (8, V_k)

    Y = np.empty((n_runs, N_COLORS, V_k), dtype=float)
    L = noise_params["spatial_loading"]
    rho = noise_params["temporal_ar1_rho"]
    for c in range(N_COLORS):
        spatial_draws = _sample_spatially_correlated(L, n_runs, rng)  # (n_runs, V_k)
        eps = _apply_ar1(spatial_draws, rho, rng)
        Y[:, c, :] = mean_pattern[c][None, :] + eps
    return Y


# ---------------------------------------------------------------------------
# Convenience: provenance block for output JSONs
# ---------------------------------------------------------------------------
def synth_provenance() -> dict:
    """Return a dict describing the synth configuration for output JSONs."""
    return {
        "module": "forward_voxel_synth.py",
        "forward_map": "Y[r,c,:] = W_k @ C(θ_c + δθ_2comp(c; β_s, β_c, family)) + ε",
        "spatial_cov_rank": int(SPATIAL_COV_RANK),
        "temporal_ar1_rho": float(AR1_RHO),
        "basis_type": DEFAULT_BASIS,
        "encoder": "ridge_gcv (matches neural_loss.precompute_loco_W_within)",
        "two_comp_source": "scripts/two_comp.py forward_2comp (closure forward)",
        "theta_conf": dict(THETA_CONF),
        "rationale": (
            "voxel-level injection per Wilson & Collins 2019 / Schütt 2021; "
            "supersedes loss-function-level Method C used in exp14/15/18/21/22."
        ),
    }


def synthesize_fake_jnd(beta_s: float, beta_c: float, family: str,
                        pool_jnd_baseline: dict, pool_jnd_sd: dict,
                        rng: np.random.Generator,
                        theta_canonical: Optional[np.ndarray] = None) -> dict:
    """Behaviorally-consistent fake CVD JND at GT (β_s, β_c).

    Mirrors the v6 γ-atom forward model:
        d_perceived(i,j) at GT = perceived hue distance under δθ_2comp(GT)
        pred_jnd          = pool_baseline * (d_phys / d_perceived)
        fake_jnd          = pred_jnd + N(0, pool_sd)

    The pool baseline/SD must come from the same pool that v6 will use
    internally (HC \\ {donor} under the LOO patch). Pairs whose pool baseline
    or SD is None are returned as None (caller forwards as missing).

    Args:
        beta_s, beta_c:     GT 2-component parameters in degrees.
        family:             'protan' | 'deutan'.
        pool_jnd_baseline:  dict {pair_name: float|None}.
        pool_jnd_sd:        dict {pair_name: float|None}.
        rng:                numpy Generator (caller controls determinism).
        theta_canonical:    (8,) hue angles for c1..c8; defaults to HUE_ANGLES.

    Returns:
        dict {pair_name: float|None} matching `load_jnd_per_pair` format.

    Notes:
        This function fixes the γ-atom inconsistency in the prior recovery
        design where fake_jnd was donor's REAL JND (effectively GT=0 for
        behaviour), creating γ-pull to δ=0 while voxel signal was at δ≠0.
    """
    # Late import to avoid circulars at module load.
    from behav_loss import PAIR_HUES  # noqa: WPS433

    if theta_canonical is None:
        theta_canonical = HUE_ANGLES.astype(float)

    delta = delta_theta_2comp(theta_canonical, beta_s, beta_c, family)
    perceived = (theta_canonical + delta) % 360.0

    fake: dict = {}
    for p_name, (theta_a, theta_b) in PAIR_HUES.items():
        bl = pool_jnd_baseline.get(p_name)
        sd = pool_jnd_sd.get(p_name)
        if bl is None or sd is None:
            fake[p_name] = None
            continue
        i = int(round(theta_a / 45.0)) % 8
        j = int(round(theta_b / 45.0)) % 8
        d_phys = min(abs(theta_a - theta_b) % 360,
                      360 - abs(theta_a - theta_b) % 360)
        d_perc_raw = abs(perceived[i] - perceived[j]) % 360
        d_perc = max(min(d_perc_raw, 360 - d_perc_raw), 1e-3)
        pred = float(bl) * (d_phys / d_perc)
        noise = float(rng.normal(0, max(float(sd), 1e-3)))
        fake[p_name] = float(pred + noise)
    return fake


__all__ = [
    "delta_theta_2comp",
    "precompute_per_hc_W",
    "estimate_noise_per_hc",
    "synthesize_voxel_response",
    "synthesize_fake_jnd",
    "synth_provenance",
    "SPATIAL_COV_RANK",
    "AR1_RHO",
]
