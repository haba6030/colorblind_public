"""two_comp.py — ★ CLOSURE 2-Component forward (single source of truth).

This is the forward map used by PIPELINE_2_CLOSURE.md (2026-05-27) Phase B v6
and every downstream closure step. All viz, post-hoc analysis, and Phase 3
stimulus synthesis MUST use `forward_2comp` (or its identical inlined form).

  δθ(θ) = β_s · cos(θ − 90°) + β_c · cos(θ − θ_conf)
  θ_perceived = (θ + δθ) mod 360°

Input convention: θ is CIElab nominal hue (0°, 45°, ..., 315° for c1..c8) —
applied directly to the trig, with NO h_base / Stockman opponent conversion.

  θ_conf:
    protan: 16°
    deutan: 150°

  β_s : S-(L+M) cardinal axis amplitude (Krauskopf 1982 convention)
  β_c : confusion-axis amplitude (Stockman cone fundamentals confusion line)

Live callers (closure-active):
  scripts/s10b_v6_pca_rdm.py:31, :231, :607  (Phase B v6 main runner)
  scripts/s17_hc_loo.py                       (strict HC LOO 7-fold)
  scripts/s13_round3.py                       (Phase D Round 3 multi-point sim)
  scripts/s12b_phase_c_v2.py                  (deprecated, L8 audit only)
  scripts/p2_primary_4col.py                  (closure-consistent 4-col viz)

⚠ DO NOT replace with the h_base / frozen variant in
  `scripts/forward_models/two_component.py` ⚠
The two functions give DIFFERENT 8-vec δθ at the same (β_s, β_c). Example
for deutan (38, −10):
    closure (here):        [+8.66, +29.46, +33.0, +17.21,
                            −8.66, −29.46, −33.0, −17.21]
    forward_models/two_component (frozen):
                           [−16.0, −23.7, −28.7, −31.8,
                            −33.8, −6.7, +31.8, +15.6]
See CLAUDE.md A13 for the project rule.

Sanity check (deutan, β_s=38, β_c=−10):
    forward_2comp(38, -10, 'deutan').round(2) ==
        array([8.66, 29.46, 33., 17.21, -8.66, -29.46, -33., -17.21])
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/two_comp.py  (development repository, commit 53c81c2)
# Role      : Two-component forward model, grid definition (BS_GRID, BC_GRID), theta_conf per subtype.
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

HUE_CANON = np.arange(0, 360, 45, dtype=float)
THETA_CONF = {'protan': 16.0, 'deutan': 150.0}

BS_GRID = np.arange(0.0, 50.0 + 1e-9, 2.0)        # 26 points
BC_GRID = np.arange(-50.0, 50.0 + 1e-9, 2.0)      # 51 points


def forward_2comp(beta_s: float, beta_c: float, cvd_family: str,
                   hues: np.ndarray = HUE_CANON) -> np.ndarray:
    """δθ(c) 8-vec for 2-Component cortical rotation model."""
    theta_conf = THETA_CONF[cvd_family]
    rad = np.deg2rad(hues)
    rad_s = np.deg2rad(hues - 90.0)
    rad_c = np.deg2rad(hues - theta_conf)
    delta = beta_s * np.cos(rad_s) + beta_c * np.cos(rad_c)
    return delta


def grid_2comp() -> tuple:
    return BS_GRID, BC_GRID


def fit_2comp(cvd_family: str, loss_fn) -> dict:
    """Grid search (β_s, β_c) for given loss function callable."""
    losses = np.zeros((len(BS_GRID), len(BC_GRID)))
    for i, bs in enumerate(BS_GRID):
        for j, bc in enumerate(BC_GRID):
            delta = forward_2comp(bs, bc, cvd_family)
            losses[i, j] = float(loss_fn(delta))
    best_idx = np.unravel_index(np.argmin(losses), losses.shape)
    bs_best = float(BS_GRID[best_idx[0]])
    bc_best = float(BC_GRID[best_idx[1]])
    return {
        'beta_s_best': bs_best,
        'beta_c_best': bc_best,
        'loss_best': float(losses[best_idx]),
        'loss_at_zero': float(losses[0, np.searchsorted(BC_GRID, 0.0)]),
        'boundary_bs': bool(best_idx[0] == 0 or best_idx[0] == len(BS_GRID) - 1),
        'boundary_bc': bool(best_idx[1] == 0 or best_idx[1] == len(BC_GRID) - 1),
        'delta_theta_at_best': forward_2comp(bs_best, bc_best, cvd_family).tolist(),
    }


if __name__ == "__main__":
    # Quick sanity check
    print("2-Component forward — sanity")
    print(f"  Grid: β_s {len(BS_GRID)} pts, β_c {len(BC_GRID)} pts = {len(BS_GRID) * len(BC_GRID)} combinations")
    print(f"\nHC baseline (β_s=0, β_c=0): δθ = {forward_2comp(0, 0, 'protan').round(3).tolist()}")
    print(f"S-cone only (β_s=20, β_c=0) protan: δθ = {forward_2comp(20, 0, 'protan').round(2).tolist()}")
    print(f"Confusion-axis only (β_s=0, β_c=20) protan: δθ = {forward_2comp(0, 20, 'protan').round(2).tolist()}")
    print(f"Sub-09 Cycle 12-like (β_s=30, β_c=26) protan: δθ = {forward_2comp(30, 26, 'protan').round(2).tolist()}")
