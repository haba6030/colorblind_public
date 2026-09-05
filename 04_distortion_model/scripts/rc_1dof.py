"""S2: rc_1dof.py — R+C 1-DOF (Δλ external + cortical g) forward, fit, inverse.

R+C model:
  δθ_Machado(c; Δλ) = retinal cone-shift distortion per color (Stockman coord)
  δθ_RC(c; Δλ, g) = (2 − g) · δθ_Machado(c; Δλ)   ← cortical linear compensation

g interpretation (advisor corrected 2026-05-21):
  g ∈ [0, 3] amplification convention:
    g = 1  : HC-like passthrough (no cortical compensation)
    g = 2  : Full cortical compensation (δθ_RC = 0)
    g = 3  : Overcompensation (δθ_RC = −δθ_Machado)
    g = 0  : Cortical attenuation (doubles retinal distortion)

Compensation magnitude: M = max(0, g − 1) · Δλ (retinal-equivalent nm).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/rc_1dof.py  (development repository, commit 53c81c2)
# Role      : Retinal-plus-cortical-gain comparison model (one free gain g in [0, 3]).
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
import json
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from utils_distortion_models import apply_distortion

HUE_CANON = np.arange(0, 360, 45, dtype=float)  # 8 canonical hues
G_MIN, G_MAX, G_STEP = 0.0, 3.0, 0.05  # 61 grid points


def delta_machado(delta_lambda: float, cvd_family: str) -> np.ndarray:
    """Machado retinal δθ 8-vec (relative to Δλ=0 baseline in Stockman coord)."""
    hue_shifted = apply_distortion('machado_1way', [delta_lambda], cvd_type=cvd_family)
    hue_baseline = apply_distortion('machado_1way', [0.0], cvd_type=cvd_family)
    delta = (hue_shifted - hue_baseline + 180.0) % 360.0 - 180.0
    return delta


def forward_rc(delta_lambda: float, g: float, cvd_family: str) -> np.ndarray:
    """R+C forward: δθ_RC(c) = (2 − g) · δθ_Machado(c) — 8-vec in degrees."""
    delta_m = delta_machado(delta_lambda, cvd_family)
    return (2.0 - g) * delta_m


def g_grid() -> np.ndarray:
    return np.arange(G_MIN, G_MAX + G_STEP / 2, G_STEP)


def fit_rc_g(delta_lambda: float, cvd_family: str, loss_fn) -> dict:
    """Grid search g for given Δλ and loss function.

    Args:
        delta_lambda: cone shift in nm
        cvd_family: 'protan' | 'deutan'
        loss_fn: callable (delta_rc_8vec) -> float scalar loss

    Returns:
        dict with g_best, loss_best, boundary flags, full grid losses
    """
    grid = g_grid()
    losses = np.zeros_like(grid)
    for i, g in enumerate(grid):
        delta_rc = forward_rc(delta_lambda, g, cvd_family)
        losses[i] = float(loss_fn(delta_rc))
    best_idx = int(np.argmin(losses))
    return {
        'g_best': float(grid[best_idx]),
        'loss_best': float(losses[best_idx]),
        'loss_at_g1': float(losses[int(np.searchsorted(grid, 1.0))]),
        'boundary_low': bool(best_idx == 0),
        'boundary_high': bool(best_idx == len(grid) - 1),
        'misspecified': bool(best_idx == 0 or best_idx == len(grid) - 1),
        'grid_min': float(grid[0]),
        'grid_max': float(grid[-1]),
        'grid_step': float(G_STEP),
        'grid': grid.tolist(),
        'losses': losses.tolist(),
    }


def compensation_magnitude(g: float, delta_lambda: float) -> float:
    """M = max(0, g − 1) · Δλ (retinal-equivalent nm)."""
    return max(0.0, g - 1.0) * delta_lambda


def inverse_rc_check(delta_lambda: float, g: float, cvd_family: str,
                     tol_deg: float = 1.0) -> dict:
    """Bijection check: forward map at 8 canonical → unique perceived hues?

    For filter design, the forward map θ_stim → θ_perceived = θ + δθ_RC(θ) must
    be bijective on the 8 canonical inputs for the inverse (filter) to exist.

    Args:
        tol_deg: tolerance for "unique" within hue space
    Returns:
        dict with n_unique, max_collision, exact_8_of_8 flag
    """
    delta_rc = forward_rc(delta_lambda, g, cvd_family)
    perceived = (HUE_CANON + delta_rc) % 360.0

    # Pairwise distances on hue circle
    n = len(perceived)
    min_pair_dist = 360.0
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(perceived[i] - perceived[j])
            d = min(d, 360.0 - d)
            min_pair_dist = min(min_pair_dist, d)

    # Each target_hue ∈ canonical: find filter input that maps to it
    filter_inputs = []
    forward_errors = []
    for target in HUE_CANON:
        diffs = np.abs(((perceived - target + 180.0) % 360.0) - 180.0)
        best = int(np.argmin(diffs))
        filter_inputs.append(float(HUE_CANON[best]))
        forward_errors.append(float(diffs[best]))

    n_within_tol = sum(1 for e in forward_errors if e <= tol_deg)
    return {
        'min_perceived_pair_dist_deg': float(min_pair_dist),
        'bijection_ok': bool(min_pair_dist > tol_deg),
        'inverse_8_of_8_within_tol': int(n_within_tol),
        'tol_deg': float(tol_deg),
        'forward_errors_per_target': forward_errors,
        'filter_inputs_per_target': filter_inputs,
        'perceived_8vec': perceived.tolist(),
    }


# ============================================================================
# Self-test (loss-agnostic) — Stage 2 validation
# ============================================================================

def self_test():
    """S2 Stage 2: validation without loss function."""
    results = {}
    print("=" * 78)
    print("S2 self-test (loss-agnostic validation)")
    print("=" * 78)

    # Test 1: HC baseline (Δλ=0, g=1) → δθ all zeros
    delta_baseline = forward_rc(0.0, 1.0, 'deutan')
    test1 = {
        'name': 'HC baseline (Δλ=0, g=1) → δθ ≈ 0',
        'delta_max_abs': float(np.max(np.abs(delta_baseline))),
        'pass': bool(np.max(np.abs(delta_baseline)) < 0.01),
    }
    results['test1_hc_baseline'] = test1
    print(f"\n[Test 1] HC baseline (Δλ=0, g=1):")
    print(f"  δθ = {delta_baseline.round(3).tolist()}")
    print(f"  max|δθ| = {test1['delta_max_abs']:.4f}°")
    print(f"  {'✓ PASS' if test1['pass'] else '✗ FAIL'}")

    # Test 2: Cortical compensation (g=2) at Δλ>0 → δθ=0 (full restoration)
    delta_comp = forward_rc(10.0, 2.0, 'protan')
    test2 = {
        'name': 'Full compensation (g=2) at Δλ=10 → δθ ≈ 0',
        'delta_max_abs': float(np.max(np.abs(delta_comp))),
        'pass': bool(np.max(np.abs(delta_comp)) < 0.01),
    }
    results['test2_full_compensation'] = test2
    print(f"\n[Test 2] Full compensation (Δλ=10, g=2, protan):")
    print(f"  δθ = {delta_comp.round(3).tolist()}")
    print(f"  max|δθ| = {test2['delta_max_abs']:.4f}°")
    print(f"  {'✓ PASS' if test2['pass'] else '✗ FAIL'}")

    # Test 3: Raw Machado (g=1) at Δλ>0 — magnitudes expected non-trivial
    delta_machado_only = forward_rc(10.0, 1.0, 'protan')
    test3 = {
        'name': 'Raw Machado (g=1) at Δλ=10 (protan) → δθ ≠ 0',
        'delta_rms_deg': float(np.sqrt(np.mean(delta_machado_only ** 2))),
        'pass': bool(np.sqrt(np.mean(delta_machado_only ** 2)) > 1.0),
    }
    results['test3_machado_magnitude'] = test3
    print(f"\n[Test 3] Raw Machado (Δλ=10, g=1, protan):")
    print(f"  δθ = {delta_machado_only.round(2).tolist()}")
    print(f"  RMS = {test3['delta_rms_deg']:.2f}°")
    print(f"  {'✓ PASS' if test3['pass'] else '✗ FAIL'}")

    # Test 4: Linear scaling check — (2-g) factor
    delta_m_only = delta_machado(10.0, 'protan')
    delta_g05 = forward_rc(10.0, 0.5, 'protan')  # factor 1.5
    expected_g05 = 1.5 * delta_m_only
    test4 = {
        'name': 'Linear scaling (2-g) factor: g=0.5 → δθ = 1.5·δθ_Machado',
        'max_diff': float(np.max(np.abs(delta_g05 - expected_g05))),
        'pass': bool(np.max(np.abs(delta_g05 - expected_g05)) < 0.01),
    }
    results['test4_linear_scaling'] = test4
    print(f"\n[Test 4] Linear scaling at g=0.5:")
    print(f"  δθ predicted = 1.5 · δθ_M = {expected_g05.round(3).tolist()}")
    print(f"  δθ actual = {delta_g05.round(3).tolist()}")
    print(f"  max diff = {test4['max_diff']:.4f}°")
    print(f"  {'✓ PASS' if test4['pass'] else '✗ FAIL'}")

    # Test 5: Grid search with dummy loss (L = sum δθ²) should give g=2 (compensation)
    def dummy_loss(delta_rc):
        return float(np.sum(delta_rc ** 2))

    fit_result = fit_rc_g(10.0, 'protan', dummy_loss)
    test5 = {
        'name': 'Grid search with sum(δθ²) loss → g* ≈ 2.0 (full compensation)',
        'g_best': fit_result['g_best'],
        'pass': bool(abs(fit_result['g_best'] - 2.0) < 0.1),
    }
    results['test5_dummy_grid'] = test5
    print(f"\n[Test 5] Dummy grid search (sum δθ² minimization):")
    print(f"  g_best = {fit_result['g_best']:.3f}")
    print(f"  Expected ≈ 2.0 (full compensation)")
    print(f"  {'✓ PASS' if test5['pass'] else '✗ FAIL'}")

    # Test 6: Inverse bijection — perceived 8-vec must be UNIQUE (min pair dist > tol)
    # Note: "8/8 exact filter to canonical target" is a different claim that requires
    # continuous-hue interpolation (deferred to later sprint). Here we check the
    # *necessary condition for inverse to exist*: forward map is bijective on 8 inputs.
    inv = inverse_rc_check(10.0, 1.5, 'protan')
    test6 = {
        'name': 'R+C forward bijection (Δλ=10, g=1.5, protan) — min pair dist > 1°',
        'min_pair_dist': inv['min_perceived_pair_dist_deg'],
        'bijection_ok': inv['bijection_ok'],
        'pass': bool(inv['bijection_ok']),
    }
    results['test6_bijection'] = test6
    print(f"\n[Test 6] R+C forward bijection (Δλ=10, g=1.5, protan):")
    print(f"  Min perceived pair distance: {inv['min_perceived_pair_dist_deg']:.2f}°")
    print(f"  Bijection (>1° tol): {inv['bijection_ok']}")
    print(f"  {'✓ PASS' if test6['pass'] else '✗ FAIL'}")

    # Test 7: Boundary detection — g at boundary 0 should be flagged
    def biased_loss(delta_rc):
        # Larger |δθ| → smaller loss → drive g toward 0
        return -float(np.sum(delta_rc ** 2))

    boundary_fit = fit_rc_g(10.0, 'protan', biased_loss)
    test7 = {
        'name': 'Boundary detection (low) — biased loss should hit g=0',
        'g_best': boundary_fit['g_best'],
        'boundary_low': boundary_fit['boundary_low'],
        'pass': bool(boundary_fit['boundary_low']),
    }
    results['test7_boundary_low'] = test7
    print(f"\n[Test 7] Boundary detection (low) with biased loss:")
    print(f"  g_best = {boundary_fit['g_best']:.3f}, boundary_low = {boundary_fit['boundary_low']}")
    print(f"  {'✓ PASS' if test7['pass'] else '✗ FAIL'}")

    # Summary
    all_pass = all(t.get('pass', False) for t in results.values())
    print(f"\n{'=' * 78}")
    print(f"S2 self-test summary: {'✓ ALL PASS' if all_pass else '✗ FAILURES'}")
    print(f"{'=' * 78}")

    # Print summary table per subject (sub-08, sub-09 with each Δλ source)
    print(f"\n--- Forward R+C magnitudes per subject × Δλ source (at g=1 raw Machado) ---")
    subjects = [
        ('sub-08', 'deutan', {'DPS': 6.0, 'JND_Lamb': 6.5, 'Boehm_8': 8.0}),
        ('sub-09', 'protan', {'DPS': 10.0, 'JND_Lamb': 1.5, 'Boehm_3': 3.0}),
    ]
    print(f"{'Subject':<10} {'Family':<8} {'Δλ source':<15} {'RMS δθ (deg)':<14} {'max|δθ|':<10}")
    for subj, fam, sources in subjects:
        for src_name, dl in sources.items():
            delta = forward_rc(dl, 1.0, fam)
            rms = float(np.sqrt(np.mean(delta ** 2)))
            max_abs = float(np.max(np.abs(delta)))
            print(f"{subj:<10} {fam:<8} {src_name:<15} {rms:<14.2f} {max_abs:<10.2f}")

    return results, all_pass


if __name__ == "__main__":
    results, all_pass = self_test()

    # Save self-test results
    OUT_DIR = SCRIPT_DIR.parent / "results" / "rc_1dof"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "self_test_results.json", 'w') as f:
        # Trim large grid arrays from results for readability
        clean_results = {}
        for k, v in results.items():
            clean_results[k] = {kk: vv for kk, vv in v.items() if kk not in ('grid', 'losses', 'forward_errors_per_target', 'filter_inputs_per_target', 'perceived_8vec')}
        json.dump({'all_pass': all_pass, 'tests': clean_results}, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'self_test_results.json'}")

    sys.exit(0 if all_pass else 1)
