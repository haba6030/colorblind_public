"""S3: behav_loss.py — Behavioral loss functions for R+C and 2-Comp models.

Provides:
  L_behav_α  (per-color 8AFC softmax MSE)
  L_behav_γ  (per-pair JND weighted MSE)
  L_behav    (subject-specific composite: sub-08 0.5α+0.5γ, sub-09 0γ+1γ)

Forward map input: δθ(c) 8-vec (degrees, perceived rotation per canonical hue).
Outputs: scalar loss.

σ_HC = 21° fixed (HC pooled empirical + Schurgin 2020 / Maloney 2010 literature 18-25°).

Sub-09 8AFC weight = 0 justification: 64/64 = 100% accuracy ceiling, L_α uninformative.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/behav_loss.py  (development repository, commit 53c81c2)
# Role      : Psychophysical JND loss atom L_gamma.
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
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[2]
DATA_DIR = ROOT / "data" / "behavior"
OUT_DIR = SCRIPT_DIR.parent / "results" / "behav_loss"

HUES = np.arange(0, 360, 45, dtype=float)
SIGMA_HC = 21.0  # HC pooled empirical (4 HC, 255 trials) + literature range 18-25°

PAIR_HUES = {
    'red-orange':    (  0.0,  45.0),
    'orange-yellow': ( 45.0,  90.0),
    'yellow-green':  ( 90.0, 135.0),
    'green-blue':    (135.0, 225.0),
    'blue-purple':   (225.0, 270.0),
    'yellow-purple': ( 90.0, 270.0),
    'cyan-magenta':  (180.0, 315.0),
    'red-cyan':      (  0.0, 180.0),
}

# Subject-specific behavioral weights (advisor catch — uniform across subjects 위해
# sub-08 도 L_α weight halved, sub-09 는 8AFC ceiling 으로 w_α=0)
BEHAV_WEIGHTS = {
    'sub-08': (0.5, 0.5),   # (w_α, w_γ)
    'sub-09': (0.0, 1.0),
    # HC subjects: default 0.5/0.5 if data exists, fallback to γ-only
}

HC_8AFC_SUBJS = ['sub-01', 'sub-03', 'sub-06', 'sub-07']  # 8AFC RSVP available
HC_JND_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']  # JND all


def hue_dist_mod(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360.0 - d)


# ============================================================================
# Data loaders
# ============================================================================

def load_8afc_per_color(subject: str) -> np.ndarray:
    """8-vec of per-color accuracy (correct rate) from RSVP CSV."""
    csv = DATA_DIR / f"{subject}_rsvp_8afc_ses1_run1.csv"
    df = pd.read_csv(csv)
    df = df[df['rt'] > 0]
    acc = np.zeros(8)
    n = np.zeros(8)
    for _, row in df.iterrows():
        try:
            i = int(row['stimulus_label'].split('_')[1]) - 1
            acc[i] += int(row['correct'])
            n[i] += 1
        except Exception:
            continue
    return acc / np.maximum(n, 1)


def load_jnd_per_pair(subject: str) -> dict:
    """Mean JND per pair (averaging sc0, sc1 staircases)."""
    csv = DATA_DIR / f"{subject}_jnd_ses1_no_filter_summary.csv"
    df = pd.read_csv(csv)
    out = {}
    for p in PAIR_HUES.keys():
        sub = df[df['pair_name'] == p]
        out[p] = float(sub['jnd_mean'].mean()) if len(sub) > 0 else None
    return out


def compute_hc_jnd_baseline() -> tuple:
    """HC pool JND mean ± SD per pair across N=7 HC."""
    hc_data = [load_jnd_per_pair(s) for s in HC_JND_SUBJS]
    baseline = {}
    sd = {}
    for p in PAIR_HUES.keys():
        vals = [d[p] for d in hc_data if d[p] is not None]
        baseline[p] = float(np.mean(vals))
        sd[p] = float(np.std(vals, ddof=1))
    return baseline, sd


# ============================================================================
# L_behav_α: per-color 8AFC softmax MSE
# ============================================================================

def softmax_pred_8afc(delta_rc_8vec: np.ndarray, sigma: float = SIGMA_HC) -> np.ndarray:
    """P(response=j | stim=i) ∝ exp(−d(perceived_i, target_j)² / σ²).

    Returns 8×8 confusion probability matrix.
    """
    perceived = (HUES + delta_rc_8vec) % 360.0
    P = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            d = hue_dist_mod(perceived[i], HUES[j])
            P[i, j] = np.exp(-(d ** 2) / (sigma ** 2))
        P[i] /= P[i].sum()
    return P


def L_behav_alpha(delta_rc: np.ndarray, obs_acc_8vec: np.ndarray,
                  sigma: float = SIGMA_HC) -> float:
    """Per-color 8AFC accuracy MSE: mean((P(correct|i) − obs_acc[i])²)."""
    P = softmax_pred_8afc(delta_rc, sigma)
    pred_acc = np.diagonal(P)
    return float(np.mean((pred_acc - obs_acc_8vec) ** 2))


# ============================================================================
# L_behav_γ: per-pair JND weighted MSE
# ============================================================================

def predict_jnd_per_pair(delta_rc: np.ndarray, hc_baseline: dict) -> dict:
    """Predict JND_AT(p) per pair under δθ = delta_rc.

    JND_AT(p) = HC_baseline(p) × (d_physical / d_perceived)
    """
    perceived = (HUES + delta_rc) % 360.0
    pred = {}
    for p, (theta_a, theta_b) in PAIR_HUES.items():
        i = int(round(theta_a / 45.0)) % 8
        j = int(round(theta_b / 45.0)) % 8
        d_phys = hue_dist_mod(theta_a, theta_b)
        d_perc = hue_dist_mod(perceived[i], perceived[j])
        d_perc = max(d_perc, 1e-3)
        pred[p] = hc_baseline[p] * (d_phys / d_perc)
    return pred


def L_behav_gamma(delta_rc: np.ndarray, jnd_obs: dict,
                   hc_baseline: dict, hc_sd: dict) -> float:
    """Per-pair JND weighted MSE: mean[((pred − obs) / σ_p)²]."""
    pred = predict_jnd_per_pair(delta_rc, hc_baseline)
    loss = 0.0
    n = 0
    for p in PAIR_HUES.keys():
        if jnd_obs.get(p) is None or pred.get(p) is None:
            continue
        sigma_p = max(hc_sd[p], 1e-3)
        loss += ((pred[p] - jnd_obs[p]) / sigma_p) ** 2
        n += 1
    return loss / max(n, 1)


# ============================================================================
# L_behav: composite (subject-specific weighting)
# ============================================================================

def L_behav_composite(delta_rc: np.ndarray, subject: str,
                       obs_acc_8vec: np.ndarray, jnd_obs: dict,
                       hc_baseline: dict, hc_sd: dict,
                       sigma: float = SIGMA_HC) -> dict:
    """Subject-specific α + γ composite. Returns dict with parts + total."""
    w_alpha, w_gamma = BEHAV_WEIGHTS.get(subject, (0.5, 0.5))
    L_a = L_behav_alpha(delta_rc, obs_acc_8vec, sigma) if w_alpha > 0 else 0.0
    L_g = L_behav_gamma(delta_rc, jnd_obs, hc_baseline, hc_sd)
    return {
        'L_alpha': L_a,
        'L_gamma': L_g,
        'L_total': w_alpha * L_a + w_gamma * L_g,
        'w_alpha': w_alpha,
        'w_gamma': w_gamma,
    }


# ============================================================================
# Self-test (Stage 2 validation)
# ============================================================================

def self_test():
    """S3 Stage 2: validation. Tests behavioral loss against expected behavior."""
    print("=" * 78)
    print("S3 self-test — behav_loss validation")
    print("=" * 78)

    hc_baseline, hc_sd = compute_hc_jnd_baseline()
    delta_zero = np.zeros(8)
    results = {}

    # Test 1: HC self-fit (δθ=0) → L_α small for each HC with 8AFC data
    print("\n[Test 1] HC self-fit L_α at δθ=0 (perceived = canonical):")
    hc_alpha_losses = []
    for h in HC_8AFC_SUBJS:
        obs_acc = load_8afc_per_color(h)
        L_a = L_behav_alpha(delta_zero, obs_acc)
        hc_alpha_losses.append(L_a)
        print(f"  {h}: obs_acc = {obs_acc.round(3).tolist()}, L_α = {L_a:.5f}")
    max_hc_alpha = max(hc_alpha_losses)
    test1 = {
        'name': 'HC self-fit L_α (δθ=0)',
        'max_hc_loss': max_hc_alpha,
        'pass': bool(max_hc_alpha < 0.01),  # ≤ 1% MSE in accuracy
    }
    results['test1_hc_alpha'] = test1
    print(f"  Max HC L_α = {max_hc_alpha:.5f}, threshold 0.01")
    print(f"  {'✓ PASS' if test1['pass'] else '✗ FAIL'}")

    # Test 2: HC self-fit L_γ at δθ=0 → mean ~ chi-square / n_pairs since obs are
    # individual HCs around pool mean (expected MSE ≈ sample variance / sd² ≈ 1)
    print(f"\n[Test 2] HC self-fit L_γ at δθ=0 (JND ratio = 1, pred = baseline):")
    hc_gamma_losses = []
    for h in HC_JND_SUBJS:
        jnd_obs = load_jnd_per_pair(h)
        L_g = L_behav_gamma(delta_zero, jnd_obs, hc_baseline, hc_sd)
        hc_gamma_losses.append(L_g)
        print(f"  {h}: L_γ = {L_g:.4f}")
    mean_hc_gamma = np.mean(hc_gamma_losses)
    # Each HC's z-score on each pair ~ N(0,1) if HC is representative → L_γ ~ mean of z² ~ 1
    test2 = {
        'name': 'HC self-fit L_γ at δθ=0 (mean ~ 1 = z² unit)',
        'mean_hc_loss': float(mean_hc_gamma),
        'pass': bool(0.3 < mean_hc_gamma < 3.0),  # within reasonable z² range
    }
    results['test2_hc_gamma'] = test2
    print(f"  Mean HC L_γ = {mean_hc_gamma:.4f}, expected ~1 (z² unit)")
    print(f"  {'✓ PASS' if test2['pass'] else '✗ FAIL'}")

    # Test 3: CVD L_α at δθ=0 should be LARGE for sub-08 (82.5% acc != ~97% predicted)
    print(f"\n[Test 3] Sub-08 L_α at δθ=0 (mismatch: pred ~97% vs obs 82.5%):")
    obs_sub08 = load_8afc_per_color('sub-08')
    L_a_sub08 = L_behav_alpha(delta_zero, obs_sub08)
    test3 = {
        'name': 'Sub-08 L_α at δθ=0 large vs HC',
        'sub08_loss': float(L_a_sub08),
        'max_hc_loss': max_hc_alpha,
        'pass': bool(L_a_sub08 > max_hc_alpha),
    }
    results['test3_sub08_alpha_large'] = test3
    print(f"  obs_acc = {obs_sub08.round(3).tolist()}")
    print(f"  L_α(sub-08) = {L_a_sub08:.5f}, max HC L_α = {max_hc_alpha:.5f}")
    print(f"  {'✓ PASS' if test3['pass'] else '✗ FAIL'}")

    # Test 4: Sub-09 L_α at δθ=0 should be SMALL (ceiling, pred 97% ≈ obs 100%)
    print(f"\n[Test 4] Sub-09 L_α at δθ=0 (ceiling: pred ~97% vs obs 100%):")
    obs_sub09 = load_8afc_per_color('sub-09')
    L_a_sub09 = L_behav_alpha(delta_zero, obs_sub09)
    test4 = {
        'name': 'Sub-09 L_α at δθ=0 small (ceiling)',
        'sub09_loss': float(L_a_sub09),
        'pass': bool(L_a_sub09 < 0.001),  # very small if ceiling
    }
    results['test4_sub09_alpha_small'] = test4
    print(f"  obs_acc = {obs_sub09.round(3).tolist()}")
    print(f"  L_α(sub-09) = {L_a_sub09:.6f}, expected < 0.001")
    print(f"  {'✓ PASS' if test4['pass'] else '✗ FAIL'}")

    # Test 5: Sub-08 L_γ at δθ=0 → should be large (yellow-axis HYPO)
    print(f"\n[Test 5] Sub-08 L_γ at δθ=0 (yellow-axis HYPO mismatch):")
    jnd_sub08 = load_jnd_per_pair('sub-08')
    L_g_sub08 = L_behav_gamma(delta_zero, jnd_sub08, hc_baseline, hc_sd)
    test5 = {
        'name': 'Sub-08 L_γ at δθ=0 >> HC mean',
        'sub08_loss': float(L_g_sub08),
        'mean_hc_loss': float(mean_hc_gamma),
        'pass': bool(L_g_sub08 > 2 * mean_hc_gamma),
    }
    results['test5_sub08_gamma_large'] = test5
    print(f"  L_γ(sub-08) = {L_g_sub08:.4f} vs HC mean {mean_hc_gamma:.4f}")
    print(f"  {'✓ PASS' if test5['pass'] else '✗ FAIL'}")

    # Test 6: Composite L_behav subject-specific weights
    print(f"\n[Test 6] L_behav composite (subject-specific weights):")
    obs_sub08 = load_8afc_per_color('sub-08')
    jnd_sub08 = load_jnd_per_pair('sub-08')
    obs_sub09 = load_8afc_per_color('sub-09')
    jnd_sub09 = load_jnd_per_pair('sub-09')

    comp08 = L_behav_composite(delta_zero, 'sub-08', obs_sub08, jnd_sub08, hc_baseline, hc_sd)
    comp09 = L_behav_composite(delta_zero, 'sub-09', obs_sub09, jnd_sub09, hc_baseline, hc_sd)
    test6 = {
        'name': 'L_behav composite weights applied',
        'sub08': comp08,
        'sub09': comp09,
        'pass': bool(comp08['w_alpha'] == 0.5 and comp09['w_alpha'] == 0.0),
    }
    results['test6_composite_weights'] = test6
    print(f"  sub-08 (w_α=0.5, w_γ=0.5): L_α={comp08['L_alpha']:.5f}, L_γ={comp08['L_gamma']:.4f}, L_total={comp08['L_total']:.4f}")
    print(f"  sub-09 (w_α=0.0, w_γ=1.0): L_α={comp09['L_alpha']:.5f} (not used), L_γ={comp09['L_gamma']:.4f}, L_total={comp09['L_total']:.4f}")
    print(f"  {'✓ PASS' if test6['pass'] else '✗ FAIL'}")

    # Test 7: σ sensitivity (loss should be moderately stable under σ ∈ {18, 21, 24, 28})
    print(f"\n[Test 7] σ sensitivity sweep for L_α (sub-08):")
    sigma_sweep = [15.0, 18.0, 21.0, 24.0, 28.0]
    L_a_per_sigma = {}
    for sg in sigma_sweep:
        L_a_per_sigma[sg] = L_behav_alpha(delta_zero, obs_sub08, sigma=sg)
        print(f"  σ={sg:.0f}°: L_α = {L_a_per_sigma[sg]:.5f}")
    test7 = {
        'name': 'σ sensitivity range reasonable',
        'L_per_sigma': L_a_per_sigma,
        'pass': True,  # informational
    }
    results['test7_sigma_sensitivity'] = test7

    # Summary
    all_pass = all(t.get('pass', False) for t in results.values())
    print(f"\n{'=' * 78}")
    print(f"S3 self-test summary: {'✓ ALL PASS' if all_pass else '✗ FAILURES'}")
    print(f"{'=' * 78}")

    return results, all_pass


if __name__ == "__main__":
    results, all_pass = self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "self_test_results.json"
    with open(out_file, 'w') as f:
        json.dump({'all_pass': all_pass, 'tests': results}, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")
    sys.exit(0 if all_pass else 1)
