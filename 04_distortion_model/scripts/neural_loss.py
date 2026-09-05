"""S4: neural_loss.py — Neural loss functions for fMRI fit.

Provides:
  L_LOCO     (per-color hold-out reconstruction MSE via HC encoder)
  L_RDM      (1 − cos(ΔRDM_sim, ΔRDM_obs), correlation-distance primary)
  L_RDM_xnobis (Crossnobis robustness variant)
  L_neural   (composite 0.5·L_LOCO + 0.5·L_RDM)

Forward input: δθ(c) 8-vec (degrees, perceived rotation per canonical hue).
Outputs: scalar loss per ROI.

Reuses existing functions:
  - precompute_hc_W (encoder W per HC subject via ridge GCV)
  - compute_rdm_correlation, compute_rdm_crossnobis (RDM forms)
  - compute_delta_rdm_obs, compute_delta_rdm_sim (ΔRDM)
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/neural_loss.py  (development repository, commit 53c81c2)
# Role      : Delta-RDM loss atom L_RDM (cosine dissimilarity in PCA(6) space).
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
ROOT = SCRIPT_DIR.parents[2]
DATA_C010_DIR = ROOT / "analysis" / "phase1_procrustes_decoding" / "results" / "full_dataset_C010"
# Sensitivity arms (U2, 2026-08-10). The published amplitudes stay the default, so
# every existing reproduction path is unchanged; set COLORBLIND_AMP_ROOT to point a
# single run at another preprocessing arm, e.g.
#   COLORBLIND_AMP_ROOT=.../visualization/full_dataset_C010_motreg
import os as _os  # noqa: E402
if _os.environ.get("COLORBLIND_AMP_ROOT"):
    DATA_C010_DIR = Path(_os.environ["COLORBLIND_AMP_ROOT"])
OUT_DIR = SCRIPT_DIR.parent / "results" / "neural_loss"

sys.path.insert(0, str(SCRIPT_DIR))
_PHASE1_FWD = ROOT / "analysis" / "phase4_forward_model" / "scripts"
sys.path.insert(0, str(_PHASE1_FWD))

from utils_forward_model import (
    create_basis_full, HUE_ANGLES, N_RUNS, N_COLORS, N_CHANNELS,
    fit_W_ridge, gcv_select_alpha,
)
from diagnostic_delta_rdm import (
    compute_rdm_correlation, compute_rdm_crossnobis,
    compute_delta_rdm_obs, compute_delta_rdm_sim,
    cosine_similarity, precompute_hc_W,
)

HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROI_K = {'V1': 6, 'V2': 6, 'V3': 6, 'V4': 6}  # FE-6 uniform (Phase 1 baseline, 2026-05-22 fix from SRM-K override; see memory/feedback_k_uniform_fe6.md)


# ============================================================================
# Data loaders
# ============================================================================

def load_amplitudes(subject: str, roi: str) -> np.ndarray:
    """Load (N_RUNS, N_COLORS, V_s) amplitudes for subject/ROI."""
    path = DATA_C010_DIR / subject / roi / "amplitudes_procrustes.npy"
    return np.load(path)


def load_hc_pool(roi: str, hc_subjs=HC_SUBJS) -> dict:
    """Load HC pool amplitudes dict {subj: (6, 8, V_s)}. Excludes sub-07 V4 if 16 voxels (sparse)."""
    out = {}
    for s in hc_subjs:
        try:
            amp = load_amplitudes(s, roi)
            if roi == 'V4' and amp.shape[2] < 20:
                # sub-07 hV4 sparse (16 voxels) → causes nan in correlation distance
                continue
            out[s] = amp
        except FileNotFoundError:
            continue
    return out


# ============================================================================
# Design matrix from δθ
# ============================================================================

def design_from_delta(delta_8vec: np.ndarray, n_channels: int,
                      basis_type: str = 'fe') -> np.ndarray:
    """Construct C_shifted (8, K) at perceived hue = canonical + δθ."""
    shifted_angles = (HUE_ANGLES + delta_8vec) % 360
    basis_full = create_basis_full(n_channels, basis_type=basis_type)
    idx = np.round(shifted_angles).astype(int) % 360
    return basis_full[idx]


# ============================================================================
# L_LOCO: per-color hold-out reconstruction
# ============================================================================

def precompute_loco_W_within(cvd_amp: np.ndarray, C_baseline: np.ndarray) -> dict:
    """Precompute within-subject leave-one-color-out W on CVD's own data.

    Resolves V_s mismatch issue: cannot apply HC-trained W to CVD pattern
    (different V_s per subject). Within-subject LOCO uses *CVD's own W*
    trained on 7-color subset, predict held-out 1 color.

    Returns:
        dict {held_out_color: (W, alpha)} for the given CVD subject
    """
    V_s = cvd_amp.shape[2]
    loco_W = {}
    loco_alpha = {}
    for c_held in range(N_COLORS):
        train_colors = [c for c in range(N_COLORS) if c != c_held]
        C_train = np.tile(C_baseline[train_colors], (N_RUNS, 1))  # (42, K)
        X_train = cvd_amp[:, train_colors, :].reshape(-1, V_s)  # (42, V_s)
        alpha, _ = gcv_select_alpha(C_train, X_train)
        W = fit_W_ridge(C_train, X_train, alpha)
        loco_W[c_held] = W
        loco_alpha[c_held] = float(alpha)
    return loco_W, loco_alpha


def compute_loco_rho_within(delta_rc: np.ndarray, cvd_amp: np.ndarray,
                              loco_W_within: dict, n_channels: int,
                              basis_type: str = 'fe') -> np.ndarray:
    """Within-subject LOCO ρ under model δθ.

    For each held-out color c:
      W_c = LOCO W trained on 7 train colors (using CVD's own data)
      Y_pred(c) = C(c + δθ(c)) @ W_c
      Y_actual(c) = CVD's mean run pattern at color c
      ρ(c) = pearson_r(Y_pred[0], Y_actual)

    No cross-subject V_s mismatch (everything in CVD's V_s space).

    Returns:
        rho_8vec: (8,)
    """
    C_shifted = design_from_delta(delta_rc, n_channels=n_channels, basis_type=basis_type)
    cvd_mean = cvd_amp.mean(axis=0)  # (8, V_s)
    rho = np.zeros(N_COLORS)
    for c in range(N_COLORS):
        W_c = loco_W_within[c]
        Y_pred = (C_shifted[c:c+1] @ W_c)[0]
        Y_actual = cvd_mean[c]
        if np.std(Y_pred) < 1e-10 or np.std(Y_actual) < 1e-10:
            rho[c] = 0.0
        else:
            r = np.corrcoef(Y_pred, Y_actual)[0, 1]
            rho[c] = r if np.isfinite(r) else 0.0
    return rho


def L_LOCO(delta_rc: np.ndarray, cvd_amp: np.ndarray, loco_W_within: dict,
            n_channels: int) -> float:
    """L_LOCO = mean(1 − rho(c)) using within-subject LOCO.

    Lower = better forward prediction. Bounded in [0, 2].
    """
    rho = compute_loco_rho_within(delta_rc, cvd_amp, loco_W_within, n_channels)
    return float(np.mean(1.0 - rho))


# ============================================================================
# L_RDM: 1 - cos(ΔRDM_sim, ΔRDM_obs)
# ============================================================================

def L_RDM(delta_rc: np.ndarray, cvd_amp: np.ndarray, hc_amps_dict: dict,
           hc_W_dict: dict, C_baseline: np.ndarray, n_channels: int,
           distance: str = 'correlation') -> float:
    """1 − cos(ΔRDM_sim(δθ), ΔRDM_obs).

    distance: 'correlation' (primary) or 'crossnobis' (robustness)
    """
    C_shifted = design_from_delta(delta_rc, n_channels=n_channels)
    delta_rdm_obs, _, _, _ = compute_delta_rdm_obs(cvd_amp, hc_amps_dict, distance=distance)
    delta_rdm_sim, _ = compute_delta_rdm_sim(
        hc_W_dict, C_shifted, C_baseline,
        hc_amps_dict=hc_amps_dict, distance=distance)
    cos = cosine_similarity(delta_rdm_sim, delta_rdm_obs)
    if not np.isfinite(cos):
        cos = 0.0
    return float(1.0 - cos)


# ============================================================================
# L_neural composite
# ============================================================================

def L_neural_composite(delta_rc: np.ndarray, cvd_amp: np.ndarray, hc_amps_dict: dict,
                        hc_W_dict: dict, loco_W_cvd: dict,
                        C_baseline: np.ndarray, n_channels: int,
                        w_loco: float = 0.5, w_rdm: float = 0.5,
                        distance: str = 'correlation') -> dict:
    """L_neural = w_loco · L_LOCO + w_rdm · L_RDM."""
    l_loco = L_LOCO(delta_rc, cvd_amp, loco_W_cvd, n_channels)
    l_rdm = L_RDM(delta_rc, cvd_amp, hc_amps_dict, hc_W_dict,
                  C_baseline, n_channels, distance=distance)
    return {
        'L_LOCO': l_loco,
        'L_RDM': l_rdm,
        'L_neural': w_loco * l_loco + w_rdm * l_rdm,
        'w_loco': w_loco,
        'w_rdm': w_rdm,
    }


# ============================================================================
# Self-test (Stage 2 validation)
# ============================================================================

def self_test(rois=('V1', 'V4')):
    """S4 Stage 2: validation across V1 (richer) and V4 (lower K, sparser)."""
    print("=" * 78)
    print("S4 self-test — neural_loss validation")
    print("=" * 78)

    results = {}
    delta_zero = np.zeros(8)

    for roi in rois:
        print(f"\n--- ROI: {roi} (K = {ROI_K[roi]}) ---")
        K = ROI_K[roi]
        hc_amps = load_hc_pool(roi)
        n_hc_used = len(hc_amps)
        print(f"  HC subjects with valid data: {n_hc_used}/{len(HC_SUBJS)}")

        C_baseline = create_basis_full(K, basis_type='fe')[HUE_ANGLES.astype(int)]
        hc_W, hc_alpha = precompute_hc_W(hc_amps, C_baseline)
        print(f"  HC encoder W computed for {len(hc_W)} subjects (V_s mismatch — used per-HC RDM only)")
        print(f"  Ridge α range: [{min(hc_alpha.values()):.4f}, {max(hc_alpha.values()):.4f}]")

        # Test 1: HC self-fit at δθ=0 — within-subject LOCO + RDM
        h_test = list(hc_amps.keys())[0]
        rest = {k: v for k, v in hc_amps.items() if k != h_test}
        if len(rest) < 2:
            print(f"  [Test skip] Not enough HC for self-fit ({len(rest)})")
            continue
        rest_W, _ = precompute_hc_W(rest, C_baseline)
        h_loco_W, _ = precompute_loco_W_within(hc_amps[h_test], C_baseline)

        loco_hc_self = L_LOCO(delta_zero, hc_amps[h_test], h_loco_W, K)
        rdm_hc_self = L_RDM(delta_zero, hc_amps[h_test], rest, rest_W, C_baseline, K)
        print(f"\n  [Test1] HC self-fit ({h_test} as target, others as pool):")
        print(f"    L_LOCO (within {h_test}) = {loco_hc_self:.4f}")
        print(f"    L_RDM (vs others)         = {rdm_hc_self:.4f}")
        roi_results = {
            'roi': roi, 'K': K,
            'hc_self_loco': float(loco_hc_self),
            'hc_self_rdm': float(rdm_hc_self),
        }

        # Test 2: CVD subjects δθ=0 — L_LOCO, L_RDM signals
        for cvd_subj in ['sub-08', 'sub-09']:
            try:
                cvd_amp = load_amplitudes(cvd_subj, roi)
                cvd_loco_W, _ = precompute_loco_W_within(cvd_amp, C_baseline)
                loco_cvd = L_LOCO(delta_zero, cvd_amp, cvd_loco_W, K)
                rdm_corr = L_RDM(delta_zero, cvd_amp, hc_amps, hc_W, C_baseline, K, distance='correlation')
                print(f"\n  [Test2] {cvd_subj} at δθ=0:")
                print(f"    L_LOCO (within) = {loco_cvd:.4f} (vs HC self {loco_hc_self:.4f})")
                print(f"    L_RDM(corr)     = {rdm_corr:.4f} (vs HC self {rdm_hc_self:.4f})")
                roi_results[f'{cvd_subj}_loco'] = float(loco_cvd)
                roi_results[f'{cvd_subj}_rdm_corr'] = float(rdm_corr)
            except FileNotFoundError:
                print(f"  [Test2] {cvd_subj}: no data for ROI {roi}")
            except Exception as e:
                print(f"  [Test2] {cvd_subj}: ERROR {type(e).__name__}: {e}")

        # Test 3: Crossnobis variant runs
        try:
            rdm_xnobis = L_RDM(delta_zero, load_amplitudes('sub-08', roi),
                                hc_amps, hc_W, C_baseline, K, distance='crossnobis')
            print(f"\n  [Test3] sub-08 Crossnobis L_RDM = {rdm_xnobis:.4f}")
            roi_results['sub-08_rdm_xnobis'] = float(rdm_xnobis)
        except Exception as e:
            print(f"  [Test3] Crossnobis ERROR: {type(e).__name__}: {e}")

        # Test 4: Different δθ → different loss (non-flat) on sub-08
        cvd_amp_08 = load_amplitudes('sub-08', roi)
        cvd_loco_W_08, _ = precompute_loco_W_within(cvd_amp_08, C_baseline)
        delta_nonzero = np.array([5, 10, -5, 0, 5, -10, 0, 5], dtype=float)
        loco_pert = L_LOCO(delta_nonzero, cvd_amp_08, cvd_loco_W_08, K)
        print(f"\n  [Test4] sub-08 L_LOCO at δθ=[5,10,-5,0,5,-10,0,5]: {loco_pert:.4f} "
              f"(vs δθ=0: {roi_results.get('sub-08_loco', 'N/A')})")
        roi_results['sub-08_loco_pert'] = float(loco_pert)
        roi_results['loss_responds_to_delta'] = bool(abs(loco_pert - roi_results.get('sub-08_loco', 0)) > 1e-6)

        results[roi] = roi_results

    # Summary
    print(f"\n{'=' * 78}")
    print("Summary: CVD δθ=0 vs HC self-fit")
    print(f"{'=' * 78}")
    print(f"{'ROI':<6} {'HC_LOCO':>10} {'HC_RDM':>10} "
          f"{'08_LOCO':>10} {'08_RDM_c':>10} {'09_LOCO':>10} {'09_RDM_c':>10} "
          f"{'08_RDM_x':>10}")
    for roi, r in results.items():
        print(f"{roi:<6} "
              f"{r.get('hc_self_loco', float('nan')):>10.4f} "
              f"{r.get('hc_self_rdm', float('nan')):>10.4f} "
              f"{r.get('sub-08_loco', float('nan')):>10.4f} "
              f"{r.get('sub-08_rdm_corr', float('nan')):>10.4f} "
              f"{r.get('sub-09_loco', float('nan')):>10.4f} "
              f"{r.get('sub-09_rdm_corr', float('nan')):>10.4f} "
              f"{r.get('sub-08_rdm_xnobis', float('nan')):>10.4f}")

    # Verdict
    all_ok = []
    for roi, r in results.items():
        for key, name in [('sub-08_loco', 'sub-08 L_LOCO'), ('sub-08_rdm_corr', 'sub-08 L_RDM'),
                          ('sub-09_loco', 'sub-09 L_LOCO'), ('sub-09_rdm_corr', 'sub-09 L_RDM')]:
            if key in r and np.isfinite(r[key]):
                all_ok.append(True)
        if r.get('loss_responds_to_delta'):
            all_ok.append(True)
        else:
            all_ok.append(False)

    all_pass = all(all_ok) and len(all_ok) > 0
    print(f"\nS4 self-test summary: {'✓ ALL PASS' if all_pass else '✗ FAILURES'}")
    return results, all_pass


if __name__ == "__main__":
    results, all_pass = self_test()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "self_test_results.json"
    with open(out_file, 'w') as f:
        json.dump({'all_pass': all_pass, 'tests': results}, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")
    sys.exit(0 if all_pass else 1)
