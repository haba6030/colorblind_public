"""S17: Strict HC LOO (leave-one-HC-out, deterministic 7-fold) on Pipeline 2 final candidates.

User directive (2026-05-26): augment Phase B v6 (s10b_v6_pca_rdm.py) 5/2 random resampling
with a strict HC LOO procedure for the 5 final candidates (Pipeline 2 closure RQ3 limitation).

Design:
- 7-fold LOO over HC_SUBJS (each HC excluded once)
- Per fold: train pool = 6 HCs (HC_i excluded), test pool = {HC_i}
- Train atoms: gamma (baseline from 6 train HCs), RDM (HC mean from 6 train HCs),
  LOCO (CVD-internal, HC-independent — same per fold)
- Composite: z(train_grid) sum / sqrt(n_atoms) → argmin → fit param
- Test atoms: same form, single test HC. For RDM atom, drop `len(pool) < 2` guard
  (single HC RDM is still computable). For γ atom, jnd_baseline_from_pool with n=1
  yields sd=0 → floored to 1e-3 by existing code (z-rescaling uses TRAIN's (mu, sd)).
- Outlier detection: per-candidate, identify HC whose exclusion produces param fit
  most distant from Phase B v6 median (esp. sub-04 outlier per CLAUDE §2.5).

Reuses atom factories from `s10b_v6_pca_rdm.py` (imported via importlib for clean separation).

Output: `results/s10_inclusion/s17_hc_loo_results.json`

Runtime: 5 candidates × 7 folds × (1326 grid_2c | 61 grid_rc) ≈ a few minutes.

Run:
  python s17_hc_loo.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/s17_hc_loo.py  (development repository, commit 53c81c2)
# Role      : Strict 7-fold leave-one-control-out refit of the candidates -> s17_hc_loo_results.json
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

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# Direct imports
from rc_1dof import forward_rc, G_MIN, G_MAX, G_STEP
from two_comp import forward_2comp, BS_GRID, BC_GRID
from neural_loss import (
    load_amplitudes, load_hc_pool, ROI_K,
    precompute_loco_W_within, L_LOCO,
)
from behav_loss import (
    load_jnd_per_pair, L_behav_gamma, PAIR_HUES, HC_JND_SUBJS,
)
from utils_forward_model import create_basis_full, HUE_ANGLES
from s8_loo_train_test import jnd_baseline_from_pool, DELTA_LAMBDA_BY_FAMILY

# Import s10b_v6 module to reuse atom factories without copying
_S10B_PATH = SCRIPT_DIR / "s10b_v6_pca_rdm.py"
_spec = importlib.util.spec_from_file_location("s10b_v6_pca_rdm", _S10B_PATH)
_v6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v6)

make_gamma_pair_atom = _v6.make_gamma_pair_atom
make_rdm_atom = _v6.make_rdm_atom  # original (requires n_pool >= 2)
make_loco_atom = _v6.make_loco_atom
grid_eval_rc = _v6.grid_eval_rc
grid_eval_2comp = _v6.grid_eval_2comp
zscore_grid = _v6.zscore_grid
argmin_rc = _v6.argmin_rc
argmin_2comp = _v6.argmin_2comp
g_grid = _v6.g_grid

OUT_DIR = SCRIPT_DIR.parent / "results" / "s10_inclusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROIS = ['V1', 'V2', 'V3', 'V4']
HUES = np.arange(0, 360, 45, dtype=float)

# ------------------------------------------------------------------------------
# Variant of make_rdm_atom that allows single-HC pool (test atom for LOO).
# Only differs by dropping the `len(pool_amps_dict) < 2` guard.
# ------------------------------------------------------------------------------
def make_rdm_atom_n1ok(roi, cvd_amp, pool_amps_dict, C_baseline, K):
    """RDM atom that accepts a 1-HC pool (used for TEST atom in strict LOO).

    Same as `make_rdm_atom` from s10b_v6 except the `< 2` guard is replaced
    with `< 1`. With n=1, hc_rdm_mean is that single HC's RDM (still well-defined).
    """
    if len(pool_amps_dict) < 1:
        return None
    try:
        from scipy.spatial.distance import squareform
        K_PCA = 6

        def voxel_pca_components(pattern_8xV, k):
            mp = pattern_8xV - pattern_8xV.mean(axis=0, keepdims=True)
            try:
                U, S, Vt = np.linalg.svd(mp, full_matrices=False)
                k_eff = min(k, U.shape[1])
                return U[:, :k_eff] * S[:k_eff]
            except Exception:
                return mp[:, :k]

        def compute_rdm_correlation(scores_8xk):
            n = scores_8xk.shape[0]
            out = np.zeros((n * (n - 1)) // 2)
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = scores_8xk[i], scores_8xk[j]
                    am = a - a.mean(); bm = b - b.mean()
                    denom = (np.linalg.norm(am) * np.linalg.norm(bm))
                    out[idx] = 1.0 - (am @ bm / denom if denom > 1e-9 else 0.0)
                    idx += 1
            return out

        hc_rdms = []
        for sid, amp in pool_amps_dict.items():
            try:
                mean_pat = amp.mean(axis=0)
                scores = voxel_pca_components(mean_pat, K_PCA)
                rdm_mat = squareform(compute_rdm_correlation(scores))
                hc_rdms.append(rdm_mat)
            except Exception:
                continue
        if not hc_rdms:
            return None
        hc_rdm_mean = np.mean(np.stack(hc_rdms, axis=0), axis=0)
        triu = np.triu_indices(8, k=1)
        cvd_mean = cvd_amp.mean(axis=0)
        cvd_scores = voxel_pca_components(cvd_mean, K_PCA)
        cvd_rdm = squareform(compute_rdm_correlation(cvd_scores))
        delta_rdm_obs = (cvd_rdm - hc_rdm_mean)[triu]
        n_obs = float(np.linalg.norm(delta_rdm_obs))
    except Exception:
        return None

    def loss_fn(delta_8vec):
        try:
            perceived = (HUES + delta_8vec) % 360.0
            sim_shifted = np.zeros((8, 8))
            for i in range(8):
                for j in range(8):
                    p_i = int(round(perceived[i] / 45.0)) % 8
                    p_j = int(round(perceived[j] / 45.0)) % 8
                    sim_shifted[i, j] = hc_rdm_mean[p_i, p_j]
            delta_rdm_sim = (sim_shifted - hc_rdm_mean)[triu]
            n_sim = float(np.linalg.norm(delta_rdm_sim))
            if n_sim < 1e-9 or n_obs < 1e-9:
                return 1.0
            cos_sim = float(np.dot(delta_rdm_sim, delta_rdm_obs) / (n_sim * n_obs))
            return 1.0 - cos_sim
        except Exception:
            return np.nan
    return loss_fn


# ------------------------------------------------------------------------------
# Candidate definitions — 5 cells from Pipeline 2 (Phase B v6 final candidates +
# sub-primary alternates).
# ------------------------------------------------------------------------------
CANDIDATES = [
    # ---- sub-08 (deutan, focal=yellow-purple) ----
    {
        'id': 'S08-stable',
        'subject': 'sub-08',
        'family': 'deutan',
        'focal_pair': 'yellow-purple',
        'model': '2comp',
        'combo_label': 'γALL|RDMV1|noLOCO',
        'gamma_pairs': ['ALL'],
        'rdm_rois': ['V1'],
        'loco_v4': False,
        'phase_b_fit': {'beta_s': 38.0, 'beta_c': -10.0},
        'description': 'primary stable (multi-pair γ + V1 RDM)',
    },
    {
        'id': 'S08-robust',
        'subject': 'sub-08',
        'family': 'deutan',
        'focal_pair': 'yellow-purple',
        'model': '2comp',
        'combo_label': 'γOY|RDMV2|noLOCO',
        'gamma_pairs': ['OY'],
        'rdm_rois': ['V2'],
        'loco_v4': False,
        'phase_b_fit': {'beta_s': 6.0, 'beta_c': -42.0},
        'description': 'primary robust (OY γ + V2 RDM)',
    },
    {
        'id': 'S08-rc-subprimary',
        'subject': 'sub-08',
        'family': 'deutan',
        'focal_pair': 'yellow-purple',
        'model': 'rc',
        'dl_src': 'JND_Lamb',
        'dl_value': 6.5,
        'combo_label': 'γ_|RDMV1|noLOCO',
        'gamma_pairs': [],
        'rdm_rois': ['V1'],
        'loco_v4': False,
        'phase_b_fit': {'g': 2.25},
        'description': 'R+C sub-primary (V1 RDM only, Δλ=6.5)',
    },
    # ---- sub-09 (protan, focal=green-blue) ----
    {
        'id': 'S09-stable',
        'subject': 'sub-09',
        'family': 'protan',
        'focal_pair': 'green-blue',
        'model': '2comp',
        'combo_label': 'γALL|RDMV1|noLOCO',
        'gamma_pairs': ['ALL'],
        'rdm_rois': ['V1'],
        'loco_v4': False,
        'phase_b_fit': {'beta_s': 2.0, 'beta_c': 24.0},
        'description': 'primary (multi-pair γ + V1 RDM)',
    },
    {
        'id': 'S09-rc-subprimary',
        'subject': 'sub-09',
        'family': 'protan',
        'focal_pair': 'green-blue',
        'model': 'rc',
        'dl_src': 'Boehm_low',
        'dl_value': 3.0,
        'combo_label': 'γALL|RDMV1|noLOCO',
        'gamma_pairs': ['ALL'],
        'rdm_rois': ['V1'],
        'loco_v4': False,
        'phase_b_fit': {'g': 2.95},
        'description': 'R+C sub-primary (γ_all + V1 RDM, Δλ=3.0)',
    },
]


# ------------------------------------------------------------------------------
# Data preloading: load CVD + all HC amps once
# ------------------------------------------------------------------------------
def preload_data(subjects_needed):
    """Preload CVD amps + HC pool amps + JND for all subjects we'll need."""
    cvd_amps_by_subj = {}
    K_by_roi = {}
    C_by_roi = {}
    for s in subjects_needed:
        cvd_amps_by_subj[s] = {}
        for roi in ROIS:
            try:
                cvd_amps_by_subj[s][roi] = load_amplitudes(s, roi)
                if roi not in K_by_roi:
                    K_by_roi[roi] = ROI_K[roi]
                    C_by_roi[roi] = create_basis_full(K_by_roi[roi], basis_type='fe')[
                        HUE_ANGLES.astype(int)]
            except FileNotFoundError:
                pass

    # HC pool (all 7) per ROI
    hc_amps_by_roi = {}
    for roi in ROIS:
        hc_amps_by_roi[roi] = load_hc_pool(roi)

    cvd_jnd_by_subj = {}
    for s in subjects_needed:
        try:
            cvd_jnd_by_subj[s] = load_jnd_per_pair(s)
        except Exception:
            cvd_jnd_by_subj[s] = None

    return cvd_amps_by_subj, hc_amps_by_roi, K_by_roi, C_by_roi, cvd_jnd_by_subj


# ------------------------------------------------------------------------------
# Per-fold fit for one candidate
# ------------------------------------------------------------------------------
def fit_candidate_fold(cand, held_out_hc, cvd_amps, hc_amps_by_roi,
                        K_by_roi, C_by_roi, cvd_jnd):
    """Single fold: held_out_hc is the test HC, remaining 6 are training pool.

    Returns dict with fit params + train/test loss + boundary flag.
    """
    family = cand['family']
    subject = cand['subject']
    train_hcs = [h for h in HC_SUBJS if h != held_out_hc]
    train_jnd_pool = [h for h in train_hcs if h in HC_JND_SUBJS]
    test_jnd_pool = [held_out_hc] if held_out_hc in HC_JND_SUBJS else []

    # Build TRAIN atoms (6-HC pool)
    train_atoms = {}
    for p in cand['gamma_pairs']:
        fn = make_gamma_pair_atom(p, cvd_jnd, train_jnd_pool) if cvd_jnd else None
        if fn is not None:
            train_atoms[f'gamma_{p}'] = fn
    for roi in cand['rdm_rois']:
        if roi in cvd_amps:
            pool_amps = {h: hc_amps_by_roi[roi][h] for h in train_hcs
                          if h in hc_amps_by_roi[roi]}
            if len(pool_amps) >= 2:
                fn = make_rdm_atom(roi, cvd_amps[roi], pool_amps,
                                    C_by_roi[roi], K_by_roi[roi])
                if fn is not None:
                    train_atoms[f'rdm_{roi}'] = fn
    if cand['loco_v4'] and 'V4' in cvd_amps:
        fn = make_loco_atom(cvd_amps['V4'], K_by_roi['V4'])
        if fn is not None:
            train_atoms['loco_V4'] = fn

    if not train_atoms:
        return None

    # Evaluate train atom grids
    is_rc = (cand['model'] == 'rc')
    if is_rc:
        dl = cand['dl_value']
        train_grids = {name: grid_eval_rc(fn, dl, family)
                        for name, fn in train_atoms.items()}
    else:
        train_grids = {name: grid_eval_2comp(fn, family)
                        for name, fn in train_atoms.items()}

    # Per-atom train stats for z-renormalizing test atom values
    train_stats = {name: (float(np.nanmean(g)), float(np.nanstd(g)))
                   for name, g in train_grids.items()}

    # z-sum composite (train)
    n_a = len(train_atoms)
    z_sum = None
    for name in train_atoms:
        z = zscore_grid(train_grids[name])
        if np.all(np.isnan(z)):
            return None
        z_sum = z if z_sum is None else z_sum + z
    comp_train = z_sum / np.sqrt(n_a)

    # Argmin
    if is_rc:
        fit = argmin_rc(comp_train)
    else:
        fit = argmin_2comp(comp_train)
    if fit is None:
        return None

    # Train loss (composite z-sum value at argmin)
    train_loss_val = float(np.nanmin(comp_train))

    # delta at argmin
    if is_rc:
        delta_at_argmin = forward_rc(cand['dl_value'], fit['g'], family)
    else:
        delta_at_argmin = forward_2comp(fit['beta_s'], fit['beta_c'], family)

    # Build TEST atoms (single test HC pool)
    # NOTE: jnd_baseline_from_pool requires n>=2 pool subjs to compute baseline.
    # With n=1, both bl and sd are None for every pair. make_gamma_pair_atom's
    # single-pair path crashes on None sd; 'ALL' path returns None gracefully.
    # We guard by simulating the baseline check; if n=1, gamma test atom is skipped.
    test_atoms = {}
    can_build_gamma_test = (cvd_jnd is not None) and (len(test_jnd_pool) >= 2)
    for p in cand['gamma_pairs']:
        if not can_build_gamma_test:
            continue
        fn = make_gamma_pair_atom(p, cvd_jnd, test_jnd_pool)
        if fn is not None:
            test_atoms[f'gamma_{p}'] = fn
    for roi in cand['rdm_rois']:
        if roi in cvd_amps and held_out_hc in hc_amps_by_roi[roi]:
            test_pool = {held_out_hc: hc_amps_by_roi[roi][held_out_hc]}
            fn = make_rdm_atom_n1ok(roi, cvd_amps[roi], test_pool,
                                     C_by_roi[roi], K_by_roi[roi])
            if fn is not None:
                test_atoms[f'rdm_{roi}'] = fn
    if cand['loco_v4'] and 'loco_V4' in train_atoms:
        # LOCO is HC-independent → reuse train atom
        test_atoms['loco_V4'] = train_atoms['loco_V4']

    # Compute test loss = sum of (test_atom_value - train_mu) / train_sd, scaled by sqrt(n_a)
    z_test_parts = []
    test_atoms_used = []
    for nm in train_atoms.keys():
        if nm not in test_atoms:
            continue
        mu, sd = train_stats[nm]
        if not np.isfinite(sd) or sd < 1e-10:
            continue
        try:
            t_val = float(test_atoms[nm](delta_at_argmin))
        except Exception:
            continue
        if not np.isfinite(t_val):
            continue
        z_test_parts.append((t_val - mu) / sd)
        test_atoms_used.append(nm)
    test_loss_val = (float(sum(z_test_parts) / np.sqrt(len(z_test_parts)))
                     if z_test_parts else None)

    # test_focal & test_agg
    focal_pair = cand['focal_pair']
    if test_jnd_pool:
        try:
            test_bl, test_sd = jnd_baseline_from_pool(test_jnd_pool)
        except Exception:
            test_bl, test_sd = {}, {}
    else:
        test_bl, test_sd = {}, {}

    def _test_focal(delta):
        if cvd_jnd is None or focal_pair not in test_bl:
            return None
        focal_obs = cvd_jnd.get(focal_pair)
        focal_base = test_bl.get(focal_pair)
        if focal_obs is None or focal_base is None:
            return None
        theta_a, theta_b = PAIR_HUES[focal_pair]
        i = int(round(theta_a / 45.0)) % 8
        j = int(round(theta_b / 45.0)) % 8
        d_phys = min(abs(theta_a - theta_b) % 360,
                      360 - abs(theta_a - theta_b) % 360)
        perceived = (HUES + delta) % 360.0
        d_perc_raw = abs(perceived[i] - perceived[j]) % 360
        d_perc = max(min(d_perc_raw, 360 - d_perc_raw), 1e-3)
        pred = focal_base * (d_phys / d_perc)
        sd_raw = test_sd.get(focal_pair)
        sd_v = max(sd_raw if sd_raw is not None else 1e-3, 1e-3)
        return float(((pred - focal_obs) / sd_v) ** 2)

    def _test_agg(delta):
        if cvd_jnd is None or not test_bl:
            return None
        valid = {p: cvd_jnd[p] for p in test_bl.keys()
                 if cvd_jnd.get(p) is not None and test_bl.get(p) is not None}
        if not valid:
            return None
        sd_d = {p: max(test_sd.get(p) if test_sd.get(p) is not None else 1e-3, 1e-3)
                for p in valid}
        # filter test_bl to only include keys with non-None baselines
        bl_filtered = {p: test_bl[p] for p in valid}
        try:
            return float(L_behav_gamma(delta, valid, bl_filtered, sd_d))
        except Exception:
            return None

    t_focal = _test_focal(delta_at_argmin)
    t_agg = _test_agg(delta_at_argmin)

    # Boundary check (already in fit dict)
    fold_result = {
        'held_out_hc': held_out_hc,
        'train_hcs': train_hcs,
        'train_loss': train_loss_val,
        'test_loss': test_loss_val,
        'test_focal': t_focal,
        'test_agg': t_agg,
        'boundary': bool(fit.get('boundary', False)),
        'train_atoms_used': list(train_atoms.keys()),
        'test_atoms_used': test_atoms_used,
    }
    if is_rc:
        fold_result['g'] = float(fit['g'])
    else:
        fold_result['beta_s'] = float(fit['beta_s'])
        fold_result['beta_c'] = float(fit['beta_c'])
    return fold_result


# ------------------------------------------------------------------------------
# Aggregate per-candidate LOO results
# ------------------------------------------------------------------------------
def summarize_candidate(cand, folds):
    """Compute median, IQR, range across the 7 folds + identify outlier holdouts."""
    valid = [f for f in folds if f is not None]
    if not valid:
        return {'n_folds_valid': 0}

    pb_fit = cand.get('phase_b_fit', {})
    is_rc = (cand['model'] == 'rc')

    summary = {
        'n_folds_valid': len(valid),
        'phase_b_v6_fit': pb_fit,
    }

    if is_rc:
        gs = np.array([f['g'] for f in valid if f.get('g') is not None])
        summary['g_median'] = float(np.median(gs))
        summary['g_iqr'] = float(np.percentile(gs, 75) - np.percentile(gs, 25))
        summary['g_min'] = float(np.min(gs))
        summary['g_max'] = float(np.max(gs))
        summary['g_range'] = float(summary['g_max'] - summary['g_min'])
        # outlier vs Phase B v6 median
        pb_g = pb_fit.get('g')
        if pb_g is not None:
            diffs = [(f['held_out_hc'], abs(f['g'] - pb_g)) for f in valid]
            diffs.sort(key=lambda x: -x[1])
            summary['outlier_ranking'] = [{'held_out_hc': h, 'abs_delta_g': float(d)}
                                          for h, d in diffs]
            summary['outlier_holdouts'] = [h for h, d in diffs if d >= 0.5]
    else:
        bss = np.array([f['beta_s'] for f in valid if f.get('beta_s') is not None])
        bcs = np.array([f['beta_c'] for f in valid if f.get('beta_c') is not None])
        summary['bs_median'] = float(np.median(bss))
        summary['bs_iqr'] = float(np.percentile(bss, 75) - np.percentile(bss, 25))
        summary['bs_min'] = float(np.min(bss))
        summary['bs_max'] = float(np.max(bss))
        summary['bs_range'] = float(summary['bs_max'] - summary['bs_min'])
        summary['bc_median'] = float(np.median(bcs))
        summary['bc_iqr'] = float(np.percentile(bcs, 75) - np.percentile(bcs, 25))
        summary['bc_min'] = float(np.min(bcs))
        summary['bc_max'] = float(np.max(bcs))
        summary['bc_range'] = float(summary['bc_max'] - summary['bc_min'])
        # outlier vs Phase B v6 (β_s, β_c)
        pb_bs = pb_fit.get('beta_s'); pb_bc = pb_fit.get('beta_c')
        if pb_bs is not None and pb_bc is not None:
            diffs = []
            for f in valid:
                d = float(np.hypot(f['beta_s'] - pb_bs, f['beta_c'] - pb_bc))
                diffs.append((f['held_out_hc'], d))
            diffs.sort(key=lambda x: -x[1])
            summary['outlier_ranking'] = [
                {'held_out_hc': h, 'euclid_delta_bsc': float(d)}
                for h, d in diffs
            ]
            # Flag holdouts producing >20° euclidean shift in (β_s,β_c) plane
            summary['outlier_holdouts'] = [h for h, d in diffs if d >= 20.0]

    summary['boundary_rate'] = float(np.mean([f['boundary'] for f in valid]))
    # train/test loss summary
    tr_losses = [f['train_loss'] for f in valid if f.get('train_loss') is not None]
    te_losses = [f['test_loss'] for f in valid if f.get('test_loss') is not None]
    summary['train_loss_median'] = float(np.median(tr_losses)) if tr_losses else None
    summary['train_loss_iqr'] = (float(np.percentile(tr_losses, 75) - np.percentile(tr_losses, 25))
                                  if len(tr_losses) >= 2 else None)
    summary['test_loss_median'] = float(np.median(te_losses)) if te_losses else None
    summary['test_loss_iqr'] = (float(np.percentile(te_losses, 75) - np.percentile(te_losses, 25))
                                 if len(te_losses) >= 2 else None)
    # test_focal / test_agg
    tf = [f['test_focal'] for f in valid if f.get('test_focal') is not None]
    ta = [f['test_agg'] for f in valid if f.get('test_agg') is not None]
    summary['test_focal_median'] = float(np.median(tf)) if tf else None
    summary['test_agg_median'] = float(np.median(ta)) if ta else None

    return summary


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    print("=" * 100, flush=True)
    print("S17 — Strict HC LOO (7-fold) on Pipeline 2 final candidates", flush=True)
    print(f"  {len(CANDIDATES)} candidates × {len(HC_SUBJS)} folds", flush=True)
    print("=" * 100, flush=True)

    t0 = time.time()

    subjects_needed = sorted({c['subject'] for c in CANDIDATES})
    cvd_amps_by_subj, hc_amps_by_roi, K_by_roi, C_by_roi, cvd_jnd_by_subj = \
        preload_data(subjects_needed)

    results = {'candidates': [], 'meta': {
        'design': 'strict HC LOO 7-fold',
        'n_train_hcs_per_fold': 6,
        'n_test_hcs_per_fold': 1,
        'hc_subjects': HC_SUBJS,
        'rdm_test_handling': 'single-HC RDM atom (n=1 allowed in test only)',
        'gamma_test_handling': 'jnd_baseline_from_pool with n=1 → sd floored to 1e-3, z-renormalized to TRAIN (mu,sd)',
        'loco_atom_invariance': True,
        'loco_atom_note': 'LOCO atom is CVD-internal (HC-independent); same value across folds',
        'phase_b_v6_comparison': 'param IQR not directly comparable to 5/2 × 300 resamples; range [min,max] is more honest stability indicator for n=7',
    }}

    for cand in CANDIDATES:
        t_cand = time.time()
        cvd_amps = cvd_amps_by_subj[cand['subject']]
        cvd_jnd = cvd_jnd_by_subj[cand['subject']]
        print(f"\n[{cand['id']}] {cand['subject']} {cand['model']} {cand['combo_label']}",
              flush=True)
        folds = []
        for held_out_hc in HC_SUBJS:
            fold = fit_candidate_fold(
                cand, held_out_hc, cvd_amps, hc_amps_by_roi,
                K_by_roi, C_by_roi, cvd_jnd)
            if fold is None:
                print(f"  fold {held_out_hc}: SKIPPED (no train atoms or fit failed)",
                      flush=True)
            else:
                tl = fold['test_loss']
                tl_str = 'None' if tl is None else f"{tl:.4f}"
                if cand['model'] == 'rc':
                    print(f"  fold {held_out_hc}: g={fold['g']:.3f} "
                          f"train_loss={fold['train_loss']:.4f} "
                          f"test_loss={tl_str} "
                          f"boundary={fold['boundary']}", flush=True)
                else:
                    print(f"  fold {held_out_hc}: bs={fold['beta_s']:.1f} bc={fold['beta_c']:.1f} "
                          f"train_loss={fold['train_loss']:.4f} "
                          f"test_loss={tl_str} "
                          f"boundary={fold['boundary']}", flush=True)
            folds.append(fold)

        summary = summarize_candidate(cand, folds)
        cand_record = {
            'id': cand['id'],
            'subject': cand['subject'],
            'family': cand['family'],
            'model': cand['model'],
            'combo_label': cand['combo_label'],
            'gamma_pairs': cand['gamma_pairs'],
            'rdm_rois': cand['rdm_rois'],
            'loco_v4': cand['loco_v4'],
            'description': cand['description'],
            'phase_b_v6_fit': cand.get('phase_b_fit', {}),
            'loo_folds': folds,
            'summary': summary,
        }
        if cand['model'] == 'rc':
            cand_record['dl_src'] = cand['dl_src']
            cand_record['dl_value'] = cand['dl_value']
        results['candidates'].append(cand_record)
        print(f"  -- summary: {summary}", flush=True)
        print(f"  elapsed: {time.time() - t_cand:.1f}s", flush=True)

    elapsed = round(time.time() - t0, 1)
    results['meta']['elapsed_sec'] = elapsed
    results['meta']['n_candidates'] = len(CANDIDATES)

    out_file = OUT_DIR / "s17_hc_loo_results.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[done] elapsed={elapsed}s  saved={out_file}", flush=True)


if __name__ == "__main__":
    main()
