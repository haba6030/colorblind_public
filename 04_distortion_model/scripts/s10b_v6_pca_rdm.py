"""S10b v3: Inclusion screening with extended test metrics + storage subset ids.

Changes from v2:
- N_RESAMPLES = 300 (was 1000) — 후보 식별 충분
- Additional test metrics on complement HC baseline:
    * test_focal (per-subject focal pair) — same as v2
    * test_agg (8-pair aggregate L_γ) — same as v2
    * test_V1_RDM (V1 RDM cosine distance) — NEW
    * test_per_pair (8 pair z² individually) — NEW
- Storage: subset_idx + train/test HC ids per fit (post-hoc analysis enabled)
- Combo chunk split via --combo-start / --combo-end (SLURM array support)

Run modes:
  python s10b_v3_extended.py --subject sub-08
  python s10b_v3_extended.py --subject sub-08 --combo-start 0 --combo-end 10
  python s10b_v3_extended.py --subject sub-09
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/s10b_v6_pca_rdm.py  (development repository, commit 53c81c2)
# Role      : Production fit: all loss combinations x {2-component, R+C at three anchors} over N = 300 resamples -> s10b_v6_pca_rdm_results_sub-*.json
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
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rc_1dof import forward_rc, G_MIN, G_MAX, G_STEP
from two_comp import forward_2comp, BS_GRID, BC_GRID
from neural_loss import (
    load_amplitudes, load_hc_pool, ROI_K,
    precompute_loco_W_within, L_LOCO, L_RDM,
)
from diagnostic_delta_rdm import precompute_hc_W
from behav_loss import (
    load_jnd_per_pair, L_behav_gamma, PAIR_HUES, HC_JND_SUBJS,
)
from utils_forward_model import create_basis_full, HUE_ANGLES
from s8_loo_train_test import jnd_baseline_from_pool, DELTA_LAMBDA_BY_FAMILY

OUT_DIR = SCRIPT_DIR.parent / "results" / "s10_inclusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROIS = ['V1', 'V2', 'V3', 'V4']

N_RESAMPLES = 300
SUBSET_SIZE = 5
RNG_SEED = 42

PAIR_KEYS = {'OY': 'orange-yellow', 'YG': 'yellow-green',
              'YP': 'yellow-purple', 'GB': 'green-blue'}

SUBJECTS = {
    'sub-08': {'family': 'deutan', 'pairs': ['OY', 'YG', 'YP', 'ALL'],
                'rdm_rois': ['V1', 'V2', 'V3', 'V4'],
                'focal_pair': 'yellow-purple'},
    'sub-09': {'family': 'protan', 'pairs': ['GB', 'ALL'],
                'rdm_rois': ['V1'],
                'focal_pair': 'green-blue'},
}

K_RC = 1
K_2C = 2
HUES = np.arange(0, 360, 45, dtype=float)


def g_grid():
    return np.arange(G_MIN, G_MAX + 1e-9, G_STEP)


def make_gamma_pair_atom(pair_key, cvd_jnd, pool_jnd_subjs):
    """Single-pair γ atom (existing) OR 'ALL' = sum over all 8 pairs (Cycle 3 / v5)."""
    if pair_key == 'ALL':
        bl, sd = jnd_baseline_from_pool(pool_jnd_subjs)
        # Collect all valid pairs (cvd_jnd present + bl present)
        valid = []
        for pn, (ta, tb) in PAIR_HUES.items():
            if (pn in bl and cvd_jnd.get(pn) is not None
                    and bl.get(pn) is not None and sd.get(pn) is not None):
                p_sd = max(sd[pn], 1e-3)
                i = int(round(ta/45.0)) % 8; j = int(round(tb/45.0)) % 8
                d_phys = min(abs(ta-tb)%360, 360-abs(ta-tb)%360)
                valid.append((i, j, d_phys, bl[pn], cvd_jnd[pn], p_sd))
        if not valid:
            return None
        def loss_fn(delta_8vec):
            perceived = (HUES + delta_8vec) % 360.0
            total = 0.0
            for i, j, d_phys, p_base, p_obs, p_sd in valid:
                d_perc_raw = abs(perceived[i] - perceived[j]) % 360
                d_perc = max(min(d_perc_raw, 360 - d_perc_raw), 1e-3)
                pred = p_base * (d_phys / d_perc)
                total += ((pred - p_obs) / p_sd) ** 2
            return total
        return loss_fn
    # Single-pair (original behavior)
    pair_name = PAIR_KEYS[pair_key]
    bl, sd = jnd_baseline_from_pool(pool_jnd_subjs)
    if (pair_name not in bl or cvd_jnd.get(pair_name) is None
            or bl.get(pair_name) is None or sd.get(pair_name) is None):
        return None
    p_obs = cvd_jnd[pair_name]
    p_base = bl[pair_name]
    p_sd = max(sd[pair_name], 1e-3)
    theta_a, theta_b = PAIR_HUES[pair_name]
    i = int(round(theta_a / 45.0)) % 8
    j = int(round(theta_b / 45.0)) % 8

    def loss_fn(delta_8vec):
        perceived = (HUES + delta_8vec) % 360.0
        d_phys = min(abs(theta_a - theta_b) % 360, 360 - abs(theta_a - theta_b) % 360)
        d_perc_raw = abs(perceived[i] - perceived[j]) % 360
        d_perc = max(min(d_perc_raw, 360 - d_perc_raw), 1e-3)
        pred = p_base * (d_phys / d_perc)
        return ((pred - p_obs) / p_sd) ** 2
    return loss_fn


def make_rdm_atom(roi, cvd_amp, pool_amps_dict, C_baseline, K):
    """v6: replaced voxel-RDM with A2 PCA-aligned RDM (Cycle 5 finding: 2× separation).

    PCA top-K_pca components per (HC, color) → 8×K_pca scores → 8×8 RDM in PC space.
    HC mean RDM (PC-space) vs CVD RDM (PC-space). Forward δθ effect: rotate CVD's
    8-color RDM via perceived-color ordering.

    Identical signature to v5 voxel-RDM for drop-in compatibility.
    """
    if len(pool_amps_dict) < 2:
        return None
    try:
        from scipy.spatial.distance import squareform
        K_PCA = 6  # matches s14 PCA_SHARED_K
        # Voxel PCA helper (inlined for self-contained module)
        def voxel_pca_components(pattern_8xV, k):
            mp = pattern_8xV - pattern_8xV.mean(axis=0, keepdims=True)
            try:
                U, S, Vt = np.linalg.svd(mp, full_matrices=False)
                k_eff = min(k, U.shape[1])
                return U[:, :k_eff] * S[:k_eff]
            except Exception:
                return mp[:, :k]
        # Correlation distance for 8x8 patterns
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
        # HC mean RDM (PC-space)
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
        # CVD RDM (PC-space)
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


def make_loco_atom(cvd_amp_v4, K_v4):
    try:
        C_b = create_basis_full(K_v4, basis_type='fe')[HUE_ANGLES.astype(int)]
        loco_W, _ = precompute_loco_W_within(cvd_amp_v4, C_b)
    except Exception:
        return None

    def loss_fn(delta_8vec):
        try:
            return L_LOCO(delta_8vec, cvd_amp_v4, loco_W, K_v4)
        except Exception:
            return np.nan
    return loss_fn


def grid_eval_rc(loss_fn, dl, family):
    grid = g_grid()
    out = np.zeros(len(grid))
    for i, g in enumerate(grid):
        delta = forward_rc(dl, g, family)
        try:
            v = float(loss_fn(delta))
            out[i] = v if np.isfinite(v) else np.nan
        except Exception:
            out[i] = np.nan
    return out


def grid_eval_2comp(loss_fn, family):
    out = np.zeros((len(BS_GRID), len(BC_GRID)))
    for i, bs in enumerate(BS_GRID):
        for j, bc in enumerate(BC_GRID):
            delta = forward_2comp(bs, bc, family)
            try:
                v = float(loss_fn(delta))
                out[i, j] = v if np.isfinite(v) else np.nan
            except Exception:
                out[i, j] = np.nan
    return out


def zscore_grid(arr):
    arr = np.asarray(arr, dtype=float)
    mu = np.nanmean(arr); s = np.nanstd(arr)
    if not np.isfinite(s) or s < 1e-10:
        return np.full_like(arr, np.nan)
    return (arr - mu) / s


def argmin_rc(arr):
    if np.all(np.isnan(arr)):
        return None
    grid = g_grid()
    idx = int(np.nanargmin(arr))
    return {'g': float(grid[idx]),
            'boundary': bool(idx == 0 or idx == len(grid) - 1)}


def argmin_2comp(arr):
    if np.all(np.isnan(arr)):
        return None
    flat = int(np.nanargmin(arr.ravel()))
    i, j = np.unravel_index(flat, arr.shape)
    return {'beta_s': float(BS_GRID[i]), 'beta_c': float(BC_GRID[j]),
            'boundary': bool(i == 0 or i == len(BS_GRID) - 1 or
                              j == 0 or j == len(BC_GRID) - 1)}


def enumerate_combos_sub08():
    # v5 (2026-05-26): added 'ALL' γ option = sum over 8 pairs (multi-pair γ_all atom).
    # Phase B v4 enumerations retained, plus new γ_all combos.
    gamma_opts = [[], ['OY'], ['YG'], ['YP'], ['OY', 'YG', 'YP'], ['ALL']]
    rdm_opts = [[], ['V1'], ['V2'], ['V3'], ['V4'], ['V1', 'V4']]
    loco_opts = [[], ['V4']]
    out = []
    for g_a, r_r, l in itertools.product(gamma_opts, rdm_opts, loco_opts):
        if not g_a and not r_r and not l:
            continue  # skip all-empty
        g_label = ','.join(g_a) if g_a else '_'
        r_label = '+'.join(r_r) if r_r else '_'
        l_label = 'LOCO' if l else 'noLOCO'
        out.append({'gamma_pairs': g_a, 'rdm_rois': r_r, 'loco_v4': bool(l),
                     'label': f"γ{g_label}|RDM{r_label}|{l_label}"})
    return out


def enumerate_combos_sub09():
    # v5: add γ_all combos for sub-09 too.
    out = []
    # First, existing v4 enumeration (binary inc/exc per atom group)
    for inc_g, inc_r, inc_l in itertools.product([False, True], repeat=3):
        if not (inc_g or inc_r or inc_l):
            continue
        out.append({'gamma_pairs': ['GB'] if inc_g else [],
                     'rdm_rois': ['V1'] if inc_r else [],
                     'loco_v4': inc_l,
                     'label': f"γ{'GB' if inc_g else '_'}|"
                              f"RDM{'V1' if inc_r else '_'}|"
                              f"{'LOCO' if inc_l else 'noLOCO'}"})
    # v5 add: γ_all combos (4 = ALL/none × R/noR × L/noL minus all-empty)
    for inc_r, inc_l in itertools.product([False, True], repeat=2):
        out.append({'gamma_pairs': ['ALL'],
                     'rdm_rois': ['V1'] if inc_r else [],
                     'loco_v4': inc_l,
                     'label': f"γALL|"
                              f"RDM{'V1' if inc_r else '_'}|"
                              f"{'LOCO' if inc_l else 'noLOCO'}"})
    return out


def enumerate_combos_sub09_full():
    """ROI audit (2026-08-05): sub-09 는 `SUBJECTS['sub-09']['rdm_rois'] = ['V1']` 로
    V1 만 탐색되어 V2/V3/V4 RDM atom 이 생성조차 되지 않았다 (sub-08 은 4-ROI 전탐색).

    **선택 규칙은 그대로다** (test_loss_median ASC → iqr ASC → boundary_rate < 0.5,
    CLAUDE.md §2.5). 바뀌는 것은 탐색 격자의 ROI 축뿐이며, γ 축은 sub-09 원본과 동일하게
    [none / GB(focal) / ALL] 로 둔다. 즉 §0 의 'selection rule reformulation 금지' 에
    해당하지 않는, **truncate 된 격자의 완성**이다.

    3 γ × 6 RDM × 2 LOCO − 1(all-empty) = 35 combos.
    """
    gamma_opts = [[], ['GB'], ['ALL']]
    rdm_opts = [[], ['V1'], ['V2'], ['V3'], ['V4'], ['V1', 'V4']]
    loco_opts = [[], ['V4']]
    out = []
    for g_a, r_r, l in itertools.product(gamma_opts, rdm_opts, loco_opts):
        if not g_a and not r_r and not l:
            continue
        g_label = ','.join(g_a) if g_a else '_'
        r_label = '+'.join(r_r) if r_r else '_'
        out.append({'gamma_pairs': g_a, 'rdm_rois': r_r, 'loco_v4': bool(l),
                     'label': f"γ{g_label}|RDM{r_label}|{'LOCO' if l else 'noLOCO'}"})
    return out


def aic_bic(test_loss_focal, k, n=2):
    if test_loss_focal is None or not np.isfinite(test_loss_focal) or test_loss_focal <= 0:
        return None, None
    L_per_n = test_loss_focal / n
    if L_per_n <= 0:
        return None, None
    return float(2 * k + n * np.log(L_per_n)), float(k * np.log(n) + n * np.log(L_per_n))


def fit_subject(subject, combo_start=None, combo_end=None, audit_full_roi=False):
    config = dict(SUBJECTS[subject])
    if audit_full_roi:
        # RDM atom 은 config['rdm_rois'] 로 *생성*되므로, 격자만 늘리면 atom 이 없어
        # 조합이 조용히 비어버린다. 생성 목록도 함께 확장한다. (원본 dict 은 불변)
        config['rdm_rois'] = list(ROIS)
    family = config['family']
    dl_sources = DELTA_LAMBDA_BY_FAMILY[family]
    focal_pair = config['focal_pair']
    print(f"\n[{subject}] family={family} focal={focal_pair}", flush=True)

    cvd_amps = {}
    hc_amps_all = {}
    K_by_roi = {}
    C_by_roi = {}
    for roi in ROIS:
        try:
            cvd_amps[roi] = load_amplitudes(subject, roi)
            hc_amps_all[roi] = load_hc_pool(roi)
            K_by_roi[roi] = ROI_K[roi]
            C_by_roi[roi] = create_basis_full(K_by_roi[roi], basis_type='fe')[
                HUE_ANGLES.astype(int)]
        except FileNotFoundError:
            pass

    try:
        cvd_jnd = load_jnd_per_pair(subject)
    except Exception:
        cvd_jnd = None

    if subject == 'sub-08':
        combos = enumerate_combos_sub08()
    else:
        combos = (enumerate_combos_sub09_full() if audit_full_roi
                   else enumerate_combos_sub09())
    if combo_start is not None and combo_end is not None:
        combos = combos[combo_start:combo_end]
        print(f"  Combo chunk: [{combo_start}:{combo_end}] = {len(combos)} combos", flush=True)
    print(f"  {len(combos)} combos × {N_RESAMPLES} resamples", flush=True)

    rng = np.random.default_rng(RNG_SEED + (0 if subject == 'sub-08' else 1))
    resample_subsets = []
    for _ in range(N_RESAMPLES):
        sel = rng.choice(len(HC_SUBJS), size=SUBSET_SIZE, replace=False)
        subset = [HC_SUBJS[i] for i in sorted(sel)]
        complement = [h for h in HC_SUBJS if h not in subset]
        resample_subsets.append((subset, complement))

    storage = {c['label']: {
        'config': c,
        **{f'rc_{src}': [] for src in dl_sources},
        '2comp': [],
    } for c in combos}

    # Test V1 RDM helper builder
    def make_test_V1_RDM(complement_pool_amps, C_baseline_v1, K_v1, cvd_v1_amp):
        if 'V1' not in cvd_amps or not complement_pool_amps:
            return None
        if len(complement_pool_amps) < 2:
            return None
        try:
            comp_W, _ = precompute_hc_W(complement_pool_amps, C_baseline_v1)
        except Exception:
            return None

        def fn(delta_8vec):
            try:
                return L_RDM(delta_8vec, cvd_v1_amp, complement_pool_amps,
                              comp_W, C_baseline_v1, K_v1, distance='correlation')
            except Exception:
                return np.nan
        return fn

    def make_test_per_pair(complement_jnd_subjs, cvd_jnd_local):
        try:
            test_bl, test_sd = jnd_baseline_from_pool(complement_jnd_subjs)
        except Exception:
            return None

        def fn(delta_8vec):
            out = {}
            perceived = (HUES + delta_8vec) % 360.0
            for p_name in PAIR_HUES.keys():
                if cvd_jnd_local is None or cvd_jnd_local.get(p_name) is None:
                    out[p_name] = None
                    continue
                if (p_name not in test_bl
                        or test_bl.get(p_name) is None
                        or test_sd.get(p_name) is None):
                    out[p_name] = None
                    continue
                theta_a, theta_b = PAIR_HUES[p_name]
                i = int(round(theta_a / 45.0)) % 8
                j = int(round(theta_b / 45.0)) % 8
                d_phys = min(abs(theta_a - theta_b) % 360,
                               360 - abs(theta_a - theta_b) % 360)
                d_perc_raw = abs(perceived[i] - perceived[j]) % 360
                d_perc = max(min(d_perc_raw, 360 - d_perc_raw), 1e-3)
                pred = test_bl[p_name] * (d_phys / d_perc)
                sigma = max(test_sd[p_name], 1e-3)
                out[p_name] = float(((pred - cvd_jnd_local[p_name]) / sigma) ** 2)
            return out
        return fn

    t_start = time.time()
    for draw_idx, (subset, complement) in enumerate(resample_subsets):
        train_jnd = [h for h in subset if h in HC_JND_SUBJS]
        test_jnd = [h for h in complement if h in HC_JND_SUBJS]
        if not train_jnd or not test_jnd:
            continue

        atoms = {}
        for p in config['pairs']:
            fn = make_gamma_pair_atom(p, cvd_jnd, train_jnd) if cvd_jnd else None
            if fn is not None:
                atoms[f'gamma_{p}'] = fn
        for roi in config['rdm_rois']:
            if roi in cvd_amps:
                pool_amps = {h: hc_amps_all[roi][h] for h in subset
                              if h in hc_amps_all[roi]}
                if len(pool_amps) >= 2:
                    fn = make_rdm_atom(roi, cvd_amps[roi], pool_amps,
                                         C_by_roi[roi], K_by_roi[roi])
                    if fn is not None:
                        atoms[f'rdm_{roi}'] = fn
        if 'V4' in cvd_amps:
            fn = make_loco_atom(cvd_amps['V4'], K_by_roi['V4'])
            if fn is not None:
                atoms['loco_V4'] = fn

        atom_grids_rc = {src: {} for src in dl_sources}
        for src, dl in dl_sources.items():
            for name, fn in atoms.items():
                atom_grids_rc[src][name] = grid_eval_rc(fn, dl, family)
        atom_grids_2c = {name: grid_eval_2comp(fn, family)
                          for name, fn in atoms.items()}

        # Per-atom TRAIN z-stats (mean, std) for re-normalizing TEST atom values
        train_stats_rc = {src: {} for src in dl_sources}
        for src in dl_sources:
            for name in atoms.keys():
                g = atom_grids_rc[src][name]
                train_stats_rc[src][name] = (float(np.nanmean(g)),
                                             float(np.nanstd(g)))
        train_stats_2c = {}
        for name in atoms.keys():
            g = atom_grids_2c[name]
            train_stats_2c[name] = (float(np.nanmean(g)),
                                    float(np.nanstd(g)))

        # Build TEST atom closures (same forms, but using complement HC pool)
        test_atoms = {}
        for p in config['pairs']:
            fn_test = (make_gamma_pair_atom(p, cvd_jnd, test_jnd)
                       if cvd_jnd else None)
            if fn_test is not None:
                test_atoms[f'gamma_{p}'] = fn_test
        for roi in config['rdm_rois']:
            if roi in cvd_amps:
                test_pool_amps = {h: hc_amps_all[roi][h] for h in complement
                                  if h in hc_amps_all[roi]}
                if len(test_pool_amps) >= 2:
                    fn_test = make_rdm_atom(roi, cvd_amps[roi], test_pool_amps,
                                            C_by_roi[roi], K_by_roi[roi])
                    if fn_test is not None:
                        test_atoms[f'rdm_{roi}'] = fn_test
        if 'V4' in cvd_amps and 'loco_V4' in atoms:
            test_atoms['loco_V4'] = atoms['loco_V4']  # CVD-only, no HC dep

        def composite_train_test(comp_train, atom_names_local, train_stats_for_model,
                                  delta_at_argmin):
            n_a_local = len(atom_names_local)
            flat = comp_train.flatten()
            if np.all(np.isnan(flat)):
                return None, None
            train_loss = float(np.nanmin(flat))
            z_test_parts = []
            for nm in atom_names_local:
                if nm not in train_stats_for_model or nm not in test_atoms:
                    continue
                mu, sd = train_stats_for_model[nm]
                if not np.isfinite(sd) or sd < 1e-10:
                    continue
                try:
                    t_val = float(test_atoms[nm](delta_at_argmin))
                except Exception:
                    continue
                if not np.isfinite(t_val):
                    continue
                z_test_parts.append((t_val - mu) / sd)
            if not z_test_parts:
                return train_loss, None
            test_loss = float(sum(z_test_parts) / np.sqrt(n_a_local))
            return train_loss, test_loss

        # Build test loss closures on TEST (complement) baseline
        test_bl, test_sd = jnd_baseline_from_pool(test_jnd)

        def test_aggregate(delta):
            if cvd_jnd is None:
                return None
            valid = {p: cvd_jnd[p] for p in test_bl.keys()
                     if cvd_jnd.get(p) is not None
                     and test_bl.get(p) is not None
                     and test_sd.get(p) is not None}
            if not valid:
                return None
            sd_d = {p: max(test_sd[p], 1e-3) for p in valid}
            try:
                return float(L_behav_gamma(delta, valid, test_bl, sd_d))
            except Exception:
                return None

        focal_theta_a, focal_theta_b = PAIR_HUES[focal_pair]
        f_i = int(round(focal_theta_a / 45.0)) % 8
        f_j = int(round(focal_theta_b / 45.0)) % 8
        focal_d_phys = min(abs(focal_theta_a - focal_theta_b) % 360,
                             360 - abs(focal_theta_a - focal_theta_b) % 360)
        focal_obs = cvd_jnd.get(focal_pair) if cvd_jnd else None
        focal_base = test_bl.get(focal_pair)
        focal_sd_v = (max(test_sd[focal_pair], 1e-3)
                      if focal_pair in test_sd and test_sd[focal_pair] is not None
                      else None)

        def test_focal(delta):
            if focal_obs is None or focal_base is None or focal_sd_v is None:
                return None
            perceived = (HUES + delta) % 360.0
            d_perc_raw = abs(perceived[f_i] - perceived[f_j]) % 360
            d_perc = max(min(d_perc_raw, 360 - d_perc_raw), 1e-3)
            pred = focal_base * (focal_d_phys / d_perc)
            return float(((pred - focal_obs) / focal_sd_v) ** 2)

        # Build test V1 RDM closure
        v1_pool = {h: hc_amps_all['V1'][h] for h in complement
                    if 'V1' in hc_amps_all and h in hc_amps_all['V1']}
        test_V1_RDM_fn = make_test_V1_RDM(v1_pool, C_by_roi.get('V1'),
                                            K_by_roi.get('V1', 6),
                                            cvd_amps.get('V1'))

        # Build test per-pair closure
        test_per_pair_fn = make_test_per_pair(test_jnd, cvd_jnd)

        for combo in combos:
            label = combo['label']
            atom_names = []
            for p in combo['gamma_pairs']:
                if f'gamma_{p}' in atoms:
                    atom_names.append(f'gamma_{p}')
            for roi in combo['rdm_rois']:
                if f'rdm_{roi}' in atoms:
                    atom_names.append(f'rdm_{roi}')
            if combo['loco_v4'] and 'loco_V4' in atoms:
                atom_names.append('loco_V4')
            if not atom_names:
                continue

            n_a = len(atom_names)

            for src in dl_sources:
                z_sum = None
                for name in atom_names:
                    z = zscore_grid(atom_grids_rc[src][name])
                    if np.all(np.isnan(z)):
                        z_sum = None; break
                    z_sum = z if z_sum is None else z_sum + z
                if z_sum is None:
                    continue
                comp = z_sum / np.sqrt(n_a)
                fit = argmin_rc(comp)
                if fit is None:
                    continue
                delta = forward_rc(dl_sources[src], fit['g'], family)
                train_loss, test_loss = composite_train_test(
                    comp, atom_names, train_stats_rc[src], delta)
                t_focal = test_focal(delta)
                t_agg = test_aggregate(delta)
                t_v1rdm = test_V1_RDM_fn(delta) if test_V1_RDM_fn else None
                t_pp = test_per_pair_fn(delta) if test_per_pair_fn else None
                aic, bic = aic_bic(t_focal, K_RC, n=2)
                storage[label][f'rc_{src}'].append({
                    'subset_idx': draw_idx,
                    'subset': subset, 'complement': complement,
                    'g': fit['g'], 'boundary': fit['boundary'],
                    'train_loss': train_loss, 'test_loss': test_loss,
                    'test_focal': t_focal, 'test_agg': t_agg,
                    'test_V1_RDM': t_v1rdm, 'test_per_pair': t_pp,
                    'aic': aic, 'bic': bic,
                })

            z_sum = None
            for name in atom_names:
                z = zscore_grid(atom_grids_2c[name])
                if np.all(np.isnan(z)):
                    z_sum = None; break
                z_sum = z if z_sum is None else z_sum + z
            if z_sum is None:
                continue
            comp = z_sum / np.sqrt(n_a)
            fit = argmin_2comp(comp)
            if fit is None:
                continue
            delta = forward_2comp(fit['beta_s'], fit['beta_c'], family)
            train_loss, test_loss = composite_train_test(
                comp, atom_names, train_stats_2c, delta)
            t_focal = test_focal(delta)
            t_agg = test_aggregate(delta)
            t_v1rdm = test_V1_RDM_fn(delta) if test_V1_RDM_fn else None
            t_pp = test_per_pair_fn(delta) if test_per_pair_fn else None
            aic, bic = aic_bic(t_focal, K_2C, n=2)
            storage[label]['2comp'].append({
                'subset_idx': draw_idx,
                'subset': subset, 'complement': complement,
                'beta_s': fit['beta_s'], 'beta_c': fit['beta_c'],
                'boundary': fit['boundary'],
                'train_loss': train_loss, 'test_loss': test_loss,
                'test_focal': t_focal, 'test_agg': t_agg,
                'test_V1_RDM': t_v1rdm, 'test_per_pair': t_pp,
                'aic': aic, 'bic': bic,
            })

        if (draw_idx + 1) % 20 == 0:
            elapsed = time.time() - t_start
            eta = elapsed * (N_RESAMPLES - draw_idx - 1) / (draw_idx + 1)
            print(f"  [{draw_idx + 1}/{N_RESAMPLES}] elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    return storage


def median_safe(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    return float(np.median(arr)) if len(arr) else None


def iqr_safe(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)])
    if len(arr) < 2:
        return None
    return float(np.percentile(arr, 75) - np.percentile(arr, 25))


def summarize(storage):
    out = {}
    for label, data in storage.items():
        config = data['config']
        per_model = {}
        for mkey, fits in data.items():
            if mkey == 'config':
                continue
            if not fits:
                per_model[mkey] = None
                continue
            test_focal = [f.get('test_focal') for f in fits]
            test_agg = [f.get('test_agg') for f in fits]
            test_v1rdm = [f.get('test_V1_RDM') for f in fits]
            train_loss_vals = [f.get('train_loss') for f in fits]
            test_loss_vals = [f.get('test_loss') for f in fits]
            boundary = [f.get('boundary', False) for f in fits]
            aic = [f.get('aic') for f in fits]
            bic = [f.get('bic') for f in fits]
            # Per-pair test medians
            per_pair_medians = {}
            for p in PAIR_HUES.keys():
                vals = []
                for f in fits:
                    pp = f.get('test_per_pair')
                    if pp and pp.get(p) is not None:
                        vals.append(pp[p])
                per_pair_medians[p] = median_safe(vals)

            if mkey.startswith('rc'):
                params = [f.get('g') for f in fits]
                param_summary = {
                    'g_median': median_safe(params),
                    'g_iqr': iqr_safe(params),
                }
            else:
                bs = [f.get('beta_s') for f in fits]
                bc = [f.get('beta_c') for f in fits]
                param_summary = {
                    'bs_median': median_safe(bs), 'bc_median': median_safe(bc),
                    'bs_iqr': iqr_safe(bs), 'bc_iqr': iqr_safe(bc),
                }
            per_model[mkey] = {
                'n': len(fits),
                'train_loss_median': median_safe(train_loss_vals),
                'train_loss_iqr': iqr_safe(train_loss_vals),
                'test_loss_median': median_safe(test_loss_vals),
                'test_loss_iqr': iqr_safe(test_loss_vals),
                'test_focal_median': median_safe(test_focal),
                'test_focal_iqr': iqr_safe(test_focal),
                'test_agg_median': median_safe(test_agg),
                'test_agg_iqr': iqr_safe(test_agg),
                'test_V1_RDM_median': median_safe(test_v1rdm),
                'test_V1_RDM_iqr': iqr_safe(test_v1rdm),
                'test_per_pair_medians': per_pair_medians,
                'boundary_rate': float(np.mean(boundary)) if boundary else None,
                'aic_median': median_safe(aic),
                'bic_median': median_safe(bic),
                'param_summary': param_summary,
            }
        out[label] = {'config': config, 'per_model': per_model}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True,
                         choices=['sub-08', 'sub-09'])
    parser.add_argument('--combo-start', type=int, default=None)
    parser.add_argument('--combo-end', type=int, default=None)
    parser.add_argument('--audit-full-roi', action='store_true',
                         help=('sub-09 의 RDM ROI 축을 V1 에서 V1--V4 로 완성해 동일 선택 규칙을 '
                               '재적용 (canonical 결과 파일은 건드리지 않고 _roi_audit 로 저장). '
                               'sub-08 은 이미 4-ROI 전탐색이라 무효.'))
    args = parser.parse_args()

    print("=" * 100, flush=True)
    print(f"S10b v3 — N={N_RESAMPLES} resample, extended test metrics", flush=True)
    print(f"  Tests: focal + agg + V1-RDM + per-pair (8)", flush=True)
    print(f"  Subject: {args.subject} chunk=[{args.combo_start}:{args.combo_end}]", flush=True)
    print("=" * 100, flush=True)

    t0 = time.time()
    storage = fit_subject(args.subject, args.combo_start, args.combo_end,
                           audit_full_roi=args.audit_full_roi)
    summary = summarize(storage)
    elapsed = round(time.time() - t0, 1)
    print(f"\n[{args.subject}] elapsed: {elapsed}s", flush=True)

    suffix = f"_{args.subject}"
    if args.audit_full_roi:
        suffix += "_roi_audit"
    if args.combo_start is not None:
        suffix += f"_c{args.combo_start:02d}-{args.combo_end:02d}"
    out_file = OUT_DIR / f"s10b_v6_pca_rdm_results{suffix}.json"
    with open(out_file, 'w') as f:
        json.dump({'subject': args.subject, 'storage': storage,
                    'summary': summary, 'elapsed': elapsed,
                    'meta': {'N_resamples': N_RESAMPLES, 'subset_size': SUBSET_SIZE,
                              'seed_base': RNG_SEED,
                              'combo_range': [args.combo_start, args.combo_end]}},
                   f, indent=2, default=str)
    print(f"Saved: {out_file}", flush=True)


if __name__ == "__main__":
    main()
