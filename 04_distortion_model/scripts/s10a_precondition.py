"""S10a: Loss precondition table — Cohen's d for CVD vs HC LOO at δθ=0.

For each loss ∈ {L_γ, L_RDM, L_LOCO} × each cell (subject × ROI):
  - HC LOO: hold out 1 HC, compute loss using 6-HC pool as baseline → 7 values
  - CVD: compute loss using all 7 HC as baseline → 1 value
  - Cohen's d = (CVD_loss − HC_LOO_mean) / HC_LOO_sd

Output: results/s10_inclusion/precondition_table.json + ASCII table.

Pass criterion: Cohen's d ≥ 0.5 (medium effect).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/s10a_precondition.py  (development repository, commit 53c81c2)
# Role      : Gate 1 separation precondition (Cohen's d of each atom vs the control LOO distribution) -> precondition_table.json
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

import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from neural_loss import (
    load_amplitudes, load_hc_pool, ROI_K, L_LOCO, L_RDM,
    precompute_loco_W_within,
)
from diagnostic_delta_rdm import precompute_hc_W, compute_rdm_correlation
from behav_loss import (
    load_jnd_per_pair, L_behav_gamma, HC_JND_SUBJS, PAIR_HUES,
)
from utils_forward_model import create_basis_full, HUE_ANGLES
from s8_loo_train_test import jnd_baseline_from_pool

OUT_DIR = SCRIPT_DIR.parent / "results" / "s10_inclusion"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CELLS = [
    ('sub-08', 'deutan'), ('sub-09', 'protan'), ('sub-10', 'deutan'),
]
ROIS = ['V1', 'V2', 'V3', 'V4']
HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']


def compute_L_gamma_at_zero(subject_jnd, pool_jnd_subjs):
    """L_γ(δθ=0) using pool_jnd_subjs as baseline.

    Returns dict: {mean, max, top3_mean, pair_z2}.
    pair_z2: per-pair squared z-score (raw, no aggregation).
    """
    if not pool_jnd_subjs or subject_jnd is None:
        return None
    bl, sd = jnd_baseline_from_pool(pool_jnd_subjs)
    pair_z2 = {}
    for p in bl.keys():
        if subject_jnd.get(p) is None:
            continue
        sigma = max(sd[p], 1e-3)
        # At δθ=0, predicted JND = baseline → z² = ((obs - baseline) / sd)²
        z = (subject_jnd[p] - bl[p]) / sigma
        pair_z2[p] = float(z ** 2)
    if not pair_z2:
        return None
    vals = list(pair_z2.values())
    sorted_vals = sorted(vals, reverse=True)
    return {
        'mean': float(np.mean(vals)),
        'max': float(np.max(vals)),
        'top3_mean': float(np.mean(sorted_vals[:3])),
        'pair_z2': pair_z2,
    }


def compute_L_RDM_at_zero(target_amp, pool_amps_dict, C_baseline, K):
    """ΔRDM_obs Frobenius norm: ||target_RDM − pool_mean_RDM||_F.

    Static precondition metric (no forward simulation).
    L_RDM cosine itself is degenerate at δθ=0 (ΔRDM_sim ≈ 0).
    """
    if len(pool_amps_dict) < 2:
        return None
    try:
        # Pool mean RDM
        pool_rdms = []
        for _s, amp in pool_amps_dict.items():
            mean_amp = amp.mean(axis=0)  # (8, V_s)
            pool_rdms.append(compute_rdm_correlation(mean_amp))
        pool_mean_rdm = np.mean(pool_rdms, axis=0)
        target_rdm = compute_rdm_correlation(target_amp.mean(axis=0))
        delta_rdm = target_rdm - pool_mean_rdm
        # delta_rdm is 1-D (28,) upper-triangle; use Euclidean norm
        return float(np.linalg.norm(delta_rdm))
    except Exception as e:
        return None


def compute_L_LOCO_at_zero(target_amp, C_baseline, K):
    """L_LOCO(δθ=0) — within-subject, no pool needed."""
    try:
        loco_W, _ = precompute_loco_W_within(target_amp, C_baseline)
        return float(L_LOCO(np.zeros(8), target_amp, loco_W, K))
    except Exception:
        return None


def cohens_d(cvd_val, hc_values):
    arr = np.array([v for v in hc_values if v is not None and np.isfinite(v)])
    if len(arr) < 2 or cvd_val is None or not np.isfinite(cvd_val):
        return None
    mu = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    if sd < 1e-10:
        return None
    return (cvd_val - mu) / sd, mu, sd


def main():
    print("=" * 92)
    print("S10a Precondition — Cohen's d (CVD vs HC LOO) at δθ=0")
    print("  Pass: d ≥ 0.5 (medium); d ≥ 0.8 strong")
    print("=" * 92)

    results = {}
    for roi in ROIS:
        K = ROI_K[roi]
        C_baseline = create_basis_full(K, basis_type='fe')[HUE_ANGLES.astype(int)]
        hc_amps_all = load_hc_pool(roi)
        hc_avail = list(hc_amps_all.keys())
        print(f"\n[{roi}] K={K}  HC w/ amps: {len(hc_avail)}/{len(HC_SUBJS)} {hc_avail}")

        # HC LOO values for each loss
        hc_loo = {'L_gamma_mean': [], 'L_gamma_max': [], 'L_gamma_top3': [],
                   'L_RDM': [], 'L_LOCO': []}
        for h in HC_SUBJS:
            # L_γ: leave h out of pool, target = h
            other_jnd = [j for j in HC_JND_SUBJS if j != h]
            if h in HC_JND_SUBJS:
                try:
                    h_jnd = load_jnd_per_pair(h)
                except Exception:
                    h_jnd = None
                if h_jnd:
                    r = compute_L_gamma_at_zero(h_jnd, other_jnd)
                    if r is not None:
                        hc_loo['L_gamma_mean'].append(r['mean'])
                        hc_loo['L_gamma_max'].append(r['max'])
                        hc_loo['L_gamma_top3'].append(r['top3_mean'])
                    else:
                        hc_loo['L_gamma_mean'].append(None)
                        hc_loo['L_gamma_max'].append(None)
                        hc_loo['L_gamma_top3'].append(None)
                else:
                    for k in ['L_gamma_mean', 'L_gamma_max', 'L_gamma_top3']:
                        hc_loo[k].append(None)
            else:
                for k in ['L_gamma_mean', 'L_gamma_max', 'L_gamma_top3']:
                    hc_loo[k].append(None)

            # L_RDM, L_LOCO: target = h amps, pool = other HC amps
            if h in hc_amps_all:
                h_amp = hc_amps_all[h]
                other_amps = {k: v for k, v in hc_amps_all.items() if k != h}
                hc_loo['L_RDM'].append(
                    compute_L_RDM_at_zero(h_amp, other_amps, C_baseline, K),
                )
                hc_loo['L_LOCO'].append(
                    compute_L_LOCO_at_zero(h_amp, C_baseline, K),
                )
            else:
                hc_loo['L_RDM'].append(None)
                hc_loo['L_LOCO'].append(None)

        results[roi] = {'hc_loo': hc_loo, 'hc_subjects': HC_SUBJS, 'cvd_values': {}, 'cohens_d': {}}

        # CVD values (full HC pool as baseline)
        for (sub, fam) in CELLS:
            try:
                cvd_amp = load_amplitudes(sub, roi)
            except FileNotFoundError:
                print(f"  {sub} {roi}: amp not found — skip")
                continue
            try:
                cvd_jnd = load_jnd_per_pair(sub)
            except Exception:
                cvd_jnd = None

            r_g = compute_L_gamma_at_zero(cvd_jnd, HC_JND_SUBJS) if cvd_jnd else None
            l_gamma_mean = r_g['mean'] if r_g else None
            l_gamma_max = r_g['max'] if r_g else None
            l_gamma_top3 = r_g['top3_mean'] if r_g else None
            pair_z2 = r_g['pair_z2'] if r_g else None
            l_rdm = compute_L_RDM_at_zero(cvd_amp, hc_amps_all, C_baseline, K)
            l_loco = compute_L_LOCO_at_zero(cvd_amp, C_baseline, K)

            cvd_vals = {
                'L_gamma_mean': l_gamma_mean, 'L_gamma_max': l_gamma_max,
                'L_gamma_top3': l_gamma_top3, 'pair_z2': pair_z2,
                'L_RDM': l_rdm, 'L_LOCO': l_loco,
            }
            results[roi]['cvd_values'][sub] = cvd_vals

            ds = {}
            for loss in ['L_gamma_mean', 'L_gamma_max', 'L_gamma_top3', 'L_RDM', 'L_LOCO']:
                res = cohens_d(cvd_vals[loss], hc_loo[loss])
                if res is None:
                    ds[loss] = {'d': None, 'hc_mean': None, 'hc_sd': None, 'pass': False}
                else:
                    d, mu, sd = res
                    ds[loss] = {
                        'd': round(d, 3), 'hc_mean': round(mu, 4),
                        'hc_sd': round(sd, 4), 'pass': bool(abs(d) >= 0.5),
                    }
            results[roi]['cohens_d'][sub] = ds

            print(f"  {sub:8s} ({fam:7s}): " + " | ".join([
                f"{loss[2:]:9s} d={ds[loss]['d']:+5.2f}" if ds[loss]['d'] is not None else f"{loss[2:]:9s} d=NA"
                for loss in ['L_gamma_mean', 'L_gamma_max', 'L_gamma_top3', 'L_RDM', 'L_LOCO']
            ]))

    out_file = OUT_DIR / "precondition_table.json"
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")

    # ASCII summary table
    LOSSES_TBL = ['L_gamma_mean', 'L_gamma_max', 'L_gamma_top3', 'L_RDM', 'L_LOCO']
    print("\n" + "=" * 120)
    print("PRECONDITION TABLE — Cohen's d (CVD − HC LOO) / HC SD at δθ=0")
    print("  L_LOCO: paper-rule V4 only (V1-V3 perm-null fail)")
    print("=" * 120)
    header = f"{'Cell':18s} | " + " | ".join([f"{l.replace('L_','').replace('gamma','γ'):>10s}" for l in LOSSES_TBL])
    print(header)
    print("-" * 120)
    pass_count = {l: 0 for l in LOSSES_TBL}
    total_cells = 0
    for roi in ROIS:
        for sub, _fam in CELLS:
            if sub not in results[roi]['cohens_d']:
                continue
            total_cells += 1
            ds = results[roi]['cohens_d'][sub]
            cell_key = f"{sub} {roi}"
            row = f"{cell_key:18s} | "
            for loss in LOSSES_TBL:
                d = ds[loss]['d']
                # L_LOCO only counted if V4 (paper rule)
                effective_pass = ds[loss]['pass'] and (loss != 'L_LOCO' or roi == 'V4')
                if effective_pass:
                    pass_count[loss] += 1
                mark = '✓' if effective_pass else ('·' if ds[loss]['pass'] else ' ')
                row += f"{d:+6.2f} {mark}   | " if d is not None else f"{'NA':>10s} | "
            print(row)
    n_v4 = sum(1 for sub, _ in CELLS if sub in results.get('V4', {}).get('cohens_d', {}))
    print(f"\nPass count (d ≥ 0.5):")
    for l in LOSSES_TBL:
        denom = n_v4 if l == 'L_LOCO' else total_cells
        print(f"  {l:15s}: {pass_count[l]:2d}/{denom}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nElapsed: {time.time() - t0:.1f}s")
