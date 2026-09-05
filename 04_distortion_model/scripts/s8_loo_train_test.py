"""S8: LOO + train-test of model-loss pairs (model-loss selection sprint).

CLAUDE.md §3 RE-OPENED (2026-05-23, USER DIRECTIVE). Phase 2 적합 모델·loss 미확정.

For each
  (model ∈ {R+C, 2-comp})
  × (loss ∈ {L_γ, L_α, L_LOCO, L_RDM})
  × (ROI ∈ {V1, V2, V3, V4})
  × (subject ∈ {sub-08, sub-09})
  × (HC LOO fold i ∈ HC_SUBJS):

  1. Form pool = HC \ {i}.
  2. Recompute pool reference (encoder W, ΔRDM ref, JND baseline) from pool.
  3. Fit CVD parameter against the loss with this pool reference → θ_CVD^(i).
  4. Fit held-out HC i with same pool reference → θ_HCi^(i) (train-test).

For R+C: also 3 Δλ sources per family (DPS/Boehm/JND-Lamb).
For 2-comp: no Δλ dependence; param = (β_s, β_c).

5 selection criteria computed downstream by s8_analysis.py:
  (a) parameter SD over 7 folds
  (b) CVD-HC separation rate (% folds CVD > 95th %ile held-out HC)
  (c) P1/P2a guardrail (internal, NOT primary endpoint per §0.1)
  (d) inter-loss correlation (across 7 folds × 4 losses)
  (e) train-test MSE on held-out HC

Output: results/s8_loo_train_test/loo_results.json
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/s8_loo_train_test.py  (development repository, commit 53c81c2)
# Role      : JND baseline from a control pool; Delta-lambda anchors per subtype.
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
import time
import numpy as np
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from rc_1dof import fit_rc_g
from two_comp import fit_2comp
from neural_loss import (
    load_amplitudes, load_hc_pool, ROI_K,
    precompute_loco_W_within, L_LOCO, design_from_delta,
)
from diagnostic_delta_rdm import precompute_hc_W, compute_rdm_correlation, cosine_similarity
from behav_loss import (
    load_jnd_per_pair, load_8afc_per_color,
    L_behav_alpha, L_behav_gamma, SIGMA_HC,
    PAIR_HUES, HC_JND_SUBJS, HC_8AFC_SUBJS,
)
from utils_forward_model import create_basis_full, HUE_ANGLES

OUT_DIR = SCRIPT_DIR.parent / "results" / "s8_loo_train_test"

HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
CVD_SUBJECTS = [('sub-08', 'deutan'), ('sub-09', 'protan')]
ROIS = ['V1', 'V2', 'V3', 'V4']

DELTA_LAMBDA_BY_FAMILY = {
    'deutan': {'DPS_lit': 6.0, 'Boehm_mid': 8.0, 'JND_Lamb': 6.5},
    'protan': {'DPS_lit': 10.0, 'Boehm_low': 3.0, 'JND_Lamb': 1.5},
}


def jnd_baseline_from_pool(pool_subjs: list) -> tuple:
    """LOO HC JND baseline: mean & SD per pair across pool_subjs."""
    hc_data = []
    for s in pool_subjs:
        try:
            hc_data.append(load_jnd_per_pair(s))
        except Exception:
            continue
    baseline = {}
    sd = {}
    for p in PAIR_HUES.keys():
        vals = [d[p] for d in hc_data if d.get(p) is not None]
        if len(vals) >= 2:
            baseline[p] = float(np.mean(vals))
            sd[p] = float(np.std(vals, ddof=1))
        else:
            baseline[p] = None
            sd[p] = None
    return baseline, sd


def make_loss_fns(target_amp, target_jnd, target_8afc, target_loco_W,
                  pool_amps_dict, pool_jnd_baseline, pool_jnd_sd,
                  pool_hc_W, C_baseline, K, family) -> dict:
    """Closures over (target subject) + (pool reference)."""
    # ΔRDM_obs = RDM(target) − mean(RDM(pool))
    rdm_target = compute_rdm_correlation(target_amp.mean(axis=0))
    rdm_pool = np.mean(
        [compute_rdm_correlation(amp.mean(axis=0)) for amp in pool_amps_dict.values()],
        axis=0,
    )
    delta_rdm_obs = rdm_target - rdm_pool
    n_obs = np.linalg.norm(delta_rdm_obs)

    def L_gamma_fn(delta_rc):
        if target_jnd is None:
            return np.nan
        # use only pairs that have target and baseline both defined
        valid_pairs = {p: target_jnd[p] for p in PAIR_HUES.keys()
                       if target_jnd.get(p) is not None and pool_jnd_baseline.get(p) is not None}
        if not valid_pairs:
            return np.nan
        bl = {p: pool_jnd_baseline[p] for p in valid_pairs}
        sd_d = {p: pool_jnd_sd[p] if pool_jnd_sd[p] else 1.0 for p in valid_pairs}
        return L_behav_gamma(delta_rc, valid_pairs, bl, sd_d)

    def L_alpha_fn(delta_rc):
        if target_8afc is None:
            return np.nan
        return L_behav_alpha(delta_rc, target_8afc, SIGMA_HC)

    def L_loco_fn(delta_rc):
        return L_LOCO(delta_rc, target_amp, target_loco_W, K)

    def L_rdm_fn(delta_rc):
        C_shifted = design_from_delta(delta_rc, n_channels=K)
        delta_rdm_sims = []
        for s_id, W in pool_hc_W.items():
            Y_shifted = C_shifted @ W
            Y_base = C_baseline @ W
            delta_rdm_sims.append(compute_rdm_correlation(Y_shifted) -
                                   compute_rdm_correlation(Y_base))
        delta_rdm_sim = np.mean(delta_rdm_sims, axis=0)
        n_sim = np.linalg.norm(delta_rdm_sim)
        if n_sim < 1e-10 or n_obs < 1e-10:
            return 1.0
        cos = float(np.dot(delta_rdm_sim, delta_rdm_obs) / (n_sim * n_obs))
        return 1.0 - cos

    return {
        'L_gamma': L_gamma_fn,
        'L_alpha': L_alpha_fn,
        'L_LOCO': L_loco_fn,
        'L_RDM': L_rdm_fn,
    }


def fit_all_losses(loss_fns: dict, family: str, dl_sources: dict) -> dict:
    """Run R+C (per Δλ source) and 2-comp fits for each loss."""
    out = {}
    for loss_name, fn in loss_fns.items():
        # Skip if loss is NaN at HC baseline (e.g., 8AFC data missing)
        try:
            test = fn(np.zeros(8))
            if not np.isfinite(test):
                out[loss_name] = None
                continue
        except Exception:
            out[loss_name] = None
            continue

        rc_fits = {}
        for src_name, dl in dl_sources.items():
            rc_fits[src_name] = fit_rc_g(dl, family, fn)
        two_comp_fit = fit_2comp(family, fn)
        out[loss_name] = {
            'rc': {k: {
                'g_best': v['g_best'],
                'loss_best': v['loss_best'],
                'boundary_low': v['boundary_low'],
                'boundary_high': v['boundary_high'],
                'delta_lambda': dl_sources[k],
            } for k, v in rc_fits.items()},
            '2comp': {
                'beta_s_best': two_comp_fit['beta_s_best'],
                'beta_c_best': two_comp_fit['beta_c_best'],
                'loss_best': two_comp_fit['loss_best'],
                'boundary_bs': two_comp_fit['boundary_bs'],
                'boundary_bc': two_comp_fit['boundary_bc'],
            },
        }
    return out


def process_subject_roi(subject: str, family: str, roi: str) -> dict:
    """LOO over HC for one (subject, ROI) cell."""
    K = ROI_K[roi]
    C_baseline = create_basis_full(K, basis_type='fe')[HUE_ANGLES.astype(int)]

    # Load CVD data
    cvd_amp = load_amplitudes(subject, roi)
    cvd_jnd = load_jnd_per_pair(subject)
    try:
        cvd_8afc = load_8afc_per_color(subject)
    except Exception:
        cvd_8afc = None
    cvd_loco_W, _ = precompute_loco_W_within(cvd_amp, C_baseline)

    # Load all 7 HC data once
    hc_amps_all = load_hc_pool(roi)  # dict {hc: amp}, may skip sparse HC for V4
    available_hc = list(hc_amps_all.keys())
    hc_loco_W = {h: precompute_loco_W_within(hc_amps_all[h], C_baseline)[0] for h in available_hc}
    hc_jnd_all = {}
    hc_8afc_all = {}
    for h in HC_SUBJS:
        try:
            hc_jnd_all[h] = load_jnd_per_pair(h)
        except Exception:
            hc_jnd_all[h] = None
        try:
            if h in HC_8AFC_SUBJS:
                hc_8afc_all[h] = load_8afc_per_color(h)
            else:
                hc_8afc_all[h] = None
        except Exception:
            hc_8afc_all[h] = None

    dl_sources = DELTA_LAMBDA_BY_FAMILY[family]

    folds = []
    for hold_out in HC_SUBJS:
        fold_t0 = time.time()
        pool_subjs = [h for h in available_hc if h != hold_out]
        if len(pool_subjs) < 2:
            continue
        pool_amps = {h: hc_amps_all[h] for h in pool_subjs}
        pool_jnd_subjs = [h for h in HC_JND_SUBJS if h != hold_out]
        pool_jnd_baseline, pool_jnd_sd = jnd_baseline_from_pool(pool_jnd_subjs)
        pool_hc_W, _ = precompute_hc_W(pool_amps, C_baseline)

        # --- Fit CVD using pool reference ---
        cvd_loss_fns = make_loss_fns(
            cvd_amp, cvd_jnd, cvd_8afc, cvd_loco_W,
            pool_amps, pool_jnd_baseline, pool_jnd_sd,
            pool_hc_W, C_baseline, K, family,
        )
        cvd_fits = fit_all_losses(cvd_loss_fns, family, dl_sources)

        # --- Fit held-out HC using SAME pool reference (train-test) ---
        if hold_out not in hc_amps_all:
            hc_fits = None
        else:
            ho_amp = hc_amps_all[hold_out]
            ho_jnd = hc_jnd_all.get(hold_out)
            ho_8afc = hc_8afc_all.get(hold_out)
            ho_loco_W = hc_loco_W[hold_out]
            ho_loss_fns = make_loss_fns(
                ho_amp, ho_jnd, ho_8afc, ho_loco_W,
                pool_amps, pool_jnd_baseline, pool_jnd_sd,
                pool_hc_W, C_baseline, K, family,
            )
            hc_fits = fit_all_losses(ho_loss_fns, family, dl_sources)

        folds.append({
            'hold_out_hc': hold_out,
            'pool_size': len(pool_subjs),
            'cvd_fits': cvd_fits,
            'hc_held_out_fits': hc_fits,
            'fold_secs': round(time.time() - fold_t0, 2),
        })
        print(f"    fold hold-out={hold_out}: {round(time.time() - fold_t0, 1)}s")

    return {
        'subject': subject,
        'family': family,
        'roi': roi,
        'K': K,
        'available_hc': available_hc,
        'folds': folds,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("S8: LOO + train-test of model-loss pairs (selection sprint)")
    print("  4 losses × 2 models × 4 ROIs × 2 CVD × 7 LOO folds")
    print("=" * 78)

    t0 = time.time()
    results = {}
    for subject, family in CVD_SUBJECTS:
        results[subject] = {'family': family, 'rois': {}}
        for roi in ROIS:
            print(f"\n[{subject} ({family}) × {roi}]")
            t_roi = time.time()
            cell = process_subject_roi(subject, family, roi)
            results[subject]['rois'][roi] = cell
            print(f"  → {round(time.time() - t_roi, 1)}s")

    total_secs = round(time.time() - t0, 1)
    out = {
        'design': '4 losses × 2 models × 4 ROIs × 2 CVD × 7 LOO HC folds',
        'losses': ['L_gamma', 'L_alpha', 'L_LOCO', 'L_RDM'],
        'models': ['rc_1dof (g)', '2comp (beta_s, beta_c)'],
        'rois': ROIS,
        'subjects': [s[0] for s in CVD_SUBJECTS],
        'delta_lambda_sources': DELTA_LAMBDA_BY_FAMILY,
        'K_per_roi': ROI_K,
        'total_secs': total_secs,
        'results': results,
    }
    with open(OUT_DIR / "loo_results.json", 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {OUT_DIR / 'loo_results.json'}")
    print(f"Total: {total_secs}s")


if __name__ == "__main__":
    main()
