#!/usr/bin/env python3
"""
exp2 HC-likeness analysis — does the Optimal filter make CVD voxel response
more HC-like than the Window filter?

PRIMARY: within-condition LOCO rho (voxel-response prediction, ridge_gcv FE-6,
         leave-one-color-out, delta_theta=0) per condition, compared to the HC
         exp1 within-subject LOCO rho distribution via Cohen's d.
CONVERGENT (descriptive): LORO color-decoding accuracy; RDM cosine to HC-mean.

Scope: sub-08 only, single-subject DESCRIPTIVE (Cohen's d, no inferential p).
V1 primary; V2/V3 supporting; V4(hV4) descriptive-only.

Reuses FROZEN helpers from phase4_forward_model/utils_forward_model.py.
LOCO loop reimplemented locally to read n_runs from data shape (canonical
neural_loss.precompute_loco_W_within hardcodes N_RUNS=6).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp2_hc_likeness.py  (development repository, commit 53c81c2)
# Role      : LORO / LOCO readouts per condition vs the run-matched control reference -> exp2_hc_likeness_sub-*_matched.json
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
import argparse
import itertools
import numpy as np
from pathlib import Path

N_RUNS_EXP2 = 4   # exp2 runs per condition; HC subsampled to this for fair d

ROOT = Path("/scratch/connectome/haba6030/colorBlind")
HC_C010 = ROOT / "derivatives" / "full_dataset_C010"
OUT_DIR = ROOT / "analysis" / "phase6_behavioral_analysis" / "exp2_neural" / "results"
# EXP2_C010 + output filename set per variant in main():
#   native  -> full_dataset_C010_exp2          (858 vox @V1; no-filter anchor 560 -> voxel caveat)
#   matched -> full_dataset_C010_exp2_matched  (560 vox; voxel-identical to no-filter anchor)
EXP2_C010 = None

sys.path.insert(0, str(ROOT / "analysis" / "phase4_forward_model" / "scripts"))
from utils_forward_model import (
    create_basis_matrix, HUE_ANGLES, fit_W_ridge, gcv_select_alpha,
    decode_hue, circular_distance,
)
from loco_canonical import loco_forward_readouts  # shared canonical (decode=ols / encode=gcv)

HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROIS = ['V1', 'V2', 'V3', 'V4']
K = 6  # FE-6 uniform (Phase-2 neural domain)
CONDITIONS = ['window', 'optimal']
N_COLORS = 8
_BASIS360 = create_basis_matrix(np.arange(360), K, basis_type='fe')  # (360, K) for decoding


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def loco_rho_within(amp, C_baseline):
    """Within-subject leave-one-color-out ENCODING rho (forward-tuning).
    amp: (n_runs, 8, V). Returns rho (8,). Thin wrapper over the shared canonical
    with decoder='gcv' (ridge-GCV) — the correct decoder for voxel-pattern
    prediction (see analysis/decoder-comparison.md). Behavior is identical to the
    previous in-line implementation (pooled-run gcv, pred vs run-mean actual)."""
    return loco_forward_readouts(amp, C_baseline, decoder='gcv', tasks=('rho',))['rho']


def loco_rho_subsampled(amp, C_baseline, n_sub):
    """Mean within-subject LOCO rho over ALL n_sub-run subsets of amp.
    Run-count-matched reference (e.g. HC 6-run -> n_sub=4 to match exp2)."""
    n_runs = amp.shape[0]
    if n_runs <= n_sub:
        return float(loco_rho_within(amp, C_baseline).mean())
    vals = []
    for subset in itertools.combinations(range(n_runs), n_sub):
        vals.append(float(loco_rho_within(amp[list(subset)], C_baseline).mean()))
    return float(np.mean(vals))


def loco_rho_loro_stability(amp, C_baseline):
    """Leave-one-run-out: recompute mean LOCO rho dropping each run in turn.
    Returns list of (n_runs) estimates — robustness to any single run."""
    n_runs = amp.shape[0]
    out = []
    for drop in range(n_runs):
        keep = [r for r in range(n_runs) if r != drop]
        out.append(float(loco_rho_within(amp[keep], C_baseline).mean()))
    return out


# ---------------------------------------------------------------------------
# LOCO interpolation ACCURACY = Phase-1 "adjacent accuracy" (methods_v2.tex
# L121-125). Decoder = ols (alpha=0 pseudoinverse, B&H 2009) — the CORRECT decoder
# for hue decoding: ridge-GCV over-regularizes the inverse and collapses adjacent
# accuracy to chance (HC hV4 0.38 vs the ols/paper 0.47). See decoder-comparison.md.
# Per-run decode over the 360-hue FE basis; adjacent = err<=45deg (±1 step),
# exact = nearest-45deg-bin hit; chance adjacent 3/8, exact 1/8.
# ---------------------------------------------------------------------------
def loco_accuracy_within(amp, C_baseline):
    """Within-subject leave-one-color-out DECODING accuracy (alpha=0 pseudoinverse).
    amp: (n_runs, 8, V). Returns (adjacent (8,), exact (8,)), each in [0,1].
    Thin wrapper over the shared canonical with decoder='ols' (B&H alpha=0)."""
    r = loco_forward_readouts(amp, C_baseline, _BASIS360, decoder='ols',
                              tasks=('adj', 'exact'))
    return r['adj'], r['exact']


def loco_accuracy_subsampled(amp, C_baseline, n_sub):
    """Mean within-subject LOCO accuracy over ALL n_sub-run subsets of amp.
    Run-count-matched reference (HC 6-run -> n_sub=4 to match exp2).
    Returns (mean_adjacent, mean_exact) scalars (means over color and subset)."""
    n_runs = amp.shape[0]
    if n_runs <= n_sub:
        adj, exa = loco_accuracy_within(amp, C_baseline)
        return float(adj.mean()), float(exa.mean())
    adj_vals, exa_vals = [], []
    for subset in itertools.combinations(range(n_runs), n_sub):
        a, e = loco_accuracy_within(amp[list(subset)], C_baseline)
        adj_vals.append(float(a.mean()))
        exa_vals.append(float(e.mean()))
    return float(np.mean(adj_vals)), float(np.mean(exa_vals))


def loco_accuracy_loro_stability(amp, C_baseline):
    """Leave-one-run-out: recompute mean LOCO adjacent accuracy dropping each
    run in turn. Returns list of (n_runs) mean-adjacent estimates."""
    n_runs = amp.shape[0]
    out = []
    for drop in range(n_runs):
        keep = [r for r in range(n_runs) if r != drop]
        adj, _ = loco_accuracy_within(amp[keep], C_baseline)
        out.append(float(adj.mean()))
    return out


def loro_accuracy(amp):
    """Leave-one-run-out color decoding via correlation template matching.
    amp: (n_runs, 8, V). Returns accuracy over n_runs*8 trials."""
    n_runs, n_colors, V = amp.shape
    correct = 0
    total = 0
    for held in range(n_runs):
        train_runs = [r for r in range(n_runs) if r != held]
        templates = amp[train_runs].mean(axis=0)  # (8, V) per-color template
        for c in range(n_colors):
            test = amp[held, c]                   # (V,)
            corrs = [np.corrcoef(test, templates[k])[0, 1] for k in range(n_colors)]
            corrs = [x if np.isfinite(x) else -np.inf for x in corrs]
            pred = int(np.argmax(corrs))
            correct += int(pred == c)
            total += 1
    return correct / total if total else np.nan


def loro_accuracy_fe6(amp, C_baseline):
    """Leave-one-run-out color decoding via an FE-6 forward encoder.
    amp: (n_runs, 8, V). C_baseline: (8, K). Returns accuracy over n_runs*8 trials.

    Per held-out run r: pooled-W ridge_gcv training on remaining runs
    (C_train = tile(C_baseline), X_train = amp[train].reshape(-1, V)),
    FE-predicted templates T = C_baseline @ W (8, V), then voxel-space
    Pearson-correlation template matching (argmax over the 8 templates).
    """
    n_runs, n_colors, V = amp.shape
    correct = 0
    total = 0
    for held in range(n_runs):
        train_runs = [r for r in range(n_runs) if r != held]
        C_train = np.tile(C_baseline, (len(train_runs), 1))   # (len*8, K)
        X_train = amp[train_runs].reshape(-1, V)              # (len*8, V)
        alpha, _ = gcv_select_alpha(C_train, X_train)
        W = fit_W_ridge(C_train, X_train, alpha)              # (K, V)
        T = C_baseline @ W                                    # (8, V) FE templates
        a_bad = np.std(amp[held], axis=1) < 1e-10             # (8,) degenerate actuals
        t_bad = np.std(T, axis=1) < 1e-10                     # (8,) degenerate templates
        for c in range(n_colors):
            actual = amp[held, c]                             # (V,)
            corrs = []
            for k in range(n_colors):
                if a_bad[c] or t_bad[k]:
                    corrs.append(-np.inf)
                    continue
                r = np.corrcoef(actual, T[k])[0, 1]
                corrs.append(r if np.isfinite(r) else -np.inf)
            pred = int(np.argmax(corrs))
            correct += int(pred == c)
            total += 1
    return correct / total if total else np.nan


def corr_rdm(mean_pattern):
    """8x8 correlation-distance RDM upper triangle (28,)."""
    rdm = 1.0 - np.corrcoef(mean_pattern)  # (8,8)
    iu = np.triu_indices(N_COLORS, k=1)
    return rdm[iu]


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return np.nan
    return float(np.dot(a, b) / (na * nb))


def cohens_d(x, ref_mean, ref_sd):
    return float((x - ref_mean) / ref_sd) if ref_sd > 1e-12 else np.nan


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def load_npy(path):
    return np.load(path) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='native', choices=['native', 'matched'])
    ap.add_argument('--subject', default='08', help='CVD subject id, e.g. 08 / 09')
    args = ap.parse_args()
    subj = f"sub-{args.subject}"
    global EXP2_C010
    EXP2_C010 = ROOT / "derivatives" / (
        "full_dataset_C010_exp2" if args.variant == 'native'
        else "full_dataset_C010_exp2_matched")
    out_name = f"exp2_hc_likeness_{subj}_{args.variant}.json"
    print(f"VARIANT={args.variant}  EXP2_C010={EXP2_C010}")

    results = {}
    C_baseline = create_basis_matrix(HUE_ANGLES, K, basis_type='fe')  # (8, K)

    for roi in ROIS:
        print(f"\n{'='*72}\nROI {roi}\n{'='*72}")

        # ---- HC exp1 distribution. Full 6-run AND run-matched (n=4) refs ----
        hc_rho6, hc_rho4, hc_acc, hc_acc_fe6, hc_rdms = [], [], [], [], []
        # LOCO interpolation accuracy (adjacent / exact), n6 full + n4 run-matched
        hc_adj6, hc_adj4, hc_exa6, hc_exa4 = [], [], [], []
        for s in HC_SUBJS:
            amp = load_npy(HC_C010 / s / roi / "amplitudes_procrustes.npy")
            if amp is None:
                continue
            if roi == 'V4' and amp.shape[2] < 20:   # sub-07 hV4 sparse
                print(f"  [HC] {s} {roi} excluded ({amp.shape[2]} voxels)")
                continue
            hc_rho6.append(float(loco_rho_within(amp, C_baseline).mean()))
            hc_rho4.append(loco_rho_subsampled(amp, C_baseline, N_RUNS_EXP2))  # run-matched
            adj6, exa6 = loco_accuracy_within(amp, C_baseline)               # n=6 full
            hc_adj6.append(float(adj6.mean())); hc_exa6.append(float(exa6.mean()))
            adj4, exa4 = loco_accuracy_subsampled(amp, C_baseline, N_RUNS_EXP2)  # run-matched
            hc_adj4.append(adj4); hc_exa4.append(exa4)
            hc_acc.append(float(loro_accuracy(amp)))
            hc_acc_fe6.append(float(loro_accuracy_fe6(amp, C_baseline)))  # FE-6 LORO
            hc_rdms.append(corr_rdm(amp.mean(axis=0)))
        hc_rho6 = np.array(hc_rho6); hc_rho4 = np.array(hc_rho4); hc_acc = np.array(hc_acc)
        hc_acc_fe6 = np.array(hc_acc_fe6)
        hc_adj6 = np.array(hc_adj6); hc_adj4 = np.array(hc_adj4)
        hc_exa6 = np.array(hc_exa6); hc_exa4 = np.array(hc_exa4)
        hc_rdm_mean = np.mean(hc_rdms, axis=0)
        hc_mu, hc_sd = hc_rho4.mean(), hc_rho4.std(ddof=1)        # PRIMARY ref = run-matched
        hc_mu6, hc_sd6 = hc_rho6.mean(), hc_rho6.std(ddof=1)
        # adjacent-accuracy HC refs (PRIMARY = run-matched n=4, parallel to rho)
        hc_adj_mu, hc_adj_sd = hc_adj4.mean(), hc_adj4.std(ddof=1)
        hc_adj_mu6, hc_adj_sd6 = hc_adj6.mean(), hc_adj6.std(ddof=1)
        hc_exa_mu, hc_exa_sd = hc_exa4.mean(), hc_exa4.std(ddof=1)
        hc_exa_mu6 = hc_exa6.mean()
        print(f"  HC LOCO rho [n=4 matched]: mean={hc_mu:.4f} sd={hc_sd:.4f} "
              f"| [n=6 full]: mean={hc_mu6:.4f} sd={hc_sd6:.4f} | LORO acc={hc_acc.mean():.3f}")
        print(f"  HC LOCO adj_acc [n=4]: mean={hc_adj_mu:.4f} sd={hc_adj_sd:.4f} "
              f"| [n=6]: mean={hc_adj_mu6:.4f} sd={hc_adj_sd6:.4f} "
              f"| exact[n=4]={hc_exa_mu:.4f} (chance adj=0.375 exact=0.125)")

        roi_res = {
            'roi': roi, 'K': K, 'hc_n': int(len(hc_rho4)),
            'hc_loco_rho_n4_mean': float(hc_mu), 'hc_loco_rho_n4_sd': float(hc_sd),
            'hc_loco_rho_n4_values': [round(x, 4) for x in hc_rho4.tolist()],
            'hc_loco_rho_n6_mean': float(hc_mu6), 'hc_loco_rho_n6_sd': float(hc_sd6),
            # LOCO interpolation accuracy (adjacent ±1 step; chance 3/8=0.375)
            'hc_loco_adjacc_n4_mean': float(hc_adj_mu), 'hc_loco_adjacc_n4_sd': float(hc_adj_sd),
            'hc_loco_adjacc_n4_values': [round(x, 4) for x in hc_adj4.tolist()],
            'hc_loco_adjacc_n6_mean': float(hc_adj_mu6), 'hc_loco_adjacc_n6_sd': float(hc_adj_sd6),
            'hc_loco_exactacc_n4_mean': float(hc_exa_mu), 'hc_loco_exactacc_n4_sd': float(hc_exa_sd),
            'hc_loco_exactacc_n6_mean': float(hc_exa_mu6),
            'hc_loro_acc_mean': float(hc_acc.mean()), 'hc_loro_acc_sd': float(hc_acc.std(ddof=1)),
            'hc_loro_fe6_acc_mean': float(hc_acc_fe6.mean()),
            'hc_loro_fe6_acc_sd': float(hc_acc_fe6.std(ddof=1)),
            'conditions': {},
        }

        # ---- no-filter baseline = this subject's exp1 (canonical maskfunc recipe) ----
        # Answers "more HC-like THAN WHAT" + sanity-checks loco code on known data.
        nf_amp = load_npy(HC_C010 / subj / roi / "amplitudes_procrustes.npy")
        nf_rho4 = np.nan
        nf_adj4 = np.nan
        if nf_amp is not None:
            nf_rho6 = float(loco_rho_within(nf_amp, C_baseline).mean())
            nf_rho4 = loco_rho_subsampled(nf_amp, C_baseline, N_RUNS_EXP2)
            nf_d = cohens_d(nf_rho4, hc_mu, hc_sd)
            nf_adj_vec, nf_exa_vec = loco_accuracy_within(nf_amp, C_baseline)
            nf_adj6 = float(nf_adj_vec.mean()); nf_exa6 = float(nf_exa_vec.mean())
            nf_adj4, nf_exa4 = loco_accuracy_subsampled(nf_amp, C_baseline, N_RUNS_EXP2)
            nf_adj_d = cohens_d(nf_adj4, hc_adj_mu, hc_adj_sd)
            roi_res['nofilter_baseline_exp1'] = {
                'n_voxels': int(nf_amp.shape[2]),
                'loco_rho_n6_full': nf_rho6, 'loco_rho_n4_matched': nf_rho4,
                'loco_d_vs_hc_n4': nf_d,
                'loco_adjacc_n6_full': nf_adj6, 'loco_adjacc_n4_matched': nf_adj4,
                'loco_exactacc_n6_full': nf_exa6, 'loco_exactacc_n4_matched': nf_exa4,
                'loco_adjacc_d_vs_hc_n4': nf_adj_d,
                'loco_adjacc_per_color': [round(x, 4) for x in nf_adj_vec.tolist()],
                'loro_acc': float(loro_accuracy(nf_amp)),
                'loro_fe6_acc': float(loro_accuracy_fe6(nf_amp, C_baseline)),
                'note': f'{subj} exp1; maskfunc recipe (recipe-matched to HC; voxel set differs from exp2 native)',
            }
            print(f"  [nofilter-exp1] LOCO rho n4={nf_rho4:.4f} n6={nf_rho6:.4f} "
                  f"(d_vs_HCn4={nf_d:+.2f}) | adj_acc n4={nf_adj4:.4f} n6={nf_adj6:.4f} "
                  f"(d_vs_HCn4={nf_adj_d:+.2f}) V={nf_amp.shape[2]}")

        # ---- exp2 conditions ----
        cond_rho = {}
        cond_adj = {}
        for cond in CONDITIONS:
            amp = load_npy(EXP2_C010 / subj / cond / roi / "amplitudes_procrustes.npy")
            if amp is None:
                print(f"  [exp2] {cond} {roi} MISSING")
                continue
            rho_vec = loco_rho_within(amp, C_baseline)
            rho_m = float(rho_vec.mean())
            adj_vec, exa_vec = loco_accuracy_within(amp, C_baseline)
            adj_m = float(adj_vec.mean()); exa_m = float(exa_vec.mean())
            acc = float(loro_accuracy(amp))
            acc_fe6 = float(loro_accuracy_fe6(amp, C_baseline))
            rdm = corr_rdm(amp.mean(axis=0))
            d_vs_hc = cohens_d(rho_m, hc_mu, hc_sd)              # vs run-matched HC
            adj_d_vs_hc = cohens_d(adj_m, hc_adj_mu, hc_adj_sd)  # vs run-matched HC (adj_acc)
            stab = loco_rho_loro_stability(amp, C_baseline)     # 4 leave-one-run-out estimates
            adj_stab = loco_accuracy_loro_stability(amp, C_baseline)  # 4 LOO-run adj_acc
            cond_rho[cond] = rho_m
            cond_adj[cond] = adj_m
            roi_res['conditions'][cond] = {
                'n_voxels': int(amp.shape[2]), 'n_runs': int(amp.shape[0]),
                'loco_rho_mean': rho_m,
                'loco_rho_per_color': [round(x, 4) for x in rho_vec.tolist()],
                'loco_d_vs_hc_n4matched': d_vs_hc,
                'loco_rho_loo_runs': [round(x, 4) for x in stab],
                'loco_rho_loo_min': float(min(stab)), 'loco_rho_loo_max': float(max(stab)),
                # --- LOCO interpolation accuracy (adjacent ±1 step; chance 3/8) ---
                'loco_adjacc_mean': adj_m,
                'loco_exactacc_mean': exa_m,
                'loco_adjacc_per_color': [round(x, 4) for x in adj_vec.tolist()],
                'loco_exactacc_per_color': [round(x, 4) for x in exa_vec.tolist()],
                'loco_adjacc_d_vs_hc_n4matched': adj_d_vs_hc,
                'loco_adjacc_loo_runs': [round(x, 4) for x in adj_stab],
                'loco_adjacc_loo_min': float(min(adj_stab)),
                'loco_adjacc_loo_max': float(max(adj_stab)),
                'loro_acc': acc,
                'loro_fe6_acc': acc_fe6,
                'rdm_cosine_to_hc': cosine(rdm, hc_rdm_mean),
            }
            print(f"  [{cond:7s}] LOCO rho={rho_m:.4f} (d_vs_HCn4={d_vs_hc:+.2f}) "
                  f"adj_acc={adj_m:.4f} (d_vs_HCn4={adj_d_vs_hc:+.2f}) "
                  f"LOO-run range=[{min(stab):+.3f},{max(stab):+.3f}] "
                  f"LORO acc={acc:.3f} RDM~HC cos={cosine(rdm, hc_rdm_mean):.3f} V={amp.shape[2]}")

        # ---- primary contrast: Optimal vs Window ----
        if 'window' in cond_rho and 'optimal' in cond_rho:
            d_rho = cond_rho['optimal'] - cond_rho['window']
            d_contrast = d_rho / hc_sd if hc_sd > 1e-12 else np.nan
            wopt = roi_res['conditions']['optimal']; wwin = roi_res['conditions']['window']
            loo_separated = bool(wopt['loco_rho_loo_min'] > wwin['loco_rho_loo_max'])
            # --- adjacent-accuracy contrast (interpolation readout) ---
            d_adj = cond_adj['optimal'] - cond_adj['window']
            d_adj_contrast = d_adj / hc_adj_sd if hc_adj_sd > 1e-12 else np.nan
            adj_loo_separated = bool(
                wopt['loco_adjacc_loo_min'] > wwin['loco_adjacc_loo_max'])
            roi_res['contrast_optimal_minus_window'] = {
                'delta_loco_rho': float(d_rho),
                'delta_in_hc_n4_sd_units': float(d_contrast),
                'optimal_closer_to_hc': bool(
                    abs(cond_rho['optimal'] - hc_mu) < abs(cond_rho['window'] - hc_mu)),
                'loo_run_separated': loo_separated,  # opt min > win max across all single-run drops
                # --- adjacent accuracy (interpolation) ---
                'delta_loco_adjacc': float(d_adj),
                'delta_adjacc_in_hc_n4_sd_units': float(d_adj_contrast),
                'optimal_closer_to_hc_adjacc': bool(
                    abs(cond_adj['optimal'] - hc_adj_mu) < abs(cond_adj['window'] - hc_adj_mu)),
                'adjacc_loo_run_separated': adj_loo_separated,
            }
            # vs no-filter baseline — answers "more HC-like than NO filter?"
            if np.isfinite(nf_rho4):
                vs_nf = {
                    'nofilter_rho_n4': float(nf_rho4),
                    'optimal_more_hc_like_than_nofilter': bool(
                        abs(cond_rho['optimal'] - hc_mu) < abs(nf_rho4 - hc_mu)),
                    'window_more_hc_like_than_nofilter': bool(
                        abs(cond_rho['window'] - hc_mu) < abs(nf_rho4 - hc_mu)),
                    'optimal_minus_nofilter': float(cond_rho['optimal'] - nf_rho4),
                    'window_minus_nofilter': float(cond_rho['window'] - nf_rho4),
                }
                if np.isfinite(nf_adj4):
                    vs_nf.update({
                        'nofilter_adjacc_n4': float(nf_adj4),
                        'optimal_more_hc_like_than_nofilter_adjacc': bool(
                            abs(cond_adj['optimal'] - hc_adj_mu) < abs(nf_adj4 - hc_adj_mu)),
                        'window_more_hc_like_than_nofilter_adjacc': bool(
                            abs(cond_adj['window'] - hc_adj_mu) < abs(nf_adj4 - hc_adj_mu)),
                        'optimal_adjacc_minus_nofilter': float(cond_adj['optimal'] - nf_adj4),
                        'window_adjacc_minus_nofilter': float(cond_adj['window'] - nf_adj4),
                    })
                roi_res['contrast_optimal_minus_window']['vs_nofilter'] = vs_nf
            print(f"  >> Optimal-Window LOCO rho delta={d_rho:+.4f} "
                  f"({d_contrast:+.2f} HCn4-SD); closer_to_HC="
                  f"{roi_res['contrast_optimal_minus_window']['optimal_closer_to_hc']} "
                  f"LOO-separated={loo_separated}")
            print(f"  >> Optimal-Window adj_acc delta={d_adj:+.4f} "
                  f"({d_adj_contrast:+.2f} HCn4-SD); closer_to_HC="
                  f"{roi_res['contrast_optimal_minus_window']['optimal_closer_to_hc_adjacc']} "
                  f"LOO-separated={adj_loo_separated}")

        results[roi] = roi_res

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / out_name, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR / out_name}")

    # ---- compact summary ----
    # NOTE (paper alignment): forward-tuning LOCO rho is the SECONDARY/corroboration
    # index (methods_v2 L265, results_v4 L191). The PRIMARY endpoint is hV4 LOCO
    # adjacent accuracy (printed in the next table). Read that table as the headline.
    print(f"\n{'='*72}\nSUMMARY ({subj}, descriptive) — forward-tuning LOCO rho (CORROBORATION)\n"
          f"  [PRIMARY endpoint = hV4 LOCO adjacent accuracy, see next table]\n{'='*72}")
    print(f"{'ROI':<5}{'HCn4':>7}{'NoFilt':>8}{'Win':>8}{'Opt':>8}"
          f"{'NF_d':>7}{'Win_d':>7}{'Opt_d':>7}{'Opt>NF':>8}{'LOOsep':>7}")
    for roi, r in results.items():
        c = r.get('conditions', {})
        w = c.get('window', {}); o = c.get('optimal', {})
        ct = r.get('contrast_optimal_minus_window', {})
        nf = r.get('nofilter_baseline_exp1', {})
        vnf = ct.get('vs_nofilter', {})
        print(f"{roi:<5}{r['hc_loco_rho_n4_mean']:>7.3f}"
              f"{nf.get('loco_rho_n4_matched', float('nan')):>8.3f}"
              f"{w.get('loco_rho_mean', float('nan')):>8.3f}"
              f"{o.get('loco_rho_mean', float('nan')):>8.3f}"
              f"{nf.get('loco_d_vs_hc_n4', float('nan')):>7.2f}"
              f"{w.get('loco_d_vs_hc_n4matched', float('nan')):>7.2f}"
              f"{o.get('loco_d_vs_hc_n4matched', float('nan')):>7.2f}"
              f"{str(vnf.get('optimal_more_hc_like_than_nofilter', '')):>8}"
              f"{str(ct.get('loo_run_separated', '')):>7}")
    print(f"\nNoFilt = {subj} exp1 no-filter baseline (maskfunc); "
          "Win/Opt exp2 (masknone n coverage). 'Opt>NF' = Optimal closer to HC than no-filter.")

    # ---- LOCO adjacent accuracy (interpolation; Phase-1 readout) ----
    # *** PRIMARY ENDPOINT *** (paper: hV4 is the validated ROI; chance = 3/8). ***
    print(f"\n{'='*72}\nLOCO adjacent accuracy (interpolation; chance 3/8) — *** PRIMARY ENDPOINT (hV4) ***\n{'='*72}")
    print(f"{'ROI':<5}{'HCn4':>7}{'NoFilt':>8}{'Win':>8}{'Opt':>8}"
          f"{'Opt_d':>7}{'Win_d':>7}{'Opt>NF':>8}{'LOOsep':>7}")
    for roi, r in results.items():
        c = r.get('conditions', {})
        w = c.get('window', {}); o = c.get('optimal', {})
        ct = r.get('contrast_optimal_minus_window', {})
        nf = r.get('nofilter_baseline_exp1', {})
        vnf = ct.get('vs_nofilter', {})
        print(f"{roi:<5}{r.get('hc_loco_adjacc_n4_mean', float('nan')):>7.3f}"
              f"{nf.get('loco_adjacc_n4_matched', float('nan')):>8.3f}"
              f"{w.get('loco_adjacc_mean', float('nan')):>8.3f}"
              f"{o.get('loco_adjacc_mean', float('nan')):>8.3f}"
              f"{o.get('loco_adjacc_d_vs_hc_n4matched', float('nan')):>7.2f}"
              f"{w.get('loco_adjacc_d_vs_hc_n4matched', float('nan')):>7.2f}"
              f"{str(vnf.get('optimal_more_hc_like_than_nofilter_adjacc', '')):>8}"
              f"{str(ct.get('adjacc_loo_run_separated', '')):>7}")
    print("\nadj_acc = proportion of LOCO predictions within +/-1 hue step (45 deg) of "
          "the held-out hue, per-run decode then averaged within color (Phase-1 "
          "interpolation readout; chance 3/8=0.375, exact 1/8=0.125). "
          "'Opt>NF' = Optimal closer to HC than no-filter (adj_acc).")

    # ---- FE-6 LORO color-decoding accuracy (paper-matched decoder) ----
    print(f"\n{'='*72}\nFE-6 LORO color-decoding accuracy  (old corr-template in parens)\n{'='*72}")
    print(f"{'ROI':<5}{'HC(FE6)':>10}{'NoFilt':>10}{'Window':>10}{'Optimal':>10}")
    for roi, r in results.items():
        c = r.get('conditions', {})
        w = c.get('window', {}); o = c.get('optimal', {})
        nf = r.get('nofilter_baseline_exp1', {})

        def cell(fe6, old):
            if fe6 is None or (isinstance(fe6, float) and not np.isfinite(fe6)):
                return f"{'--':>10}"
            return f"{fe6:>6.3f}({old:.3f})" if (old is not None) else f"{fe6:>10.3f}"

        print(f"{roi:<5}"
              f"{r.get('hc_loro_fe6_acc_mean', float('nan')):>6.3f}"
              f"({r.get('hc_loro_acc_mean', float('nan')):.3f})"
              f"{cell(nf.get('loro_fe6_acc'), nf.get('loro_acc'))}"
              f"{cell(w.get('loro_fe6_acc'), w.get('loro_acc'))}"
              f"{cell(o.get('loro_fe6_acc'), o.get('loro_acc'))}")
    print("FE-6 LORO = pooled-W ridge_gcv forward encoder + correlation template matching "
          "(paper-matched, both LOCO & LORO FE-6). Value = FE-6 acc; (..) = old empirical-mean corr-template acc.")


if __name__ == "__main__":
    main()
