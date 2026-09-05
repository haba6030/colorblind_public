#!/usr/bin/env python3
"""
exp2 convergent metrics (secondary to LOCO ρ) — does Optimal converge to HC more
than Window / no-filter on RDM geometry and SRM alignment?

1) v6 PCA-RDM similarity to HC  (s10b_v6 recipe: center-over-colors -> SVD top-6
   -> correlation RDM -> compare condition RDM to HC-mean RDM in PC space).
   Higher Pearson/cosine = more HC-like representational geometry.
2) SRM Procrustes disparity into HC-only shared space (canonical
   rerun_loo_consistent recipe, K={V1:4,V2:4,V3:3,V4:3}, LOO-consistent refs).
   Each condition treated like a CVD subject. LOWER disparity = more HC-like.

Project's framework: SRM/RDM = convergent EXISTENCE evidence, NOT fitting criterion
(LOCO ρ is primary). Descriptive, sub-08 only.

Run with: mpirun -np 1 python exp2_convergent.py --variant {native|matched}
(brainiak SRM needs MPI init; bare python forbidden per project CLAUDE.md §9.)
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp2_convergent.py  (development repository, commit 53c81c2)
# Role      : SRM disparity and shared-space RDM similarity per condition (BrainIAK) -> exp2_convergent_sub-*_matched.json
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
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import squareform, pdist
from scipy.stats import pearsonr, spearmanr, t as t_dist

ROOT = Path("/scratch/connectome/haba6030/colorBlind")
HC_C010 = ROOT / "derivatives" / "full_dataset_C010"
OUT_DIR = ROOT / "analysis" / "phase6_behavioral_analysis" / "exp2_neural" / "results"

HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROIS = ['V1', 'V2', 'V3', 'V4']
K_SRM = {'V1': 4, 'V2': 4, 'V3': 3, 'V4': 3}
K_PCA = 6
CONDITIONS = ['nofilter', 'window', 'optimal']
TRIU = np.triu_indices(8, k=1)


def load_npy(p):
    return np.load(p) if p.exists() else None


# ---------------- v6 PCA-RDM ----------------
def voxel_pca_components(pattern_8xV, k=K_PCA):
    mp = pattern_8xV - pattern_8xV.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(mp, full_matrices=False)
    ke = min(k, U.shape[1])
    return U[:, :ke] * S[:ke]


def pca_rdm_vec(pattern_8xV):
    scores = voxel_pca_components(pattern_8xV)
    n = 8
    out = np.zeros((n * (n - 1)) // 2); idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            a, b = scores[i], scores[j]
            am, bm = a - a.mean(), b - b.mean()
            den = np.linalg.norm(am) * np.linalg.norm(bm)
            out[idx] = 1.0 - (am @ bm / den if den > 1e-9 else 0.0); idx += 1
    return out


# ---------------- Euclidean / Crossnobis RDM (canonical recipe) ----------------
def euclidean_rdm_vec(mean_8xV):
    return pdist(mean_8xV, metric='euclidean')   # (28,)


def estimate_noise_cov(amp):
    """Pooled per-color run residual covariance, shrinkage 0.1 (canonical)."""
    n_runs, n_colors, V = amp.shape
    res = np.vstack([amp[:, c, :] - amp[:, c, :].mean(0, keepdims=True) for c in range(n_colors)])
    sigma = (res.T @ res) / (res.shape[0] - 1)
    tm = np.trace(sigma) / V
    return 0.9 * sigma + 0.1 * tm * np.eye(V)


def crossnobis_rdm_vec(amp):
    """Cross-validated Mahalanobis RDM over run pairs (canonical). (28,), can be <0."""
    n_runs, n_colors, V = amp.shape
    sigma = estimate_noise_cov(amp)
    try:
        L = np.linalg.cholesky(sigma); L_inv = np.linalg.solve(L, np.eye(V))
    except np.linalg.LinAlgError:
        ev, evec = np.linalg.eigh(sigma); keep = ev > ev.max() * 1e-6
        L_inv = evec[:, keep] @ np.diag(1.0 / np.sqrt(ev[keep]))
    rdm = np.zeros((n_colors, n_colors)); npair = 0
    for a in range(n_runs):
        for b in range(a + 1, n_runs):
            pa, pb = amp[a] @ L_inv, amp[b] @ L_inv
            for i in range(n_colors):
                for j in range(i + 1, n_colors):
                    v = np.dot(pa[i] - pa[j], pb[i] - pb[j])
                    rdm[i, j] += v
            npair += 1
    rdm /= npair
    return rdm[np.triu_indices(n_colors, k=1)]


def rdm_similarity_to_hc(cond_rdm, hc_mean_rdm):
    r = float(pearsonr(cond_rdm, hc_mean_rdm)[0])
    cos = float(np.dot(cond_rdm, hc_mean_rdm) /
                (np.linalg.norm(cond_rdm) * np.linalg.norm(hc_mean_rdm) + 1e-12))
    return r, cos


# ---------------- SRM (canonical) ----------------
def project_new_subject(srm_model, new_data):
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def procrustes_disparity(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Xn = Xc / np.linalg.norm(Xc, 'fro'); Yn = Yc / np.linalg.norm(Yc, 'fro')
    R, _ = orthogonal_procrustes(Xn, Yn)
    return float(np.linalg.norm(Xn @ R - Yn, 'fro'))


def crawford_howell(x, ctrl):
    n = len(ctrl); m = np.mean(ctrl); sd = np.std(ctrl, ddof=1)
    if sd == 0:
        return float('inf'), 0.0
    t = (x - m) / (sd * np.sqrt((n + 1) / n))
    return float(t), float(t_dist.cdf(t, n - 1))  # one-tailed P(disparity below ctrl); low disp -> small t


def _align_cond(srm, mean_8xV):
    return (project_new_subject(srm, mean_8xV.T).T @ mean_8xV.T).T   # (8,k)


def _loo_disp(aligned, hc_aligned):
    """Mean LOO-consistent disparity of `aligned` (8,k) vs each HC LOO ref."""
    hc_arr = np.array(hc_aligned); n = len(hc_aligned)
    return float(np.mean([procrustes_disparity(aligned, np.delete(hc_arr, i, axis=0).mean(0))
                          for i in range(n)]))


def corr_dist_rdm(pattern_8xk):
    """Paper RDM: 8x8 correlation distance (1 - Pearson r), upper triangle (28,)."""
    rdm = 1.0 - np.corrcoef(pattern_8xk)
    return rdm[np.triu_indices(8, k=1)]


def srm_disparities(hc_means, cond_means, k, nofilter_raw=None):
    """hc_means: list of (8,V). cond_means: dict name->(8,V). LOO-consistent.
    nofilter_raw: (6,8,V) -> also compute run-matched (n=4) no-filter disparity.

    Returns (hc_disp, cond_scores, rdm_paper). rdm_paper = PAPER-consistent RDM
    computed IN SRM-ALIGNED SPACE (corr-distance), per methods_v2 sec RDM:
      - mean pairwise disparity (avg of 28 upper-tri) vs HC LOO (Cohen's d, Crawford-Howell)
      - Spearman rho of condition RDM vs HC-mean RDM (global similarity)."""
    from brainiak.funcalign.srm import SRM
    srm = SRM(n_iter=10, features=k, rand_seed=0)
    srm.fit([m.T for m in hc_means])             # each (V,8)
    hc_aligned = [(w.T @ m.T).T for w, m in zip(srm.w_, hc_means)]   # (8,k)
    n_hc = len(hc_aligned); hc_arr = np.array(hc_aligned)
    hc_disp = np.array([procrustes_disparity(hc_aligned[i], np.delete(hc_arr, i, axis=0).mean(0))
                        for i in range(n_hc)])
    cond_aligned = {name: _align_cond(srm, m) for name, m in cond_means.items()}
    cond_scores = {name: _loo_disp(cond_aligned[name], hc_aligned) for name in cond_means}
    if nofilter_raw is not None and nofilter_raw.shape[0] > 4:
        subs = [_loo_disp(_align_cond(srm, nofilter_raw[list(idx)].mean(0)), hc_aligned)
                for idx in itertools.combinations(range(nofilter_raw.shape[0]), 4)]
        cond_scores['nofilter_n4'] = float(np.mean(subs))

    # ---- PAPER RDM in SRM-aligned space ----
    hc_rdms = [corr_dist_rdm(a) for a in hc_aligned]            # each (28,)
    hc_mean_rdm = np.mean(hc_rdms, axis=0)
    hc_pair_disp = np.array([float(np.mean(r)) for r in hc_rdms])  # mean pairwise disparity per HC
    mu, sd = hc_pair_disp.mean(), hc_pair_disp.std(ddof=1)
    # HC self-consistency: each HC RDM vs LOO-mean of others (Spearman noise floor)
    hc_self_rho = [float(spearmanr(hc_rdms[i], np.mean([hc_rdms[j] for j in range(n_hc) if j != i], axis=0))[0])
                   for i in range(n_hc)]
    rdm_paper = {'_hc': {'pair_disp_mean': float(mu), 'pair_disp_sd': float(sd),
                         'spearman_self_loo_mean': float(np.mean(hc_self_rho))}}
    for name, a in cond_aligned.items():
        rv = corr_dist_rdm(a)
        pd = float(np.mean(rv))
        d = float((pd - mu) / sd) if sd > 1e-12 else float('nan')  # +d = MORE dispersed = less HC-like
        rho = float(spearmanr(rv, hc_mean_rdm)[0])                 # global RDM similarity (paper metric)
        rdm_paper[name] = {'mean_pairwise_disparity': pd, 'disp_d_vs_hc': d, 'spearman_to_hc': rho}
    return hc_disp, cond_scores, rdm_paper


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='native', choices=['native', 'matched'])
    ap.add_argument('--subject', default='08', help='CVD subject id, e.g. 08 / 09')
    args = ap.parse_args()
    subj = f"sub-{args.subject}"
    exp2_dir = ROOT / "derivatives" / (
        "full_dataset_C010_exp2" if args.variant == 'native'
        else "full_dataset_C010_exp2_matched")
    print(f"SUBJECT={subj} VARIANT={args.variant} exp2_dir={exp2_dir}")

    results = {}
    for roi in ROIS:
        print(f"\n{'='*72}\nROI {roi}\n{'='*72}")
        # HC raw (n_runs,8,V) + means (8,V)
        hc_raws, hc_means = [], []
        for s in HC_SUBJS:
            amp = load_npy(HC_C010 / s / roi / "amplitudes_procrustes.npy")
            if amp is None:
                continue
            if roi == 'V4' and amp.shape[2] < 20:
                continue
            hc_raws.append(amp); hc_means.append(amp.mean(0))
        # condition raw + means: nofilter = this subject's exp1; window/optimal = exp2 variant
        cond_means, cond_raws = {}, {}
        nf_raw = load_npy(HC_C010 / subj / roi / "amplitudes_procrustes.npy")  # (6,8,V)
        if nf_raw is not None:
            cond_means['nofilter'] = nf_raw.mean(0); cond_raws['nofilter'] = nf_raw
        for c in ['window', 'optimal']:
            a = load_npy(exp2_dir / subj / c / roi / "amplitudes_procrustes.npy")
            if a is not None:
                cond_means[c] = a.mean(0); cond_raws[c] = a

        roi_res = {'roi': roi}

        # --- PCA-RDM similarity to HC ---
        hc_pca = np.mean([pca_rdm_vec(m) for m in hc_means], axis=0)
        roi_res['pca_rdm'] = {}
        for name, m in cond_means.items():
            v = pca_rdm_vec(m)
            r = float(pearsonr(v, hc_pca)[0])
            cosv = float(np.dot(v, hc_pca) / (np.linalg.norm(v) * np.linalg.norm(hc_pca) + 1e-12))
            roi_res['pca_rdm'][name] = {'pearson_to_hc': r, 'cosine_to_hc': cosv}
        # HC self consistency (LOO mean) for reference
        hc_pca_self = []
        for i in range(len(hc_means)):
            ref = np.mean([pca_rdm_vec(hc_means[j]) for j in range(len(hc_means)) if j != i], axis=0)
            hc_pca_self.append(float(pearsonr(pca_rdm_vec(hc_means[i]), ref)[0]))
        roi_res['pca_rdm']['hc_self_pearson_mean'] = float(np.mean(hc_pca_self))
        print("  PCA-RDM pearson-to-HC: " + ", ".join(
            f"{n}={roi_res['pca_rdm'][n]['pearson_to_hc']:+.3f}" for n in cond_means)
            + f"  (HC self {np.mean(hc_pca_self):+.3f})")

        # --- Euclidean + Crossnobis RDM similarity to HC ---
        for label, fn, use_raw in [('euclidean_rdm', euclidean_rdm_vec, False),
                                   ('crossnobis_rdm', crossnobis_rdm_vec, True)]:
            try:
                if use_raw:
                    hc_vecs = [fn(a) for a in hc_raws]
                    cond_vecs = {n: fn(a) for n, a in cond_raws.items()}
                else:
                    hc_vecs = [fn(m) for m in hc_means]
                    cond_vecs = {n: fn(m) for n, m in cond_means.items()}
                hc_mean_rdm = np.mean(hc_vecs, axis=0)
                roi_res[label] = {}
                for n, v in cond_vecs.items():
                    r, cos = rdm_similarity_to_hc(v, hc_mean_rdm)
                    roi_res[label][n] = {'pearson_to_hc': r, 'cosine_to_hc': cos}
                # HC self-consistency (LOO)
                self_r = []
                for i in range(len(hc_vecs)):
                    ref = np.mean([hc_vecs[j] for j in range(len(hc_vecs)) if j != i], axis=0)
                    self_r.append(float(pearsonr(hc_vecs[i], ref)[0]))
                roi_res[label]['hc_self_pearson_mean'] = float(np.mean(self_r))
                print(f"  {label} pearson-to-HC: " + ", ".join(
                    f"{n}={roi_res[label][n]['pearson_to_hc']:+.3f}" for n in cond_vecs)
                    + f"  (HC self {np.mean(self_r):+.3f})")
            except Exception as e:
                roi_res[label] = {'error': f"{type(e).__name__}: {e}"}
                print(f"  {label} ERROR: {type(e).__name__}: {e}")

        # --- SRM disparity + PAPER RDM (in SRM-aligned space) ---
        try:
            hc_disp, cond_scores, rdm_paper = srm_disparities(hc_means, cond_means, K_SRM[roi], nofilter_raw=nf_raw)
            roi_res['srm'] = {
                'k': K_SRM[roi],
                'hc_disp_mean': float(hc_disp.mean()), 'hc_disp_sd': float(hc_disp.std(ddof=1)),
                'conditions': {},
            }
            for name in cond_scores:
                t, p = crawford_howell(cond_scores[name], hc_disp)
                d = float((cond_scores[name] - hc_disp.mean()) / hc_disp.std(ddof=1))
                roi_res['srm']['conditions'][name] = {
                    'disparity': cond_scores[name], 'd_vs_hc': d, 'ch_p_below_hc': p}
            print(f"  SRM disp (lower=HC-like) HC={hc_disp.mean():.3f}±{hc_disp.std(ddof=1):.3f}: "
                  + ", ".join(f"{n}={cond_scores[n]:.3f}(d{roi_res['srm']['conditions'][n]['d_vs_hc']:+.2f})"
                              for n in cond_scores))
            # PAPER RDM (corr-distance in SRM space): mean pairwise disparity + Spearman-to-HC
            roi_res['srm_rdm_paper'] = rdm_paper
            hcr = rdm_paper['_hc']
            print(f"  PAPER-RDM(SRM-space) disp HC={hcr['pair_disp_mean']:.3f}±{hcr['pair_disp_sd']:.3f} "
                  f"(Spearman HC-self {hcr['spearman_self_loo_mean']:+.3f}); "
                  + ", ".join(f"{n}: disp={rdm_paper[n]['mean_pairwise_disparity']:.3f}(d{rdm_paper[n]['disp_d_vs_hc']:+.2f}) "
                              f"ρ_HC={rdm_paper[n]['spearman_to_hc']:+.2f}"
                              for n in cond_means))
        except Exception as e:
            import traceback
            roi_res['srm'] = {'error': f"{type(e).__name__}: {e}"}
            print(f"  SRM ERROR: {type(e).__name__}: {e}"); traceback.print_exc()

        results[roi] = roi_res

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"exp2_convergent_{subj}_{args.variant}.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")

    # summary
    print(f"\n{'='*78}\nCONVERGENT SUMMARY ({args.variant})\n"
          f"  PAPER metrics: SRMdisp & PAPER-RDM-disp lower=HC-like; PAPER-RDM Spearman-to-HC higher=HC-like\n"
          f"  (PCA/Eucl/Xnob-r = exploratory raw-voxel RDM variants, NOT the paper method)\n{'='*78}")
    print(f"{'ROI':<5}{'metric':<12}{'NoFilt':>9}{'Window':>9}{'Optimal':>9}{'HCself/HCdisp':>15}")
    for roi, r in results.items():
        s = r.get('srm', {})
        rp = r.get('srm_rdm_paper', {})
        # PAPER metrics first
        if 'conditions' in s:
            sc = s['conditions']
            nf4 = sc.get('nofilter_n4', {}).get('disparity', float('nan'))
            print(f"{roi:<5}{'SRMdisp*':<12}"
                  f"{sc.get('nofilter',{}).get('disparity',float('nan')):>9.3f}"
                  f"{sc.get('window',{}).get('disparity',float('nan')):>9.3f}"
                  f"{sc.get('optimal',{}).get('disparity',float('nan')):>9.3f}"
                  f"{s.get('hc_disp_mean',float('nan')):>15.3f}   nf_n4={nf4:.3f}")
        if rp:
            print(f"{'':<5}{'RDMdisp*':<12}"
                  f"{rp.get('nofilter',{}).get('mean_pairwise_disparity',float('nan')):>9.3f}"
                  f"{rp.get('window',{}).get('mean_pairwise_disparity',float('nan')):>9.3f}"
                  f"{rp.get('optimal',{}).get('mean_pairwise_disparity',float('nan')):>9.3f}"
                  f"{rp.get('_hc',{}).get('pair_disp_mean',float('nan')):>15.3f}")
            print(f"{'':<5}{'RDM-ρ*':<12}"
                  f"{rp.get('nofilter',{}).get('spearman_to_hc',float('nan')):>9.3f}"
                  f"{rp.get('window',{}).get('spearman_to_hc',float('nan')):>9.3f}"
                  f"{rp.get('optimal',{}).get('spearman_to_hc',float('nan')):>9.3f}"
                  f"{rp.get('_hc',{}).get('spearman_self_loo_mean',float('nan')):>15.3f}")
        # exploratory raw-voxel RDM variants
        for key, lab in [('pca_rdm', 'PCA-r'), ('euclidean_rdm', 'Eucl-r'), ('crossnobis_rdm', 'Xnob-r')]:
            p = r.get(key, {})
            print(f"{'':<5}{lab:<12}"
                  f"{p.get('nofilter',{}).get('pearson_to_hc',float('nan')):>9.3f}"
                  f"{p.get('window',{}).get('pearson_to_hc',float('nan')):>9.3f}"
                  f"{p.get('optimal',{}).get('pearson_to_hc',float('nan')):>9.3f}"
                  f"{p.get('hc_self_pearson_mean',float('nan')):>15.3f}")
    print("\n* = paper-consistent metric (SRM-aligned space). RDMdisp = mean pairwise corr-distance "
          "(lower=HC-like); RDM-ρ = Spearman of condition RDM vs HC-mean RDM (higher=HC-like).")


if __name__ == "__main__":
    main()
