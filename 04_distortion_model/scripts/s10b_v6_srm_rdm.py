"""S10b v6 SRM-RDM variant.

Mirror of s10b_v6_pca_rdm.py with ONE change: the RDM atom is computed in
SRM-aligned shared space (BrainIAK SRM) instead of PCA-aligned space.

Everything else identical: γ atoms, LOCO V4 atom, 300 resamples, combo
enumeration, RC + 2-comp grids, train/test HC split, test_V1_RDM diagnostic
(unchanged — uses voxel L_RDM as in PCA v6, so comparison isolates the atom).

Run:
  python s10b_v6_srm_rdm.py --subject sub-08
  python s10b_v6_srm_rdm.py --subject sub-09
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/s10b_v6_srm_rdm.py  (development repository, commit 53c81c2)
# Role      : Same with the RDM atom in the SRM basis (requires BrainIAK).
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
import json
import sys
import time
from pathlib import Path

import numpy as np
from brainiak.funcalign.srm import SRM
from scipy.spatial.distance import squareform

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import s10b_v6_pca_rdm as v6  # noqa: E402


def _rdm_correlation_28(pattern_kx8: np.ndarray) -> np.ndarray:
    """8x8 correlation-distance RDM, returned as 28-d upper triangle."""
    n = pattern_kx8.shape[1]
    out = np.empty((n * (n - 1)) // 2)
    idx = 0
    for i in range(n):
        a = pattern_kx8[:, i]
        am = a - a.mean()
        na = np.linalg.norm(am)
        for j in range(i + 1, n):
            b = pattern_kx8[:, j]
            bm = b - b.mean()
            nb = np.linalg.norm(bm)
            denom = na * nb
            out[idx] = 1.0 - (am @ bm / denom if denom > 1e-9 else 0.0)
            idx += 1
    return out


def make_srm_rdm_atom(roi, cvd_amp, pool_amps_dict, C_baseline, K):
    """SRM-aligned RDM atom — drop-in replacement for v6 PCA-RDM atom.

    Pipeline per call:
      1. Mean-over-runs per HC subject -> (n_vox, 8) tensors
      2. BrainIAK SRM(K, n_iter=20, seed=0) trained on HC pool
      3. HC subjects projected to shared space -> per-subject (K, 8) -> RDM (28-d)
         -> HC mean 8x8 RDM
      4. CVD projection via fixed-S Procrustes (X S^T = U D V^T -> W = U V^T)
         -> CVD (K, 8) -> CVD RDM
      5. delta_rdm_obs = CVD RDM - HC mean RDM (28-d vector)
      6. loss_fn(delta_8vec) rotates HC mean RDM via perceived-color permutation
         (same as PCA v6) and returns 1 - cos(delta_obs, delta_sim).

    Returns None on training failure or pool < 2.
    """
    if len(pool_amps_dict) < 2:
        return None
    try:
        hc_data = []
        for sid, amp in pool_amps_dict.items():
            mean_pat = amp.mean(axis=0)  # (8, n_vox)
            hc_data.append(mean_pat.T)   # (n_vox, 8)
        srm = SRM(n_iter=20, features=int(K), rand_seed=0)
        srm.fit(hc_data)
        w_hc = srm.w_  # list of (n_vox_i, K)

        # HC subjects projected to shared space (K, 8) and 8x8 RDM
        hc_rdms_mat = []
        for x, w in zip(hc_data, w_hc):
            shared = w.T @ x  # (K, 8)
            rdm_vec = _rdm_correlation_28(shared)
            hc_rdms_mat.append(squareform(rdm_vec))
        hc_rdm_mean = np.mean(np.stack(hc_rdms_mat, axis=0), axis=0)  # (8, 8)
        triu = np.triu_indices(8, k=1)

        # CVD: Procrustes against fixed S
        s_shared = srm.s_  # (K, 8)
        x_cvd = cvd_amp.mean(axis=0).T  # (n_vox, 8)
        xst = x_cvd @ s_shared.T
        u_, _, vt_ = np.linalg.svd(xst, full_matrices=False)
        w_cvd = u_ @ vt_  # (n_vox, K)
        cvd_shared = w_cvd.T @ x_cvd  # (K, 8)
        cvd_rdm = squareform(_rdm_correlation_28(cvd_shared))
        delta_rdm_obs = (cvd_rdm - hc_rdm_mean)[triu]
        n_obs = float(np.linalg.norm(delta_rdm_obs))
    except Exception:
        return None

    HUES = v6.HUES

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
            cos_sim = float(np.dot(delta_rdm_sim, delta_rdm_obs) /
                             (n_sim * n_obs))
            return 1.0 - cos_sim
        except Exception:
            return np.nan

    return loss_fn


# Monkey-patch v6 so fit_subject(...) uses SRM-RDM atom
v6.make_rdm_atom = make_srm_rdm_atom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', required=True,
                        choices=['sub-08', 'sub-09'])
    parser.add_argument('--combo-start', type=int, default=None)
    parser.add_argument('--combo-end', type=int, default=None)
    args = parser.parse_args()

    print('=' * 100, flush=True)
    print('S10b v6 SRM-RDM (mirror of v6 PCA-RDM, RDM atom swap)', flush=True)
    print(f'  Subject: {args.subject}  chunk=[{args.combo_start}:{args.combo_end}]',
          flush=True)
    print(f'  N_resamples: {v6.N_RESAMPLES}', flush=True)
    print('=' * 100, flush=True)

    t0 = time.time()
    storage = v6.fit_subject(args.subject, args.combo_start, args.combo_end)
    summary = v6.summarize(storage)
    elapsed = round(time.time() - t0, 1)
    print(f'\n[{args.subject}] elapsed: {elapsed}s', flush=True)

    suffix = f'_{args.subject}'
    if args.combo_start is not None:
        suffix += f'_c{args.combo_start:02d}-{args.combo_end:02d}'
    out_file = v6.OUT_DIR / f's10b_v6_srm_rdm_results{suffix}.json'
    with open(out_file, 'w') as f:
        json.dump({
            'subject': args.subject, 'storage': storage,
            'summary': summary, 'elapsed': elapsed,
            'meta': {
                'N_resamples': v6.N_RESAMPLES,
                'subset_size': v6.SUBSET_SIZE,
                'seed_base': v6.RNG_SEED,
                'combo_range': [args.combo_start, args.combo_end],
                'rdm_method': 'SRM_BrainIAK (K=ROI_K, n_iter=20, seed=0)',
                'rdm_distance': 'correlation (1 - centered cosine of K-d shared cols)',
            },
        }, f, indent=2, default=str)
    print(f'Saved: {out_file}', flush=True)


if __name__ == '__main__':
    main()
