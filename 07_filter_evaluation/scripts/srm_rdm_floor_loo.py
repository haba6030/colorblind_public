#!/usr/bin/env python3
"""Leakage-free SRM-RDM HC self-consistency floor.

The ResearchNOTE §6.2.1 floor (0.663/0.500 for V1/V2) comes from
exp2_convergent.py:srm_disparities -> rdm_paper['_hc']['spearman_self_loo_mean'],
which fits SRM on ALL 7 HC (including the held-out subject) and aligns each HC via
its learned w_. CVD conditions, by contrast, are OUT of the fit and projected as
NEW subjects (_align_cond / project_new_subject). So the floor is not measured the
same way the conditions are -> upward bias.

This script computes, per ROI:
  (1) LEAKY floor  -- exact reproduction of the current code (validation).
  (2) CLEAN floor  -- for each HC i: fit SRM on the OTHER 6, project i as a NEW
      subject (identical to CVD-condition treatment), compare its corr-dist RDM to
      the HC-mean RDM of the 6 in-fit subjects. This is on the SAME scale as the
      condition spearman_to_hc values.
Also prints the model-free raw-voxel floor for reference.

Run:  mpirun -np 1 python srm_rdm_floor_loo.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/srm_rdm_floor_loo.py  (development repository, commit 53c81c2)
# Role      : Control leave-one-out self-consistency for RDM similarity.
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

import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from brainiak.funcalign.srm import SRM

LOCAL_C010 = Path("/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/"
                  "colorBlind_analysis/analysis/phase1_procrustes_decoding/results/"
                  "visualization/full_dataset_C010_with_residuals")
HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROIS = ['V1', 'V2', 'V3', 'V4']
K_SRM = {'V1': 4, 'V2': 4, 'V3': 3, 'V4': 3}
_IU = np.triu_indices(8, k=1)


def load_mean(s, roi):
    p = LOCAL_C010 / s / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    amp = np.load(p)                       # (6,8,V)
    if roi == 'V4' and amp.shape[2] < 20:  # sub-07 hV4 ~16 voxels -> drop (matches convergent)
        return None
    return amp.mean(0)                      # (8,V)


def project_new_subject(srm_model, new_data):        # new_data: (V,8)
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def align_cond(srm, mean_8xV):                        # project as NEW subject -> (8,k)
    return (project_new_subject(srm, mean_8xV.T).T @ mean_8xV.T).T


def corr_dist_rdm(pattern_8xk):
    return (1.0 - np.corrcoef(pattern_8xk))[_IU]


def fit_srm(means, k):
    srm = SRM(n_iter=10, features=k, rand_seed=0)
    srm.fit([m.T for m in means])
    return srm


def leaky_floor(means, k):
    """Exact reproduction of convergent code: SRM on all, align via w_, LOO-mean RDM."""
    srm = fit_srm(means, k)
    aligned = [(w.T @ m.T).T for w, m in zip(srm.w_, means)]
    rdms = [corr_dist_rdm(a) for a in aligned]
    n = len(rdms)
    self_r = [spearmanr(rdms[i], np.mean([rdms[j] for j in range(n) if j != i], axis=0))[0]
              for i in range(n)]
    return float(np.mean(self_r)), self_r


def clean_floor(means, k):
    """Held-out HC excluded from fit + projected as NEW subject (CVD-identical)."""
    n = len(means)
    rho = []
    for i in range(n):
        others = [means[j] for j in range(n) if j != i]
        srm = fit_srm(others, k)
        others_aligned = [(w.T @ m.T).T for w, m in zip(srm.w_, others)]
        hc_mean_rdm = np.mean([corr_dist_rdm(a) for a in others_aligned], axis=0)
        proj_i = align_cond(srm, means[i])            # project held-out i like a CVD condition
        rho.append(spearmanr(corr_dist_rdm(proj_i), hc_mean_rdm)[0])
    return float(np.mean(rho)), [float(x) for x in rho]


def voxel_floor(means):
    """Model-free: raw-voxel corr-dist RDM, pure LOO (no SRM)."""
    rdms = [corr_dist_rdm(m) for m in means]  # m is (8,V) -> corrcoef over 8 rows
    n = len(rdms)
    self_r = [spearmanr(rdms[i], np.mean([rdms[j] for j in range(n) if j != i], axis=0))[0]
              for i in range(n)]
    return float(np.mean(self_r))


def main():
    print(f"{'ROI':4} {'k':>2} {'n':>2} | {'LEAKY(cur)':>11} | {'CLEAN(LOO)':>11} | {'voxel-free':>10}")
    print("-" * 60)
    out = {}
    for roi in ROIS:
        means = [m for m in (load_mean(s, roi) for s in HC_SUBJS) if m is not None]
        k = K_SRM[roi]
        lk, lk_v = leaky_floor(means, k)
        cl, cl_v = clean_floor(means, k)
        vx = voxel_floor(means)
        out[roi] = {'leaky': lk, 'clean': cl, 'voxel': vx,
                    'clean_per_subject': cl_v, 'n': len(means)}
        print(f"{roi:4} {k:>2} {len(means):>2} | {lk:>+11.3f} | {cl:>+11.3f} | {vx:>+10.3f}")
    print("\nPer-subject CLEAN floor:")
    for roi in ROIS:
        print(f"  {roi}: " + " ".join(f"{x:+.2f}" for x in out[roi]['clean_per_subject']))
    import json
    outp = Path(__file__).resolve().parent.parent / "results" / "srm_rdm_floor_loo.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"\nSAVED {outp}")


if __name__ == "__main__":
    main()
