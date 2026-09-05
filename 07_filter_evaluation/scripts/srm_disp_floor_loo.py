#!/usr/bin/env python3
"""Alignment-method test for SRM PROCRUSTES DISPARITY (Caveat 2).

Standard hc_disp (exp2_convergent.srm_disparities) aligns HC via LEARNED w_ (in-fit),
but conditions via project_new_subject (_align_cond, out-of-fit). Question: is the
condition disparity inflated purely by the weaker projection method?

Test: treat each held-out HC EXACTLY like a condition -- exclude it from the SRM fit,
project it via _align_cond (new-subject), and measure its disparity to the in-fit HC
references. Compare that "condition-method HC floor" to:
  - the standard in-fit hc_disp floor (all-7 SRM, w_-aligned)  [reproduces convergent]
  - the actual NF/Win/Opt condition disparities.
If condition-method floor ~ in-fit floor << conditions -> method is NOT the driver
(condition elevation is real). If it jumps toward the conditions -> method inflates.

Run:  mpirun -np 1 python srm_disp_floor_loo.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/srm_disp_floor_loo.py  (development repository, commit 53c81c2)
# Role      : Control leave-one-out floor for SRM disparity.
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
import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from brainiak.funcalign.srm import SRM

LOCAL_C010 = Path("/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/"
                  "colorBlind_analysis/analysis/phase1_procrustes_decoding/results/"
                  "visualization/full_dataset_C010_with_residuals")
RESULTS = Path(__file__).resolve().parent.parent / "results"
HC_SUBJS = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07']
ROIS = ['V1', 'V2', 'V3', 'V4']
K_SRM = {'V1': 4, 'V2': 4, 'V3': 3, 'V4': 3}


def load_mean(s, roi):
    p = LOCAL_C010 / s / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    amp = np.load(p)
    if roi == 'V4' and amp.shape[2] < 20:
        return None
    return amp.mean(0)


def procrustes_disparity(X, Y):                     # identical to convergent L117-121
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Xn = Xc / np.linalg.norm(Xc, 'fro'); Yn = Yc / np.linalg.norm(Yc, 'fro')
    R, _ = orthogonal_procrustes(Xn, Yn)
    return float(np.linalg.norm(Xn @ R - Yn, 'fro'))


def project_new_subject(srm_model, new_data):       # convergent L110-114
    S = srm_model.s_
    W_init = new_data @ np.linalg.pinv(S)
    U, _, Vt = np.linalg.svd(W_init, full_matrices=False)
    return U @ Vt


def align_cond(srm, mean_8xV):                       # convergent L132-133
    return (project_new_subject(srm, mean_8xV.T).T @ mean_8xV.T).T


def fit_srm(means, k):
    srm = SRM(n_iter=10, features=k, rand_seed=0)
    srm.fit([m.T for m in means])
    return srm


def infit_floor(means, k):
    """Standard hc_disp: all-N SRM, HC via w_, disparity vs LOO-mean of others."""
    srm = fit_srm(means, k)
    aligned = [(w.T @ m.T).T for w, m in zip(srm.w_, means)]
    arr = np.array(aligned); n = len(aligned)
    d = [procrustes_disparity(aligned[i], np.delete(arr, i, axis=0).mean(0)) for i in range(n)]
    return float(np.mean(d)), [round(x, 3) for x in d]


def condmethod_floor(means, k):
    """Held-out HC treated like a condition: SRM on others, project i as new subject,
    disparity vs LOO-mean of the in-fit references (same as _loo_disp for conditions)."""
    n = len(means)
    vals = []
    for i in range(n):
        others = [means[j] for j in range(n) if j != i]
        srm = fit_srm(others, k)
        ref_aligned = [(w.T @ m.T).T for w, m in zip(srm.w_, others)]
        arr = np.array(ref_aligned)
        proj_i = align_cond(srm, means[i])
        d = np.mean([procrustes_disparity(proj_i, np.delete(arr, j, axis=0).mean(0))
                     for j in range(len(ref_aligned))])
        vals.append(float(d))
    return float(np.mean(vals)), [round(x, 3) for x in vals]


def main():
    # actual condition disparities from convergent JSON (native)
    conv = {sid: json.load(open(RESULTS / f"exp2_convergent_sub-{sid}_native.json"))
            for sid in ['08', '09']}
    print(f"{'ROI':4} {'k':>2} | {'in-fit floor':>12} | {'cond-method floor':>17} | conditions (NF/Win/Opt)")
    print("-" * 96)
    out = {}
    for roi in ROIS:
        means = [m for m in (load_mean(s, roi) for s in HC_SUBJS) if m is not None]
        k = K_SRM[roi]
        inf, inf_v = infit_floor(means, k)
        cmf, cmf_v = condmethod_floor(means, k)
        out[roi] = {'infit_floor': inf, 'condmethod_floor': cmf,
                    'condmethod_per_subj': cmf_v, 'infit_per_subj': inf_v}
        conds = {}
        for sid in ['08', '09']:
            d = conv[sid][roi]['srm']['conditions']
            conds[sid] = f"s{sid}[{d['nofilter']['disparity']:.2f}/{d['window']['disparity']:.2f}/{d['optimal']['disparity']:.2f}]"
        print(f"{roi:4} {k:>2} | {inf:>12.3f} | {cmf:>17.3f} | {conds['08']} {conds['09']}")
    (RESULTS / "srm_disp_floor_loo.json").write_text(json.dumps(out, indent=1))
    print(f"\nper-subject cond-method floor:")
    for roi in ROIS:
        print(f"  {roi}: {out[roi]['condmethod_per_subj']}")
    print(f"\nSAVED {RESULTS / 'srm_disp_floor_loo.json'}")


if __name__ == "__main__":
    main()
