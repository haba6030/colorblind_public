#!/usr/bin/env python3
"""Run-count-matched SRM disparity and RDM for the exp2 filter evaluation.

WHY. The published geometry numbers compare 4-run filter conditions against a
6-run no-filter baseline and a 6-run HC reference. Disparity and RDM similarity
are both inflated by noise in EITHER pattern, so the filter conditions are
penalised twice: once against the baseline and once against the HC distribution
used for the single-case test. exp2_convergent.py already stores a run-matched
no-filter disparity (`nofilter_n4`) but the manuscript quotes the 6-run value,
and no run-matched variant exists at all for the RDM index.

WHAT. For each of the C(6,4)=15 four-run subsets S of the Session-1 runs:
  - every HC mean is rebuilt from runs S, and SRM is refit on those means
  - the no-filter condition is rebuilt from runs S
  - window / optimal are left untouched (they only ever had 4 runs)
  - HC leave-one-out disparities, condition disparities, and the paper RDM
    metrics are computed exactly as in exp2_convergent.srm_disparities
Metrics are then averaged over the 15 subsets. Subsets are enumerated
exhaustively, not sampled, so there is no seed dependence.

Reuses the frozen primitives from exp2_convergent.py (procrustes_disparity,
_align_cond, _loo_disp, corr_dist_rdm, crawford_howell) so the run-matched and
published numbers differ only in which runs enter.

Run with `mpirun -np 1 python` (BrainIAK requirement).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp2_runmatched_geometry.py  (development repository, commit 53c81c2)
# Role      : Run-matched geometry over all C(6,4) subsets -> exp2_runmatched_geometry_sub-*_matched.json (tab:exp2_geometry)
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

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCRIPTS = Path("/scratch/connectome/haba6030/colorBlind/analysis/"
               "phase6_behavioral_analysis/exp2_neural/scripts")
sys.path.insert(0, str(SCRIPTS))

from exp2_convergent import (            # noqa: E402
    HC_SUBJS, ROIS, K_SRM, HC_C010, ROOT, OUT_DIR,
    load_npy, procrustes_disparity, _align_cond, _loo_disp,
    corr_dist_rdm, crawford_howell,
)

N_RUNS_MATCH = 4


def srm_block(hc_means, cond_means, k):
    """One SRM fit -> (hc_disp[7], cond disparities, per-condition RDM metrics).

    Mirrors exp2_convergent.srm_disparities without the nofilter_n4 branch.
    """
    from brainiak.funcalign.srm import SRM
    srm = SRM(n_iter=10, features=k, rand_seed=0)
    srm.fit([m.T for m in hc_means])
    hc_aligned = [(w.T @ m.T).T for w, m in zip(srm.w_, hc_means)]
    hc_arr = np.array(hc_aligned)
    n_hc = len(hc_aligned)
    hc_disp = np.array([procrustes_disparity(hc_aligned[i],
                                             np.delete(hc_arr, i, axis=0).mean(0))
                        for i in range(n_hc)])
    cond_aligned = {n: _align_cond(srm, m) for n, m in cond_means.items()}
    cond_disp = {n: _loo_disp(cond_aligned[n], hc_aligned) for n in cond_means}

    hc_rdms = [corr_dist_rdm(a) for a in hc_aligned]
    hc_mean_rdm = np.mean(hc_rdms, axis=0)
    hc_pair = np.array([float(np.mean(r)) for r in hc_rdms])
    mu, sd = hc_pair.mean(), hc_pair.std(ddof=1)
    hc_self = [float(spearmanr(hc_rdms[i],
                               np.mean([hc_rdms[j] for j in range(n_hc) if j != i], axis=0))[0])
               for i in range(n_hc)]
    rdm = {"_hc": {"pair_disp_mean": float(mu), "pair_disp_sd": float(sd),
                   "spearman_self_loo_mean": float(np.mean(hc_self))}}
    for n, a in cond_aligned.items():
        rv = corr_dist_rdm(a)
        pd = float(np.mean(rv))
        rdm[n] = {"mean_pairwise_disparity": pd,
                  "disp_d_vs_hc": float((pd - mu) / sd) if sd > 1e-12 else float("nan"),
                  "spearman_to_hc": float(spearmanr(rv, hc_mean_rdm)[0])}
    return hc_disp, cond_disp, rdm


def run_subject(subj, variant="matched"):
    exp2_dir = ROOT / "derivatives" / (
        "full_dataset_C010_exp2" if variant == "native" else "full_dataset_C010_exp2_matched")
    out = {"subject": subj, "variant": variant, "n_runs_match": N_RUNS_MATCH,
           "design": "exhaustive C(6,4)=15 subsets; HC + nofilter subsampled, "
                     "window/optimal untouched (already 4 runs)", "rois": {}}

    for roi in ROIS:
        hc_raw = []
        for s in HC_SUBJS:
            a = load_npy(HC_C010 / s / roi / "amplitudes_procrustes.npy")
            if a is None:
                continue
            if roi == "V4" and a.shape[2] < 20:      # sub-07 hV4 = 16 voxels, as upstream
                continue
            hc_raw.append(a)
        nf_raw = load_npy(HC_C010 / subj / roi / "amplitudes_procrustes.npy")
        fixed = {}
        for c in ["window", "optimal"]:
            a = load_npy(exp2_dir / subj / c / roi / "amplitudes_procrustes.npy")
            if a is not None:
                fixed[c] = a.mean(0)
        if nf_raw is None or not fixed or not hc_raw:
            out["rois"][roi] = {"error": "missing input"}
            continue

        n_src = nf_raw.shape[0]
        subsets = list(itertools.combinations(range(n_src), N_RUNS_MATCH))
        acc_hc, acc_disp, acc_rdm = [], {}, {}
        for idx in subsets:
            idx = list(idx)
            hc_means = [a[idx].mean(0) for a in hc_raw]
            cond = dict(fixed)
            cond["nofilter"] = nf_raw[idx].mean(0)
            hc_disp, cond_disp, rdm = srm_block(hc_means, cond, K_SRM[roi])
            acc_hc.append(hc_disp)
            for n, v in cond_disp.items():
                acc_disp.setdefault(n, []).append(v)
            for n, m in rdm.items():
                for kk, vv in m.items():
                    acc_rdm.setdefault(n, {}).setdefault(kk, []).append(vv)

        hc_disp_mean_per_subset = np.array(acc_hc)          # (15, n_hc)
        hc_disp = hc_disp_mean_per_subset.mean(axis=0)      # per-HC, averaged over subsets
        res = {"k": K_SRM[roi], "n_subsets": len(subsets), "n_hc": len(hc_raw),
               "srm": {"hc_disp_mean": float(hc_disp.mean()),
                       "hc_disp_sd": float(hc_disp.std(ddof=1)),
                       "conditions": {}},
               "srm_rdm_paper": {}}
        for n, vals in acc_disp.items():
            v = float(np.mean(vals))
            t, p = crawford_howell(v, hc_disp)
            res["srm"]["conditions"][n] = {
                "disparity": v, "disparity_sd_over_subsets": float(np.std(vals, ddof=1)),
                "t": float(t), "p": float(p),
                "d": float((v - hc_disp.mean()) / hc_disp.std(ddof=1))}
        for n, m in acc_rdm.items():
            res["srm_rdm_paper"][n] = {kk: float(np.mean(vv)) for kk, vv in m.items()}
            res["srm_rdm_paper"][n].update(
                {f"{kk}_sd_over_subsets": float(np.std(vv, ddof=1)) for kk, vv in m.items()})
        out["rois"][roi] = res
        c = res["srm"]["conditions"]
        print(f"  {subj} {roi}: HC={hc_disp.mean():.4f}  "
              f"nofilter={c['nofilter']['disparity']:.4f}  "
              f"window={c['window']['disparity']:.4f}  "
              f"optimal={c['optimal']['disparity']:.4f}", flush=True)

    dest = OUT_DIR / f"exp2_runmatched_geometry_{subj}_{variant}.json"
    with open(dest, "w") as f:
        json.dump(out, f, indent=1)
    print(f"WROTE {dest}", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:] or ["sub-08", "sub-09"]:
        subj = s if s.startswith("sub-") else f"sub-{s}"
        print(f"=== {subj} ===", flush=True)
        run_subject(subj)
    print("DONE", flush=True)
