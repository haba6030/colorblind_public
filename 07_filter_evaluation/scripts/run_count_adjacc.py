#!/usr/bin/env python
"""
run_count_adjacc.py — Run-count subsampling validation for LOCO ADJACENT ACCURACY.

Gap addressed: the original run-count validation (run_count_saturation.py,
Tier 1/2/3) certified n=4 adequacy on the ENCODING-direction LOCO rho, LORO
accuracy, and split-half geometric-stability metrics. It never tracked the
DECODING adjacent-accuracy metric that the paper now reports as the primary
interpolation readout. This script closes that gap: it computes LOCO adjacent
accuracy (canonical FE-6 + OLS pseudoinverse decoder, loco_forward_readouts)
across every C(6,n) run subset for n in {2..6}, per subject per ROI, and reports
how the hV4 landmark finding (HC above 3/8 chance; CVD deutan/protan below)
retains as run count drops.

Output: run_count_validation/adjacc_saturation.json
        run_count_validation/adjacc_retention_summary.json
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/scripts/run_count_adjacc.py  (development repository, commit 53c81c2)
# Role      : Run-count adequacy of LOCO adjacent accuracy over all C(6,n) subsets (S14, Figure S2) -> adjacc_retention_summary.json
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
from itertools import combinations
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "analysis" / "phase4_forward_model" / "scripts"))

from loco_canonical import loco_forward_readouts  # noqa: E402
from utils_forward_model import (  # noqa: E402
    HC_SUBJECTS, CVD_SUBJECTS, N_CHANNELS,
    create_basis_matrix, HUE_ANGLES, load_amplitudes,
)

BASELINE_DIR = PROJECT_ROOT / "analysis" / "phase1_procrustes_decoding" / \
    "results" / "visualization" / "full_dataset_C010_with_residuals"
OUT_DIR = PROJECT_ROOT / "analysis" / "phase6_behavioral_analysis" / \
    "run_count_validation"

ROIS = ["V1", "V2", "V3", "V4"]
ALL_SUBJECTS = HC_SUBJECTS + CVD_SUBJECTS          # HC x7 + CVD x3
HC_HV4 = [s for s in HC_SUBJECTS if s != "sub-07"]  # sub-07 excluded at hV4 (low voxels)
N_VALUES = [2, 3, 4, 5, 6]
N_RUNS_TOTAL = 6
CHANCE = 3.0 / 8.0
N_JOBS = 8

# Canonical FE-6 basis (uniform), OLS decoder — same config as the main paper.
C8 = create_basis_matrix(HUE_ANGLES, N_CHANNELS, "fe")
BASIS_FULL = create_basis_matrix(np.arange(360), N_CHANNELS, "fe")


def adjacc_cell(amp_sub: np.ndarray) -> dict:
    r = loco_forward_readouts(amp_sub, C8, BASIS_FULL, decoder="ols", tasks=("adj",))
    adj = np.asarray(r["adj"], float)
    return {"adjacc": float(adj.mean()), "per_color": adj.tolist()}


def enumerate_subsets(n: int) -> list[tuple[int, ...]]:
    return [tuple(c) for c in combinations(range(N_RUNS_TOTAL), n)]


def crawford_howell_d(x: float, hc_vals: np.ndarray) -> tuple[float, float]:
    """Single-case effect size (Crawford-Howell d_cc) and one-tailed lower p."""
    from scipy import stats
    m, sd, n = hc_vals.mean(), hc_vals.std(ddof=1), len(hc_vals)
    if sd <= 0:
        return 0.0, 1.0
    t = (x - m) / (sd * np.sqrt((n + 1) / n))
    p = float(stats.t.cdf(t, df=n - 1))   # lower tail = deficit
    d = (x - m) / sd
    return float(d), p


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subsets_by_n = {n: enumerate_subsets(n) for n in N_VALUES}
    total = sum(len(v) for v in subsets_by_n.values())
    print(f"Subsets/n: {[(n, len(subsets_by_n[n])) for n in N_VALUES]} | total {total} "
          f"| subjects {len(ALL_SUBJECTS)} | ROIs {len(ROIS)}")

    results = {"config": {"rois": ROIS, "subjects": ALL_SUBJECTS, "n_values": N_VALUES,
                          "basis": f"FE-{N_CHANNELS} uniform", "decoder": "ols",
                          "chance": CHANCE, "hc_hv4_excludes": "sub-07",
                          "script": str(Path(__file__).name)},
               "per_roi": {}}

    for roi in ROIS:
        amps = {s: load_amplitudes(BASELINE_DIR, s, roi) for s in ALL_SUBJECTS}
        jobs = []
        for n in N_VALUES:
            for si, subset in enumerate(subsets_by_n[n]):
                for subj in ALL_SUBJECTS:
                    jobs.append((n, si, subset, subj, amps[subj][np.array(subset)]))
        t0 = time.time()
        outs = Parallel(n_jobs=N_JOBS)(delayed(adjacc_cell)(a) for (*_, a) in jobs)
        print(f"  {roi}: {len(jobs)} cells in {time.time()-t0:.1f}s")
        roi_out = {n: {} for n in N_VALUES}
        for (n, si, subset, subj, _), res in zip(jobs, outs):
            roi_out[n].setdefault(subj, {})[f"subset_{si:02d}"] = {
                "runs": list(subset), **res}
        results["per_roi"][roi] = roi_out

    with open(OUT_DIR / "adjacc_saturation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {OUT_DIR/'adjacc_saturation.json'}")

    # ---- Retention summary: per ROI, per n, HC mean and CVD single-case d ----
    summary = {"chance": CHANCE, "metric": "LOCO adjacent accuracy (FE-6, OLS)",
               "note": "HC mean over subsets+subjects; CVD d_cc vs that n's HC subset distribution",
               "per_roi": {}}
    for roi in ROIS:
        hc_pool = HC_HV4 if roi == "V4" else HC_SUBJECTS
        roi_sum = {}
        for n in N_VALUES:
            rn = results["per_roi"][roi][n]
            # HC: average each HC subject's adjacc across that subject's subsets, then across subjects
            hc_subj_means = []
            hc_all_subset_vals = []   # pooled subset values for the single-case reference SD
            for s in hc_pool:
                vals = [c["adjacc"] for c in rn[s].values()]
                hc_subj_means.append(np.mean(vals))
                hc_all_subset_vals.extend(vals)
            hc_subj_means = np.array(hc_subj_means)
            hc_ref = np.array(hc_subj_means)   # per-subject means = single-case reference distribution
            entry = {
                "hc_mean": float(hc_subj_means.mean()),
                "hc_sem": float(hc_subj_means.std(ddof=1) / np.sqrt(len(hc_subj_means))),
                "hc_above_chance": bool(hc_subj_means.mean() > CHANCE),
                "cvd": {},
            }
            for s in CVD_SUBJECTS:
                vals = [c["adjacc"] for c in rn[s].values()]
                x = float(np.mean(vals))
                d, p = crawford_howell_d(x, hc_ref)
                entry["cvd"][s] = {"adjacc": x, "d_cc": round(d, 3), "p": round(p, 4),
                                   "below_chance": bool(x < CHANCE)}
            roi_sum[n] = entry
        summary["per_roi"][roi] = roi_sum

    with open(OUT_DIR / "adjacc_retention_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {OUT_DIR/'adjacc_retention_summary.json'}")

    # ---- Console: hV4 retention table (the paper's landmark ROI) ----
    print("\n=== hV4 LOCO adjacent-accuracy retention (chance=0.375) ===")
    print(f"{'n':>2} {'HC mean±SEM':>14} {'>chance':>7} | "
          f"{'s08 deut':>16} {'s09 prot':>16} {'s10 deut':>16}")
    for n in N_VALUES:
        e = summary["per_roi"]["V4"][n]
        row = f"{n:>2} {e['hc_mean']:.3f}±{e['hc_sem']:.3f}   {str(e['hc_above_chance']):>5} | "
        for s in CVD_SUBJECTS:
            c = e["cvd"][s]
            row += f"{c['adjacc']:.3f} d={c['d_cc']:+.2f}  "
        print(row)


if __name__ == "__main__":
    main()
