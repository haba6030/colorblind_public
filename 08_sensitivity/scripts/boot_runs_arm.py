"""_boot_runs_arm.py — run-level bootstrap CI for LOCO adjacent accuracy.

Addresses the open item U5 of REVISION_PLAN_MOTION_GEOMETRY (run-level bootstrap,
6 runs x 2000 resamples) and the question raised by the hmc arm: the protan hV4
point estimate moves from 0.125 (canonical) to 0.271 (hmc). Neither number means
much unless we know how tightly a single participant's six runs determine it.

For each arm x subject x ROI we resample the six runs with replacement, recompute
adjacent accuracy with the frozen estimator (`loco_adj`, verified elsewhere to
match `loco_canonical` to 1e-12), and report the percentile CI.

We additionally bootstrap the *contrast*: on each iteration the HC mean is formed
from the resampled HC subjects' own resampled runs, so the CI covers both sources
of variability that enter the Crawford-Howell numerator.

    python _boot_runs_arm.py --arms with_residuals hmc_v2 --n-boot 2000
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 08_sensitivity
# Manuscript: Supplementary S2 (tab:motion_arms, tab:interp_arms, session-2 endpoints), S13 sign stability, S18 (tab:color_specificity, head-motion column)
# Original  : analysis/future_phase1_sensitivity/scripts/_boot_runs_arm.py  (development repository, commit 53c81c2)
# Role      : Run-bootstrap confidence intervals of adjacent accuracy per arm.
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

import sys as _sys
from pathlib import Path as _Path

# Moved out of docs/PAPER/repro on 2026-08-17 (analysis/future_phase1_sensitivity).
# _repro_util still lives there and stays the single definition of the data roots.
_REPRO = _Path(__file__).resolve().parents[3] / "docs" / "PAPER" / "repro"
_sys.path.insert(0, str(_REPRO))
OUT = _Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np

import _repro_util as U
from _perm_adjacent_arm import (  # noqa: E402
    CVD, HC, ROI_DIR, arm_root, fold_pinvs, load, loco_adj,
)

SEED = 20260815


def boot_subject(amp, pinvs, n_boot, rng):
    n_runs = amp.shape[0]
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.randint(0, n_runs, n_runs)
        out[b] = loco_adj(amp[idx], pinvs)
    return out


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def run(arm, n_boot):
    root = arm_root(arm)
    print(f"\n=== arm: {arm} ({root.name}) ===", flush=True)
    res = {"arm": arm, "root": str(root), "n_boot": n_boot, "rois": {}}

    for roi, rdir in ROI_DIR.items():
        rng = np.random.RandomState(SEED)
        amps = {s: load(root, s, rdir) for s in HC + list(CVD)}
        pinvs = fold_pinvs(next(iter(amps.values())).shape[0])

        point = {s: loco_adj(a, pinvs) for s, a in amps.items()}
        boots = {s: boot_subject(a, pinvs, n_boot, rng) for s, a in amps.items()}

        hc_stack = np.vstack([boots[s] for s in HC])          # (7, n_boot)
        rng2 = np.random.RandomState(SEED + 1)
        hc_mean_boot = np.empty(n_boot)
        for b in range(n_boot):                                # resample subjects too
            pick = rng2.randint(0, len(HC), len(HC))
            hc_mean_boot[b] = hc_stack[pick, b].mean()

        entry = {"hc_point_mean": float(np.mean([point[s] for s in HC])),
                 "hc_point_sd": float(np.std([point[s] for s in HC], ddof=1)),
                 "hc_mean_ci": ci(hc_mean_boot), "subjects": {}, "contrast": {}}
        for s in HC + list(CVD):
            lo, hi = ci(boots[s])
            entry["subjects"][f"sub-{s}"] = {
                "point": round(point[s], 4), "boot_median": round(float(np.median(boots[s])), 4),
                "ci95": [round(lo, 4), round(hi, 4)], "width": round(hi - lo, 4),
                "group": "CVD" if s in CVD else "HC"}
        for s, kind in CVD.items():
            diff = boots[s] - hc_mean_boot
            lo, hi = ci(diff)
            entry["contrast"][f"sub-{s}"] = {
                "cvd_type": kind,
                "deficit_point": round(point[s] - entry["hc_point_mean"], 4),
                "deficit_boot_median": round(float(np.median(diff)), 4),
                "deficit_ci95": [round(lo, 4), round(hi, 4)],
                "frac_below_zero": round(float((diff < 0).mean()), 4)}
        res["rois"][roi] = entry

        print(f"  [{roi}] HC {entry['hc_point_mean']:.3f} "
              f"CI[{entry['hc_mean_ci'][0]:.3f},{entry['hc_mean_ci'][1]:.3f}]", flush=True)
        for s, kind in CVD.items():
            e = entry["subjects"][f"sub-{s}"]; c = entry["contrast"][f"sub-{s}"]
            print(f"        {kind:6s} {e['point']:.3f} CI[{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}] "
                  f"w={e['width']:.3f} | deficit {c['deficit_point']:+.3f} "
                  f"CI[{c['deficit_ci95'][0]:+.3f},{c['deficit_ci95'][1]:+.3f}] "
                  f"P(<0)={c['frac_below_zero']:.3f}", flush=True)

    with open(OUT / f"boot_runs_{arm}.json", "w") as f:
        json.dump(res, f, indent=2)
    print(f"  wrote boot_runs_{arm}.json", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["with_residuals", "hmc_v2"])
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    for arm in a.arms:
        run(arm, a.n_boot)


if __name__ == "__main__":
    main()
