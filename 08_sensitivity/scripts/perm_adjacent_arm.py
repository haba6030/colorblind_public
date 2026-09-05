
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 08_sensitivity
# Manuscript: Supplementary S2 (tab:motion_arms, tab:interp_arms, session-2 endpoints), S13 sign stability, S18 (tab:color_specificity, head-motion column)
# Original  : analysis/future_phase1_sensitivity/scripts/_perm_adjacent_arm.py  (development repository, commit 53c81c2)
# Role      : Control interpolation gate on a given pipeline arm -> perm_adjacent_arm_<arm>.json + null arrays.
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

# Adjacent-accuracy above-chance permutation across PREPROCESSING ARMS.
#
# Companion to _perm_adjacent_n7.py, which runs the published arm only. S2 states
# that "every neural endpoint was recomputed" under motion regression, but only
# disparity, the frozen-projection permutation and split-half reliability were.
# The hV4 LOCO interpolation result -- the sole permutation-gate survivor -- was
# never recomputed. This driver closes that gap.
#
# Arms (all four C010 trees exist locally):
#   with_residuals  published arm, 1 resampling, drift regressors only
#   motreg          0 resamplings, drift + 12 MCFLIRT regressors
#   motshift        0 resamplings, drift + the same 12 regressors circularly
#                   shifted within run -- isolates the cost of adding regressors
#                   from the removal of motion-aligned variance
#
# Design is identical to _perm_adjacent_n7.py in every other respect: per-subject
# INDEPENDENT color label permutation, N = 1000, seed 42, add-one p (Phipson &
# Smyth 2010), FE-6 uniform basis, OLS decoder, sub-07 retained at hV4.
# verify() asserts loco_adj agrees with loco_canonical to 1e-12 on the arm being
# run, so a mis-shaped tree fails loudly instead of silently.
#
# Also computes, per arm, each CVD participant's adjacent accuracy and the
# Crawford & Howell single-case test against the seven controls (lower tail),
# which is the form reported in the paper.
import argparse
import json
import sys

import sys as _sys
from pathlib import Path as _Path

# Moved out of docs/PAPER/repro on 2026-08-17 (analysis/future_phase1_sensitivity).
# _repro_util still lives there and stays the single definition of the data roots.
_REPRO = _Path(__file__).resolve().parents[3] / "docs" / "PAPER" / "repro"
_sys.path.insert(0, str(_REPRO))
OUT = _Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np
from scipy import stats

import _repro_util as U

sys.path.insert(0, str(U.P1 / "scripts"))
from utils_forward_model import create_basis_matrix, HUE_ANGLES  # noqa: E402

K = 6
N_COLORS = 8
STEP = 360.0 / N_COLORS
HUES = np.asarray(HUE_ANGLES, dtype=float)
C8 = create_basis_matrix(HUES, K, "fe")
BASIS = create_basis_matrix(np.arange(360), K, "fe")

_b = BASIS - BASIS.mean(axis=1, keepdims=True)
_bsd = _b.std(axis=1, keepdims=True)
BASIS_Z = np.divide(_b, _bsd, out=np.zeros_like(_b), where=_bsd > 0)
BASIS_OK = _bsd[:, 0] > 0

HC = ["01", "02", "03", "04", "05", "06", "07"]
CVD = {"08": "deutan", "09": "protan"}
ROI_DIR = {"V1": "V1", "V2": "V2", "V3": "V3", "hV4": "V4"}
N_PERM = 1000
SEED = 42

VIS = U.ANALYSIS / "phase1_procrustes_decoding/results/visualization"


def arm_root(arm):
    return VIS / f"full_dataset_C010_{arm}"


def fold_pinvs(n_runs):
    out = []
    for c in range(N_COLORS):
        train = [k for k in range(N_COLORS) if k != c]
        out.append((train, np.linalg.pinv(np.tile(C8[train], (n_runs, 1)))))
    return out


def loco_adj(amp, pinvs):
    """Mean adjacent accuracy over the eight leave-one-color-out folds."""
    n_runs, n_colors, V = amp.shape
    adj = np.zeros(n_colors)
    for c in range(n_colors):
        train, P = pinvs[c]
        W = P @ amp[:, train, :].reshape(-1, V)
        resp = W @ amp[:, c, :].T
        rz = resp - resp.mean(axis=0, keepdims=True)
        rsd = rz.std(axis=0, keepdims=True)
        ok = rsd[0] > 0
        if not ok.any():
            continue
        rz = np.divide(rz, rsd, out=np.zeros_like(rz), where=rsd > 0)
        corrs = (BASIS_Z @ rz) / K
        corrs[~BASIS_OK, :] = np.nan
        pred = np.nanargmax(corrs, axis=0).astype(float)[ok]
        err = np.abs(pred - HUES[c])
        adj[c] = float(np.mean(np.minimum(err, 360.0 - err) <= STEP))
    return float(adj.mean())


def load(root, sub, roi_dir):
    return np.load(root / f"sub-{sub}/{roi_dir}/amplitudes_procrustes.npy")


def verify(root):
    from loco_canonical import loco_forward_readouts
    rng = np.random.RandomState(7)
    pin = fold_pinvs(6)
    bad = 0
    for d in ROI_DIR.values():
        for s in ["01", "04", "07", "08", "09"]:
            amp = load(root, s, d)
            for perm in [None, rng.permutation(8), rng.permutation(8)]:
                a = amp if perm is None else amp[:, perm, :]
                ref = loco_forward_readouts(a, C8, basis_full=BASIS, decoder="ols",
                                            tasks=("adj",))["adj"].mean()
                if not np.isclose(ref, loco_adj(a, pin), atol=1e-12):
                    bad += 1
    print(f"  verify: {'EXACT MATCH' if bad == 0 else str(bad) + ' MISMATCHES'}", flush=True)
    return bad == 0


def run_arm(arm):
    root = arm_root(arm)
    print(f"\n=== arm: {arm}  ({root.name}) ===", flush=True)
    if not verify(root):
        raise SystemExit(f"verify failed for arm {arm}")

    pin = fold_pinvs(6)
    out = {}
    for roi, d in ROI_DIR.items():
        amps = [load(root, s, d) for s in HC]
        per = [loco_adj(a, pin) for a in amps]
        obs = float(np.mean(per))
        rng = np.random.RandomState(SEED)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = np.mean([loco_adj(a[:, rng.permutation(8), :], pin) for a in amps])
        p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))

        hc_sd = float(np.std(per, ddof=1))
        cvd = {}
        for s, label in CVD.items():
            v = loco_adj(load(root, s, d), pin)
            t = (v - obs) / (hc_sd * np.sqrt((len(HC) + 1) / len(HC)))
            cvd[label] = dict(
                adjacent=float(v),
                t=float(t),
                p_one_tailed_lower=float(stats.t.cdf(t, len(HC) - 1)),
                d_cc=float((v - obs) / hc_sd),
            )

        out[roi] = dict(observed=obs, p_perm=p, null_mean=float(null.mean()),
                        null_sd=float(null.std()), n_subjects=len(HC), n_perms=N_PERM,
                        hc_sd=hc_sd, hc_sem=float(hc_sd / np.sqrt(len(HC))),
                        per_subject=dict(zip(HC, map(float, per))), cvd=cvd)
        print(f"  {roi:4s} HC obs={obs:.4f} (SD {hc_sd:.4f})  null={null.mean():.4f}"
              f"+-{null.std():.4f}  p_perm={p:.4f}   "
              f"deutan={cvd['deutan']['adjacent']:.3f} (p={cvd['deutan']['p_one_tailed_lower']:.3f})  "
              f"protan={cvd['protan']['adjacent']:.3f} (p={cvd['protan']['p_one_tailed_lower']:.3f})",
              flush=True)
        np.save(OUT / f"perm_arm_{arm}_null_{roi}.npy", null)

    with open(OUT / f"perm_adjacent_arm_{arm}.json", "w") as f:
        json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["with_residuals", "motreg", "motshift"])
    a = ap.parse_args()
    for arm in a.arms:
        run_arm(arm)
