
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : docs/PAPER/repro/_perm_adjacent_n7.py  (development repository, commit 53c81c2)
# Role      : Control interpolation gate: 1,000 per-participant colour-label permutations of LOCO adjacent accuracy, all four ROIs, n = 7. Output: results/perm_adjacent_n7.json + perm_n7_null_*.npy
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

# Adjacent-accuracy above-chance permutation for ALL FOUR regions, n = 7 HC.
#
# Supersedes _perm_definitive_hv4.py (hV4 only, n = 6) and _perm_v1.py (V1 only,
# n = 7). Those two used different HC samples, so the published p-values were not
# comparable across regions, and V2/V3 were never run under this design at all.
#
# Design is unchanged from _perm_definitive_hv4.py: per-subject INDEPENDENT color
# label permutation (each HC subject draws its own rng.permutation(8) each draw),
# N = 1000, seed 42, add-one p (Phipson & Smyth 2010).
#
# sub-07 is retained at hV4 (16 voxels vs 67-70 in the other six). Excluding it
# leaves every conclusion unchanged and slightly weakens the CVD contrasts.
#
# SPEED. loco_forward_readouts is exact but costs ~29 s per draw over 7 subjects at
# V1, which is ~8 h for one region. loco_adj below is an algebraically exact
# reimplementation of loco_forward_readouts(..., decoder='ols', tasks=('adj',)):
#   (1) decode_hue loops 360 np.corrcoef calls per test pattern. Pearson r between
#       the K-vector response and each basis row is one matmul after z-scoring.
#   (2) fit_W_decoder('ols') calls lstsq(C_train, X_train). C_train depends only on
#       the fold, never on the data, so pinv(C_train) is hoisted out of the loop.
# verify() asserts agreement with the canonical routine to 1e-12 over 60 cases
# (4 regions x 5 subjects x {identity, two random permutations}). Run it first.
import json
import sys

import numpy as np

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
ROI_DIR = {"V1": "V1", "V2": "V2", "V3": "V3", "hV4": "V4"}
N_PERM = 1000
SEED = 42


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
            continue                       # degenerate fold stays 0.0, as in the canonical routine
        rz = np.divide(rz, rsd, out=np.zeros_like(rz), where=rsd > 0)
        corrs = (BASIS_Z @ rz) / K
        corrs[~BASIS_OK, :] = np.nan
        pred = np.nanargmax(corrs, axis=0).astype(float)[ok]
        err = np.abs(pred - HUES[c])
        adj[c] = float(np.mean(np.minimum(err, 360.0 - err) <= STEP))
    return float(adj.mean())


def verify():
    from loco_canonical import loco_forward_readouts
    rng = np.random.RandomState(7)
    pin = fold_pinvs(6)
    bad = 0
    for d in ROI_DIR.values():
        for s in ["01", "04", "07", "08", "09"]:
            amp = np.load(U.C010 / f"sub-{s}/{d}/amplitudes_procrustes.npy")
            for perm in [None, rng.permutation(8), rng.permutation(8)]:
                a = amp if perm is None else amp[:, perm, :]
                ref = loco_forward_readouts(a, C8, basis_full=BASIS, decoder="ols",
                                            tasks=("adj",))["adj"].mean()
                if not np.isclose(ref, loco_adj(a, pin), atol=1e-12):
                    bad += 1
    print("EXACT MATCH" if bad == 0 else f"{bad} MISMATCHES")
    return bad == 0


def main():
    pin = fold_pinvs(6)
    out = {}
    for roi, d in ROI_DIR.items():
        amps = [np.load(U.C010 / f"sub-{s}/{d}/amplitudes_procrustes.npy") for s in HC]
        per = [loco_adj(a, pin) for a in amps]
        obs = float(np.mean(per))
        rng = np.random.RandomState(SEED)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            null[i] = np.mean([loco_adj(a[:, rng.permutation(8), :], pin) for a in amps])
        p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
        out[roi] = dict(observed=obs, p_perm=p, null_mean=float(null.mean()),
                        null_sd=float(null.std()), n_subjects=len(HC), n_perms=N_PERM,
                        hc_sem=float(np.std(per, ddof=1) / np.sqrt(len(HC))),
                        per_subject=dict(zip(HC, map(float, per))))
        np.save(U.REPRO / f"perm_n7_null_{roi}.npy", null)
        print(f"{roi:4s} obs={obs:.4f} (SEM {out[roi]['hc_sem']:.4f})  "
              f"null={null.mean():.4f}+-{null.std():.4f}  p_perm={p:.4f}", flush=True)
    with open(U.REPRO / "perm_adjacent_n7.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    if verify():
        main()
