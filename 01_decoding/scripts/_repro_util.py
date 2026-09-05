"""Shared helpers for the colorBlind PAPER reproduction notebooks.

Provenance: built by ~/.claude/skills/repro-notebook/SKILL.md (Phase 3).
Companion: docs/PAPER/repro/{MANIFEST.md, MAP.md, REPORT.md}.
Target paper: docs/PAPER/main.tex (results_v4).

Two execution modes are used in the notebooks:
  * RECOMPUTE (cheap, local)  -> E2 adjacent accuracy, E4.13 pre-image |dtheta|.
  * LOAD+VERIFY (heavy/MPI)   -> everything else: load the committed result
    JSON and check the stored value against the number printed in the paper.

Nothing here runs BrainIAK/MPI; SRM-dependent numbers are read from committed JSON.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : docs/PAPER/repro/_repro_util.py  (development repository, commit 53c81c2)
# Role      : Path and Crawford-Howell helpers used by perm_adjacent_n7.py (development-repository layout).
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
import os
from pathlib import Path

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- paths
# repo root = parent of 'analysis' and 'docs'. This file lives at
# docs/PAPER/repro/_repro_util.py
REPO = Path(__file__).resolve().parents[3]
REPRO = Path(__file__).resolve().parent          # docs/PAPER/repro (committed null arrays live here)
ANALYSIS = REPO / "analysis"
P1 = ANALYSIS / "phase4_forward_model"
P2 = ANALYSIS / "phase5_filter_optimization"
P3 = ANALYSIS / "phase6_behavioral_analysis"
SRM = ANALYSIS / "phase2_SRM_across_between"
DEC = ANALYSIS / "phase3_decoder_comparing"
# local C010 amplitudes (procrustes), shape (6, 8, V) per subject/ROI
C010 = ANALYSIS / "phase1_procrustes_decoding/results/visualization/full_dataset_C010_with_residuals"

SEED = 42
np.random.seed(SEED)

HC = [f"{i:02d}" for i in range(1, 8)]
HUE_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]


# ---------------------------------------------------------------- io
def load_json(path):
    path = Path(path)
    with open(path) as f:
        return json.load(f)


def exists(path):
    return Path(path).exists()


# ---------------------------------------------------------------- verifier
_RESULTS = []


def check(label, produced, reported, tol=None, mode="round", ndigits=2):
    """Compare a produced value to the paper-reported value.

    mode='round'     : equal after rounding to `ndigits` (default), or |diff|<=tol if tol given.
    mode='threshold' : pass if produced < reported (used for p<chance style claims) -- pass tol as cmp.
    mode='ge'        : pass if produced >= reported (e.g. accuracy above chance).
    Records and prints a line; returns bool.
    """
    ok = False
    try:
        if mode == "round":
            if tol is not None:
                ok = abs(float(produced) - float(reported)) <= tol
            else:
                ok = round(float(produced), ndigits) == round(float(reported), ndigits)
        elif mode == "ge":
            ok = float(produced) >= float(reported)
        elif mode == "threshold":
            ok = float(produced) < float(reported)
    except Exception as e:  # noqa
        ok = False
        produced = f"ERR:{e}"
    mark = "OK " if ok else "XX "
    _RESULTS.append((ok, label, produced, reported))
    print(f"[{mark}] {label}: produced={produced}  reported={reported}")
    return ok


def summary():
    n = len(_RESULTS)
    p = sum(1 for r in _RESULTS if r[0])
    print(f"\n=== {p}/{n} checks reproduced ===")
    for ok, label, prod, rep in _RESULTS:
        if not ok:
            print(f"  MISMATCH {label}: produced={prod} reported={rep}")
    return p, n


# ---------------------------------------------------------------- E2 recompute
def _p1_scripts():
    import sys
    s = str(P1 / "scripts")
    if s not in sys.path:
        sys.path.insert(0, s)


def adjacent_accuracy_profile(subject, roi="V4", K=6, decoder="ols"):
    """Per-hue LOCO adjacent accuracy (canonical: FE-6 uniform basis + OLS).

    Returns 8-vector v in [0,1] or None if amplitudes absent.
    """
    _p1_scripts()
    from loco_canonical import loco_forward_readouts
    from utils_forward_model import create_basis_matrix, HUE_ANGLES

    amp_path = C010 / f"sub-{subject}/{roi}/amplitudes_procrustes.npy"
    if not amp_path.exists():
        return None
    amp = np.load(amp_path)
    hues = np.asarray(HUE_ANGLES, float)
    C8 = create_basis_matrix(hues, K, basis_type="fe")
    basis_full = create_basis_matrix(np.arange(360), K, basis_type="fe")
    out = loco_forward_readouts(amp, C8, basis_full=basis_full, decoder=decoder, tasks=("adj",))
    return out["adj"]


def crawford_howell(case, control, tail="lower"):
    """Crawford & Howell (1998) single-case t. Returns (t, p_one_tailed, d_cc)."""
    control = np.asarray(control, float)
    control = control[~np.isnan(control)]
    n = len(control)
    m = control.mean()
    sd = control.std(ddof=1)
    if sd == 0:
        return np.nan, np.nan, np.nan
    t = (case - m) / (sd * np.sqrt((n + 1) / n))
    if tail == "lower":
        p = stats.t.cdf(t, n - 1)
    elif tail == "upper":
        p = stats.t.sf(t, n - 1)
    else:
        p = stats.t.sf(abs(t), n - 1)
    d = (case - m) / sd
    return float(t), float(p), float(d)


def hc_adjacent_matrix(roi="V4", K=6, decoder="ols", exclude=("07",)):
    """HC per-hue adjacent-accuracy matrix (n_hc x 8). hV4 excludes sub-07."""
    rows = []
    for s in HC:
        if s in exclude:
            continue
        v = adjacent_accuracy_profile(s, roi=roi, K=K, decoder=decoder)
        if v is not None:
            rows.append(v)
    return np.array(rows)


# ---------------------------------------------------------------- E4.13 recompute
def preimage_mean_abs_delta(subject):
    """mean |delta_theta| of the committed 2-component pre-image filter."""
    j = load_json(P2 / f"results/exp2_preimage/sub-{subject}_2component_preimage.json")
    d = np.asarray(j["delta_theta_apply_deg"], float)
    return float(np.mean(np.abs(d)))
