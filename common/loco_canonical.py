#!/usr/bin/env python3
"""
Canonical leave-one-color-out (LOCO) forward-encoding readouts — SHARED by
Phase-1 and exp2 so the two never diverge again.

Decoder rule (see analysis/decoder-comparison.md):
  - ENCODING  (voxel-pattern prediction rho)         -> decoder='gcv' (ridge-GCV)
  - DECODING  (hue interpolation adjacent/exact acc)  -> decoder='ols' (alpha=0,
                                                         B&H 2009 pseudoinverse)
Computing the WRONG decoder for a readout is the bug this module exists to prevent
(ridge-GCV collapses decoding accuracy to chance; alpha=0 collapses encoding rho).

All primitives are imported FROZEN from utils_forward_model (not reimplemented):
create_basis_matrix, HUE_ANGLES, fit_W_ridge, gcv_select_alpha, decode_hue,
circular_distance.

Training (both decoders): pooled-run, leave-one-color-out, on the other 7 colors.
Decoding: per-RUN decode of the held-out color (decode_hue over the 360-hue FE
basis), averaged within color — matches methods_v2.tex "averaged across the runs".
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : common
# Manuscript: shared module
# Original  : analysis/phase4_forward_model/scripts/loco_canonical.py  (development repository, commit 53c81c2)
# Role      : Canonical leave-one-color-out readouts (adjacent / exact accuracy, forward-tuning rho). Methods 'Cross-color interpolation (LOCO)'.
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[1]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p

import numpy as np

from utils_forward_model import (
    create_basis_matrix, HUE_ANGLES, fit_W_ridge, gcv_select_alpha,
    decode_hue, circular_distance,
)

VALID_DECODERS = ("ols", "gcv")
VALID_TASKS = ("rho", "adj", "exact")


def fit_W_decoder(C_train, X_train, decoder):
    """Fit the channel->voxel weight matrix W (K, V) with the chosen decoder.
    'ols' = alpha=0 pseudoinverse (B&H 2009; for DECODING accuracy).
    'gcv' = ridge with GCV-selected alpha (for ENCODING voxel prediction)."""
    if decoder == "ols":
        return np.linalg.lstsq(C_train, X_train, rcond=None)[0]
    if decoder == "gcv":
        alpha, _ = gcv_select_alpha(C_train, X_train)
        return fit_W_ridge(C_train, X_train, alpha)
    raise ValueError(f"decoder must be one of {VALID_DECODERS}, got {decoder!r}")


def loco_forward_readouts(amp, C8, basis_full=None, decoder="ols",
                          tasks=("rho", "adj", "exact"), adjacent_deg=None,
                          hues=None):
    """Leave-one-color-out forward-encoding readouts for ONE decoder.

    amp        : (n_runs, n_colors, V) per-run voxel amplitudes.
    C8         : (n_colors, K) FE basis evaluated at the stimulus hues.
    basis_full : (360, K) FE basis over integer hues (for decode_hue). Built from
                 C8's K if None and a decoding task is requested.
    decoder    : 'ols' (alpha=0) or 'gcv' (ridge-GCV).
    tasks      : subset of ('rho','adj','exact') to compute (skip decode if no
                 adj/exact -> keeps the rho-only path as fast as before).
    adjacent_deg : ±tolerance for adjacent accuracy (45 deg = ±1 hue step).
    hues       : (n_colors,) true hue of each color (deg); defaults to HUE_ANGLES.

    Returns dict with a (n_colors,) array per requested task:
      'rho'   ENCODING : Pearson(pred pattern C8[c]@W, run-mean actual)   (0 if degenerate)
      'adj'   DECODING : per-run frac within ±adjacent_deg of true hue, mean over runs
      'exact' DECODING : per-run exact 45-deg-bin hit, mean over runs
    """
    tasks = tuple(tasks)
    bad = set(tasks) - set(VALID_TASKS)
    if bad:
        raise ValueError(f"unknown tasks {bad}; valid: {VALID_TASKS}")
    amp = np.asarray(amp, dtype=float)
    n_runs, n_colors, V = amp.shape
    step = 360.0 / n_colors                     # one hue step (45 deg for 8 colors)
    if adjacent_deg is None:                    # default ±1 step; derived, not hardcoded
        adjacent_deg = step
    if hues is None:
        hues = np.asarray(HUE_ANGLES, dtype=float)
    need_decode = ("adj" in tasks) or ("exact" in tasks)
    if need_decode and basis_full is None:
        basis_full = create_basis_matrix(np.arange(360), C8.shape[1], basis_type="fe")

    amp_mean = amp.mean(axis=0)
    out = {t: np.zeros(n_colors) for t in tasks}
    for c in range(n_colors):
        train = [k for k in range(n_colors) if k != c]
        C_train = np.tile(C8[train], (n_runs, 1))         # (n_runs*(n_colors-1), K)
        X_train = amp[:, train, :].reshape(-1, V)         # (n_runs*(n_colors-1), V)
        W = fit_W_decoder(C_train, X_train, decoder)      # (K, V)

        if "rho" in tasks:
            Y_pred = (C8[c:c + 1] @ W)[0]
            Y_act = amp_mean[c]
            if Y_pred.std() < 1e-10 or Y_act.std() < 1e-10:
                out["rho"][c] = 0.0
            else:
                r = np.corrcoef(Y_pred, Y_act)[0, 1]
                out["rho"][c] = r if np.isfinite(r) else 0.0

        if need_decode:
            pred = decode_hue(W, basis_full, amp[:, c, :])   # (n_runs,) deg, NaN if degenerate
            valid = ~np.isnan(pred)
            if valid.any():
                pv = pred[valid]
                if "adj" in tasks:
                    err = circular_distance(np.full(pv.shape, hues[c]), pv)
                    out["adj"][c] = float(np.mean(err <= adjacent_deg))
                if "exact" in tasks:
                    out["exact"][c] = float(
                        np.mean((np.round(pv / step).astype(int) % n_colors) == c))
            # CONVENTION: degenerate fold (all-NaN decode, e.g. std=0 channel
            # response) leaves adj/exact at 0.0, which the caller averages in -> it
            # deflates the per-color mean BELOW chance rather than being excluded.
            # This matches the existing reference pipeline (so HC/CVD stay
            # comparable); a NaN-skip (excluded-fold) policy would change CVD and
            # low-run-count numbers. Documented, not silently changed.
    return out
