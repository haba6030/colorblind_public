"""null_within_hc_loo.py — Source B null for v6 PCA RDM verification.

Implements two within-HC LOO null variants for each production candidate
(S08-stable, S08-robust, S09-primary):

  B1  Real-data fake-CVD:
        fake_CVD := HC_k's amplitudes (untouched, real data)
        reference HC pool := HC \\ {HC_k}
        Run v6 fit on this fake CVD; record argmin (β_s, β_c) and losses.

  B2  Synth-GT-zero null:
        fake_CVD := synthesize_voxel_response(W_k, GT=(0,0), ...)
        reference HC pool := HC \\ {HC_k}
        Repeat for M noise realizations.

For each candidate × {B1, B2} × HC_k, record:
  - argmin β_s, β_c
  - loss at argmin
  - loss at origin (β_s=β_c=0)

DO NOT execute via this header — run from sbatch / interactive after review.

Output: results/redteam/null_within_hc_loo_v6_pca.json
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 05_identifiability
# Manuscript: Methods 'Identifiability and recovery'; Supplementary S12 (tab:identifiability); S13 control leave-one-out magnitude anchor
# Original  : analysis/phase5_filter_optimization/scripts/null_within_hc_loo.py  (development repository, commit 53c81c2)
# Role      : Test 2a (origin null) and Test 2b (control pseudo-CVD) -> null_within_hc_loo_v6_pca.json
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
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import s10b_v6_pca_rdm as v6                                     # noqa: E402
import two_comp                                                  # noqa: E402
from two_comp import forward_2comp                               # noqa: E402
from neural_loss import (                                        # noqa: E402
    load_amplitudes as _load_amps, ROI_K,
)
from behav_loss import load_jnd_per_pair as _load_jnd            # noqa: E402
from forward_voxel_synth import (                                # noqa: E402
    precompute_per_hc_W,
    estimate_noise_per_hc,
    synthesize_voxel_response,
    synth_provenance,
    SPATIAL_COV_RANK,
    AR1_RHO,
)

_PHASE1_FWD = SCRIPT_DIR.parents[1] / "phase4_forward_model" / "scripts"
sys.path.insert(0, str(_PHASE1_FWD))
from utils_forward_model import create_basis_full, HUE_ANGLES   # noqa: E402

OUT = (SCRIPT_DIR.parent / "results" / "redteam"
       / "null_within_hc_loo_v6_pca.json")

# ---------------------------------------------------------------------------
HC_ALL = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]
RNG_SEED = 27182                  # matches Exp 14 carrier seed for parity

# v6 fit cadence (single-fit per HC carrier; comparable to single CVD argmin).
v6.N_RESAMPLES = 1
v6.SUBSET_SIZE = 5                # production-matched (5 train / 2 complement)

M_NOISE_REALIZATIONS = 20         # for B2 only (B1 has no stochasticity)

CANDIDATES = [
    {"id": "S08-stable",  "subject": "sub-08", "family": "deutan",
     "combo_label": "γALL|RDMV1|noLOCO",
     "production_beta_s": 38.0, "production_beta_c": -10.0},
    {"id": "S08-robust",  "subject": "sub-08", "family": "deutan",
     "combo_label": "γOY|RDMV2|noLOCO",
     "production_beta_s":  6.0, "production_beta_c": -42.0},
    {"id": "S09-primary", "subject": "sub-09", "family": "protan",
     "combo_label": "γALL|RDMV1|noLOCO",
     "production_beta_s":  2.0, "production_beta_c":  24.0},
]

ROIS = list(v6.ROIS)


# ---------------------------------------------------------------------------
# Data preload
# ---------------------------------------------------------------------------
def _preload_data() -> tuple[dict, dict]:
    all_hc_amps: dict = {}
    for hc in HC_ALL:
        all_hc_amps[hc] = {}
        for roi in ROIS:
            try:
                amp = _load_amps(hc, roi)
                if roi == "V4" and amp.shape[2] < 20:
                    continue   # match neural_loss.load_hc_pool sparse-skip
                all_hc_amps[hc][roi] = amp
            except Exception:
                pass
    all_hc_jnd = {hc: _load_jnd(hc) for hc in HC_ALL}
    return all_hc_amps, all_hc_jnd


# ---------------------------------------------------------------------------
# Loss helper: evaluate composite loss at a single (β_s, β_c) given fit storage
# ---------------------------------------------------------------------------
def _argmin_record_from_storage(storage: dict, combo_label: str) -> Optional[dict]:
    """Pull the single 2comp argmin record from a v6 storage block.

    Returns dict with beta_s, beta_c, train_loss, boundary, or None.
    """
    block = storage.get(combo_label)
    if block is None:
        return None
    recs = block.get("2comp", [])
    if not recs:
        return None
    r = recs[0]
    return {
        "beta_s": float(r["beta_s"]),
        "beta_c": float(r["beta_c"]),
        "train_loss": float(r["train_loss"]) if r.get("train_loss") is not None else None,
        "boundary": bool(r.get("boundary", False)),
    }


# ---------------------------------------------------------------------------
# Monkey-patched v6 loaders for one (carrier_subject, fake_amps, pool)
# ---------------------------------------------------------------------------
def _run_v6_with_fake(subject_label: str, fake_amps_by_roi: dict,
                      pool_amps_by_roi_excl_donor: dict,
                      all_hc_jnd: dict, h_jnd_carrier: str,
                      original_hc_jnd_subjs: list) -> dict:
    """Monkey-patch v6 loaders and run fit_subject; restore on exit.

    Args:
        subject_label: 'sub-08' or 'sub-09' (drives combo enumeration).
        fake_amps_by_roi: {roi: (6, 8, V) ndarray} acting as fake CVD.
        pool_amps_by_roi_excl_donor: {roi: {hc_id: amp}} excluding the donor.
        all_hc_jnd: {hc_id: jnd dict} preloaded.
        h_jnd_carrier: the HC whose JND is used as the fake CVD's JND.
        original_hc_jnd_subjs: original v6.HC_JND_SUBJS list to restore.

    Returns:
        v6 storage dict (subject-level).
    """
    o_amps, o_hc, o_jnd = (v6.load_amplitudes, v6.load_hc_pool,
                           v6.load_jnd_per_pair)
    o_jnd_subjs = list(v6.HC_JND_SUBJS)
    o_hc_subjs = list(v6.HC_SUBJS)

    fake_jnd = dict(all_hc_jnd[h_jnd_carrier])

    def p_amps(sid: str, roi: str):
        if sid == subject_label:
            return fake_amps_by_roi.get(roi)
        # HC requested by name: serve from preloaded pool (full set)
        # (used by hc_amps_all in v6.fit_subject's load_hc_pool path).
        return None  # not used directly; load_hc_pool handles HC pool

    def p_hc(roi: str):
        return dict(pool_amps_by_roi_excl_donor.get(roi, {}))

    def p_jnd(sid: str):
        if sid == subject_label:
            return fake_jnd
        return all_hc_jnd.get(sid)

    v6.load_amplitudes = p_amps
    v6.load_hc_pool = p_hc
    v6.load_jnd_per_pair = p_jnd
    # Per-iteration LOO: exclude donor HC from v6's subset draw so the
    # resample is "5 train / 1 test" from remaining 6 (option α — train
    # pool size matches production).
    v6.HC_SUBJS = [h for h in HC_ALL if h != h_jnd_carrier]
    v6.HC_JND_SUBJS = [h for h in original_hc_jnd_subjs if h != h_jnd_carrier]

    try:
        storage = v6.fit_subject(subject_label)
    finally:
        v6.load_amplitudes = o_amps
        v6.load_hc_pool = o_hc
        v6.load_jnd_per_pair = o_jnd
        v6.HC_JND_SUBJS = o_jnd_subjs
        v6.HC_SUBJS = o_hc_subjs
    return storage


# ---------------------------------------------------------------------------
# B1: real-data fake-CVD
# ---------------------------------------------------------------------------
def run_b1(candidate: dict, all_hc_amps: dict, all_hc_jnd: dict,
           original_hc_jnd_subjs: list) -> list:
    """Loop over each HC_k acting as fake CVD; fit and record argmin."""
    subject_label = candidate["subject"]
    combo_label = candidate["combo_label"]
    records = []
    for hc_k in HC_ALL:
        donor_amps = all_hc_amps.get(hc_k, {})
        if not donor_amps:
            continue
        # Reference HC pool excludes donor for every ROI
        pool = {roi: {h: all_hc_amps[h][roi] for h in HC_ALL
                       if h != hc_k and roi in all_hc_amps[h]}
                for roi in ROIS}
        # Fake CVD ROI bundle = donor's amps; for V4 sparse-drop honoured
        fake_amps = {roi: donor_amps[roi] for roi in ROIS if roi in donor_amps}
        if not fake_amps:
            continue
        try:
            storage = _run_v6_with_fake(
                subject_label, fake_amps, pool,
                all_hc_jnd, hc_k, original_hc_jnd_subjs,
            )
        except Exception as e:
            records.append({"hc_donor": hc_k, "error": repr(e)})
            continue
        rec = _argmin_record_from_storage(storage, combo_label)
        if rec is None:
            records.append({"hc_donor": hc_k, "error": "no 2comp record"})
            continue
        rec["hc_donor"] = hc_k
        records.append(rec)
    return records


# ---------------------------------------------------------------------------
# B2: synth GT=(0,0) null per HC carrier
# ---------------------------------------------------------------------------
def _build_synth_fake_amps(hc_k: str, family: str,
                           all_hc_amps: dict,
                           rng: np.random.Generator) -> Optional[dict]:
    """One realization of synth GT=(0,0) fake CVD using HC_k as donor."""
    donor_amps = all_hc_amps.get(hc_k, {})
    if not donor_amps:
        return None
    fake_amps = {}
    for roi in ROIS:
        if roi not in donor_amps:
            continue
        K = ROI_K[roi]
        C_b = create_basis_full(K, basis_type="fe")[HUE_ANGLES.astype(int)]
        # encode this donor once
        W_dict = precompute_per_hc_W({hc_k: donor_amps[roi]}, C_b)
        if hc_k not in W_dict:
            continue
        W_k = W_dict[hc_k]
        noise = estimate_noise_per_hc(donor_amps[roi], W_k, C_b)
        Y_synth = synthesize_voxel_response(
            W_k=W_k, beta_s=0.0, beta_c=0.0, family=family,
            theta_canonical=HUE_ANGLES.astype(float),
            noise_params=noise, rng=rng,
        )
        fake_amps[roi] = Y_synth
    return fake_amps


def run_b2(candidate: dict, all_hc_amps: dict, all_hc_jnd: dict,
           original_hc_jnd_subjs: list,
           m_realizations: int = M_NOISE_REALIZATIONS) -> list:
    """Synth GT=(0,0) null × M noise realizations per HC carrier."""
    subject_label = candidate["subject"]
    family = candidate["family"]
    combo_label = candidate["combo_label"]
    records = []
    for hc_k in HC_ALL:
        for m in range(m_realizations):
            seed = (RNG_SEED + 7919 * hash(candidate["id"] + hc_k + str(m))) % (2**32)
            rng = np.random.default_rng(seed)
            fake_amps = _build_synth_fake_amps(hc_k, family, all_hc_amps, rng)
            if not fake_amps:
                continue
            pool = {roi: {h: all_hc_amps[h][roi] for h in HC_ALL
                          if h != hc_k and roi in all_hc_amps[h]}
                    for roi in ROIS}
            try:
                storage = _run_v6_with_fake(
                    subject_label, fake_amps, pool,
                    all_hc_jnd, hc_k, original_hc_jnd_subjs,
                )
            except Exception as e:
                records.append({"hc_donor": hc_k, "realization": m,
                                "error": repr(e)})
                continue
            rec = _argmin_record_from_storage(storage, combo_label)
            if rec is None:
                records.append({"hc_donor": hc_k, "realization": m,
                                "error": "no 2comp record"})
                continue
            rec["hc_donor"] = hc_k
            rec["realization"] = m
            records.append(rec)
    return records


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------
def _summarize_argmins(records: list) -> dict:
    valid = [r for r in records
             if "error" not in r and r.get("beta_s") is not None]
    if not valid:
        return {"n": 0}
    bs = np.array([r["beta_s"] for r in valid], dtype=float)
    bc = np.array([r["beta_c"] for r in valid], dtype=float)
    train = np.array([r["train_loss"] for r in valid
                       if r["train_loss"] is not None], dtype=float)
    return {
        "n": len(valid),
        "beta_s_median": float(np.median(bs)),
        "beta_s_iqr": float(np.percentile(bs, 75) - np.percentile(bs, 25)),
        "beta_s_range": [float(bs.min()), float(bs.max())],
        "beta_c_median": float(np.median(bc)),
        "beta_c_iqr": float(np.percentile(bc, 75) - np.percentile(bc, 25)),
        "beta_c_range": [float(bc.min()), float(bc.max())],
        "loss_at_argmin_median": float(np.median(train)) if train.size else None,
        "raw_beta_s": bs.tolist(),
        "raw_beta_c": bc.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-noise", type=int, default=M_NOISE_REALIZATIONS,
                        help="B2 noise realizations per HC carrier (default 20).")
    parser.add_argument("--candidates", type=str, default="all",
                        help="Comma-separated candidate ids or 'all'.")
    parser.add_argument("--skip-b1", action="store_true",
                        help="Skip B1 (real-data fake-CVD).")
    parser.add_argument("--skip-b2", action="store_true",
                        help="Skip B2 (synth GT=0).")
    args = parser.parse_args(argv)

    cand_filter = (None if args.candidates == "all"
                   else set(args.candidates.split(",")))
    cands = [c for c in CANDIDATES
             if cand_filter is None or c["id"] in cand_filter]

    print(f"[null_within_hc_loo] candidates: {[c['id'] for c in cands]}",
          flush=True)
    print(f"  SUBSET_SIZE={v6.SUBSET_SIZE} (production-matched 5/2)", flush=True)
    print(f"  N_RESAMPLES={v6.N_RESAMPLES} (single fit per HC carrier)",
          flush=True)

    print("[null_within_hc_loo] pre-loading data...", flush=True)
    all_hc_amps, all_hc_jnd = _preload_data()
    original_hc_jnd_subjs = list(v6.HC_JND_SUBJS)

    out = {
        "config": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seed": RNG_SEED,
            "m_noise_realizations": int(args.m_noise),
            "candidates": [c["id"] for c in cands],
            "subset_size": v6.SUBSET_SIZE,
            "n_resamples": v6.N_RESAMPLES,
            "spatial_cov_rank": int(SPATIAL_COV_RANK),
            "temporal_ar1_rho": float(AR1_RHO),
            "synth_provenance": synth_provenance(),
            "ROI_K": dict(ROI_K),
            "hc_pool": HC_ALL,
            "notes": (
                "B1 = real HC_k as fake CVD; reference pool excludes HC_k. "
                "B2 = synth (GT=0) via voxel-level forward through HC_k's W "
                "with HC_k's residual covariance + AR(1) noise. "
                "B1 single-fit because the underlying donor data is deterministic; "
                "B2 has M realizations of stochastic noise."
            ),
        },
        "cells": {},
    }

    t_all = time.time()
    for cand in cands:
        print(f"\n=== {cand['id']} ({cand['subject']}, {cand['family']}, "
              f"{cand['combo_label']}) ===", flush=True)
        cell = {"candidate": cand}

        if not args.skip_b1:
            t = time.time()
            b1_recs = run_b1(cand, all_hc_amps, all_hc_jnd,
                             original_hc_jnd_subjs)
            cell["B1_records"] = b1_recs
            cell["B1_summary"] = _summarize_argmins(b1_recs)
            print(f"  [B1] {cell['B1_summary'].get('n', 0)} HC carriers, "
                  f"{time.time() - t:.0f}s", flush=True)

        if not args.skip_b2:
            t = time.time()
            b2_recs = run_b2(cand, all_hc_amps, all_hc_jnd,
                             original_hc_jnd_subjs,
                             m_realizations=args.m_noise)
            cell["B2_records"] = b2_recs
            cell["B2_summary"] = _summarize_argmins(b2_recs)
            print(f"  [B2] {cell['B2_summary'].get('n', 0)} realizations, "
                  f"{time.time() - t:.0f}s", flush=True)

        out["cells"][cand["id"]] = cell

    out["elapsed_s"] = round(time.time() - t_all, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {OUT}  ({out['elapsed_s']}s total)", flush=True)


if __name__ == "__main__":
    main()
