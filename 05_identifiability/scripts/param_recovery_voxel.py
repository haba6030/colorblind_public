"""param_recovery_voxel.py — Stage 1: voxel-level parameter recovery.

Replaces Exp 21 (which used loss-function Method C). For each candidate ×
HC carrier:

  GT_β_s = candidate.production_beta_s
  GT_β_c = candidate.production_beta_c
  For m ∈ {1, ..., M=20}:
      synth_k_m = synthesize_voxel_response(W_k, GT_β_s, GT_β_c, family, noise_k)
      reference HC pool = HC \\ {HC_k}
      v6 fit_subject(...)        → recovered (β_s_hat, β_c_hat)

Per cell (candidate, HC_k):
  median recovered (β_s, β_c), IQR, bias, distance from GT, frac_within_5°
  AND frac_within_10° (for verdict).

GT=(0,0) is covered by null_within_hc_loo B2 (Source A). 0.5×/1.5× sweep
removed — recovery test only needs the production GT.

Total fits: 3 candidates × 7 HCs × M=20 = 420
(V4-only carriers drop one HC ⇒ effective ≤ 420; the actual count is
recorded per cell.)

DO NOT execute via this header. Run via SLURM (parallelise on candidate × mag).

Output: results/redteam/param_recovery_voxel_v6_pca.json
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 05_identifiability
# Manuscript: Methods 'Identifiability and recovery'; Supplementary S12 (tab:identifiability); S13 control leave-one-out magnitude anchor
# Original  : analysis/phase5_filter_optimization/scripts/param_recovery_voxel.py  (development repository, commit 53c81c2)
# Role      : Test 1: recovery fraction within 10 deg and bias -> param_recovery_voxel_v6_pca_v2.json
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
    synthesize_fake_jnd,
    synth_provenance,
    SPATIAL_COV_RANK,
    AR1_RHO,
)
from s8_loo_train_test import jnd_baseline_from_pool             # noqa: E402

_PHASE1_FWD = SCRIPT_DIR.parents[1] / "phase4_forward_model" / "scripts"
sys.path.insert(0, str(_PHASE1_FWD))
from utils_forward_model import create_basis_full, HUE_ANGLES   # noqa: E402

OUT = (SCRIPT_DIR.parent / "results" / "redteam"
       / "param_recovery_voxel_v6_pca_v2.json")

# ---------------------------------------------------------------------------
HC_ALL = ["sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06", "sub-07"]
RNG_SEED = 31337
M_REALIZATIONS = 20
MAGNITUDES = [1.0]

v6.N_RESAMPLES = 1
v6.SUBSET_SIZE = 5                # production-matched

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
def _preload_data() -> tuple[dict, dict]:
    all_hc_amps: dict = {}
    for hc in HC_ALL:
        all_hc_amps[hc] = {}
        for roi in ROIS:
            try:
                amp = _load_amps(hc, roi)
                if roi == "V4" and amp.shape[2] < 20:
                    continue
                all_hc_amps[hc][roi] = amp
            except Exception:
                pass
    all_hc_jnd = {hc: _load_jnd(hc) for hc in HC_ALL}
    return all_hc_amps, all_hc_jnd


def _build_synth_fake_amps(hc_k: str, family: str,
                            gt_bs: float, gt_bc: float,
                            all_hc_amps: dict,
                            rng: np.random.Generator) -> dict:
    """Synthesize fake CVD voxels at GT for each ROI using HC_k as donor.

    Returns {roi: (n_runs, 8, V_k)} keyed only on ROIs HC_k has.
    """
    donor_amps = all_hc_amps.get(hc_k, {})
    fake_amps = {}
    for roi in ROIS:
        if roi not in donor_amps:
            continue
        K = ROI_K[roi]
        C_b = create_basis_full(K, basis_type="fe")[HUE_ANGLES.astype(int)]
        W_dict = precompute_per_hc_W({hc_k: donor_amps[roi]}, C_b)
        if hc_k not in W_dict:
            continue
        W_k = W_dict[hc_k]
        noise = estimate_noise_per_hc(donor_amps[roi], W_k, C_b)
        Y_synth = synthesize_voxel_response(
            W_k=W_k, beta_s=gt_bs, beta_c=gt_bc, family=family,
            theta_canonical=HUE_ANGLES.astype(float),
            noise_params=noise, rng=rng,
        )
        fake_amps[roi] = Y_synth
    return fake_amps


def _build_synth_fake_jnd(family: str, gt_bs: float, gt_bc: float,
                          pool_jnd_subjs: list,
                          rng: np.random.Generator) -> dict:
    """Behaviorally-consistent fake CVD JND at GT, using v6's eventual pool.

    Computes (baseline, sd) per pair across HC \\ {donor} (which is
    pool_jnd_subjs as set by the alpha LOO patch), then draws
    pred_jnd + N(0, sd_pair) per pair.

    Returns dict {pair_name: float|None}.
    """
    bl, sd = jnd_baseline_from_pool(pool_jnd_subjs)
    return synthesize_fake_jnd(
        beta_s=gt_bs, beta_c=gt_bc, family=family,
        pool_jnd_baseline=bl, pool_jnd_sd=sd, rng=rng,
        theta_canonical=HUE_ANGLES.astype(float),
    )


def _run_v6_synth(subject_label: str, fake_amps_by_roi: dict,
                  fake_jnd: dict,
                  pool_amps_by_roi_excl_donor: dict,
                  all_hc_jnd: dict, h_jnd_carrier: str,
                  original_hc_jnd_subjs: list) -> dict:
    """Mirror of null_within_hc_loo's monkeypatch helper. Inlined for clarity.

    fake_jnd is passed in (built once per cell at GT) — this fixes the prior
    inconsistency where fake_jnd was donor's REAL JND while voxels were
    synth at GT≠0, biasing γ-atom toward δ=0.
    """
    o_amps, o_hc, o_jnd = (v6.load_amplitudes, v6.load_hc_pool,
                           v6.load_jnd_per_pair)
    o_jnd_subjs = list(v6.HC_JND_SUBJS)
    o_hc_subjs = list(v6.HC_SUBJS)

    def p_amps(sid, roi):
        if sid == subject_label:
            return fake_amps_by_roi.get(roi)
        return None

    def p_hc(roi):
        return dict(pool_amps_by_roi_excl_donor.get(roi, {}))

    def p_jnd(sid):
        if sid == subject_label:
            return fake_jnd
        return all_hc_jnd.get(sid)

    v6.load_amplitudes = p_amps
    v6.load_hc_pool = p_hc
    v6.load_jnd_per_pair = p_jnd
    # Per-iteration LOO: exclude donor HC from v6's subset draw (option α —
    # 5 train / 1 test from remaining 6, preserving production train pool).
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


def _argmin_record(storage: dict, combo_label: str) -> Optional[dict]:
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
def run_cell(candidate: dict, magnitude: float, hc_k: str,
             all_hc_amps: dict, all_hc_jnd: dict,
             original_hc_jnd_subjs: list,
             m_realizations: int = M_REALIZATIONS) -> dict:
    """One cell = (candidate, magnitude, HC_k). M noise realizations."""
    subject_label = candidate["subject"]
    family = candidate["family"]
    combo_label = candidate["combo_label"]
    gt_bs = candidate["production_beta_s"] * magnitude
    gt_bc = candidate["production_beta_c"] * magnitude

    pool = {roi: {h: all_hc_amps[h][roi] for h in HC_ALL
                  if h != hc_k and roi in all_hc_amps[h]}
            for roi in ROIS}

    records = []
    for m in range(m_realizations):
        seed = (RNG_SEED
                + 7919 * (hash(candidate["id"] + hc_k
                               + str(magnitude) + str(m)) % 99991)) % (2**32)
        rng = np.random.default_rng(seed)
        fake_amps = _build_synth_fake_amps(
            hc_k, family, gt_bs, gt_bc, all_hc_amps, rng,
        )
        if not fake_amps:
            records.append({"realization": m, "error": "no fake amps"})
            continue
        # GT-consistent fake JND from pool = HC \ {donor} (v6's effective pool)
        pool_jnd_subjs = [h for h in original_hc_jnd_subjs if h != hc_k]
        fake_jnd = _build_synth_fake_jnd(
            family, gt_bs, gt_bc, pool_jnd_subjs, rng,
        )
        try:
            storage = _run_v6_synth(
                subject_label, fake_amps, fake_jnd, pool,
                all_hc_jnd, hc_k, original_hc_jnd_subjs,
            )
        except Exception as e:
            records.append({"realization": m, "error": repr(e)})
            continue
        rec = _argmin_record(storage, combo_label)
        if rec is None:
            records.append({"realization": m, "error": "no 2comp record"})
            continue
        rec["realization"] = m
        records.append(rec)

    valid = [r for r in records
             if "error" not in r and r.get("beta_s") is not None]
    if not valid:
        return {"hc_donor": hc_k, "magnitude": magnitude,
                "gt": [gt_bs, gt_bc], "n": 0, "records": records}

    bs = np.array([r["beta_s"] for r in valid], dtype=float)
    bc = np.array([r["beta_c"] for r in valid], dtype=float)
    return {
        "hc_donor": hc_k,
        "magnitude": magnitude,
        "gt": [gt_bs, gt_bc],
        "n": len(valid),
        "recovered_bs_median": float(np.median(bs)),
        "recovered_bc_median": float(np.median(bc)),
        "recovered_bs_iqr": float(np.percentile(bs, 75) - np.percentile(bs, 25)),
        "recovered_bc_iqr": float(np.percentile(bc, 75) - np.percentile(bc, 25)),
        "bias_bs": float(np.median(bs) - gt_bs),
        "bias_bc": float(np.median(bc) - gt_bc),
        "distance_from_GT": float(
            np.sqrt((np.median(bs) - gt_bs) ** 2
                    + (np.median(bc) - gt_bc) ** 2)),
        "frac_within_5deg": float(np.mean(
            (np.abs(bs - gt_bs) < 5.0) & (np.abs(bc - gt_bc) < 5.0))),
        "frac_within_10deg": float(np.mean(
            (np.abs(bs - gt_bs) < 10.0) & (np.abs(bc - gt_bc) < 10.0))),
        "raw_beta_s": bs.tolist(),
        "raw_beta_c": bc.tolist(),
        "records": records,
    }


# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m-realizations", type=int, default=M_REALIZATIONS)
    parser.add_argument("--magnitudes", type=str, default=None,
                        help="Comma-separated magnitudes (default: 1.0).")
    parser.add_argument("--candidates", type=str, default="all")
    parser.add_argument("--hc-subset", type=str, default=None,
                        help="Comma-separated HC ids (e.g. sub-01,sub-02).")
    args = parser.parse_args(argv)

    cand_filter = (None if args.candidates == "all"
                   else set(args.candidates.split(",")))
    cands = [c for c in CANDIDATES
             if cand_filter is None or c["id"] in cand_filter]

    if args.magnitudes:
        mags = [float(x) for x in args.magnitudes.split(",")]
    else:
        mags = list(MAGNITUDES)

    hc_subset = (HC_ALL if args.hc_subset is None
                 else [h.strip() for h in args.hc_subset.split(",")])

    print(f"[param_recovery_voxel] candidates={[c['id'] for c in cands]}",
          flush=True)
    print(f"  magnitudes={mags}  HCs={hc_subset}  M={args.m_realizations}",
          flush=True)

    print("[param_recovery_voxel] pre-loading data...", flush=True)
    all_hc_amps, all_hc_jnd = _preload_data()
    original_hc_jnd_subjs = list(v6.HC_JND_SUBJS)

    out = {
        "config": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "seed": RNG_SEED,
            "m_realizations": int(args.m_realizations),
            "magnitudes": mags,
            "candidates": [c["id"] for c in cands],
            "hc_subset": hc_subset,
            "subset_size": v6.SUBSET_SIZE,
            "n_resamples": v6.N_RESAMPLES,
            "spatial_cov_rank": int(SPATIAL_COV_RANK),
            "temporal_ar1_rho": float(AR1_RHO),
            "synth_provenance": synth_provenance(),
            "ROI_K": dict(ROI_K),
            "note": (
                "Voxel-level synthesis through HC_k's W with HC_k residual "
                "covariance + AR(1) — replaces Method C (loss-level injection)."
            ),
            "synth_jnd_consistent": True,
            "synth_jnd_method": (
                "fake_jnd = (HC \\ donor pool baseline) * d_phys/d_perc(GT) + "
                "N(0, pool_sd_per_pair). Replaces v1 behaviour of using donor's "
                "REAL JND (which biases γ-atom toward δ=0 when GT≠0)."
            ),
        },
        "cells": {},
    }

    t_all = time.time()
    for cand in cands:
        for mag in mags:
            for hc_k in hc_subset:
                key = f"{cand['id']}_mag{mag}_donor{hc_k}"
                t0 = time.time()
                print(f"\n=== {key} GT=({cand['production_beta_s']*mag:+.1f},"
                      f"{cand['production_beta_c']*mag:+.1f}) ===",
                      flush=True)
                cell = run_cell(cand, mag, hc_k, all_hc_amps, all_hc_jnd,
                                original_hc_jnd_subjs,
                                m_realizations=args.m_realizations)
                cell["candidate"] = cand
                cell["elapsed_s"] = round(time.time() - t0, 1)
                out["cells"][key] = cell
                if cell.get("n", 0) > 0:
                    print(f"  → bs_med={cell['recovered_bs_median']:+.1f} "
                          f"bc_med={cell['recovered_bc_median']:+.1f} "
                          f"|bias|=({cell['bias_bs']:+.1f},"
                          f"{cell['bias_bc']:+.1f}) "
                          f"f5°={cell['frac_within_5deg']:.2f} "
                          f"f10°={cell['frac_within_10deg']:.2f} "
                          f"[{cell['elapsed_s']}s]", flush=True)

    out["elapsed_s"] = round(time.time() - t_all, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {OUT}  ({out['elapsed_s']}s total)", flush=True)


if __name__ == "__main__":
    main()
