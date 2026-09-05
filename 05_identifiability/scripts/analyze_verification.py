"""analyze_verification.py — Aggregation + verdict matrix for v6 PCA RDM verification.

Reads outputs from:
  - param_recovery_voxel_v6_pca.json   (Script 4)
  - null_label_permutation_v6_pca.json (Script 3; also any sliced *_pNNNN-MMMM.json)
  - null_within_hc_loo_v6_pca.json     (Script 2)

For each candidate, computes four verdicts:

  1. Identifiability (Script 4 @ magnitude=1.0):
       - bias_bs_median, bias_bc_median across HC carriers
       - IQR across HC carriers
       - frac_within_10° (median across HCs)
       - PASS if frac_within_10° ≥ 0.5 AND |bias_bs|<10° AND |bias_bc|<10°

  2. Within-subject SIG (Script 3):
       - Real production loss_at_argmin obtained from
         results/s10_inclusion/s10b_v6_pca_rdm_results_{subject}.json
         (the production run with N_RESAMPLES=300; we use median train_loss
         at the chosen combo as the real reference).
       - p_perm = #{perm_loss ≤ real_loss} / N_PERM
       - PASS if p_perm < 0.05

  3. Specificity vs HC heterogeneity (Script 2 B1):
       - For N_real = 1 production estimate, percentile rank against the
         7 HC null argmins (β_s and β_c separately, plus distance from origin).
       - PASS if real β_s rank > 6/7 (i.e., real is at or beyond every HC null,
         one-sided high) AND likewise for |β_c| AND distance from origin.

  4. Algorithm validation (Script 4 @ magnitude=0):
       - PASS if median recovered (β_s, β_c) satisfies |bias|<5° in both axes.

FDR (Benjamini-Hochberg) is applied across the 3 candidates × 3 main tests
(identifiability, within-subject SIG, specificity) = 9 tests.

Output:
  - results/redteam/verdict_matrix_v6_pca.json
  - results/redteam/verdict_matrix.md (table-only markdown)

DO NOT execute via this header. Run after Scripts 2/3/4 complete.
"""
from __future__ import annotations

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 05_identifiability
# Manuscript: Methods 'Identifiability and recovery'; Supplementary S12 (tab:identifiability); S13 control leave-one-out magnitude anchor
# Original  : analysis/phase5_filter_optimization/scripts/analyze_verification.py  (development repository, commit 53c81c2)
# Role      : Verdict matrix and BH correction over the six test-bearing checks -> verdict_matrix_v6_pca_v2.json
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
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REDTEAM_DIR = SCRIPT_DIR.parent / "results" / "redteam"
INCLUSION_DIR = SCRIPT_DIR.parent / "results" / "s10_inclusion"

RECOVERY_JSON = REDTEAM_DIR / "param_recovery_voxel_v6_pca.json"
PERM_JSON = REDTEAM_DIR / "null_label_permutation_v6_pca.json"
LOO_JSON = REDTEAM_DIR / "null_within_hc_loo_v6_pca.json"

OUT_JSON = REDTEAM_DIR / "verdict_matrix_v6_pca.json"
OUT_MD = REDTEAM_DIR / "verdict_matrix.md"

CANDIDATES = [
    {"id": "S08-stable",  "subject": "sub-08", "combo_label": "γALL|RDMV1|noLOCO",
     "production_beta_s": 38.0, "production_beta_c": -10.0},
    {"id": "S08-robust",  "subject": "sub-08", "combo_label": "γOY|RDMV2|noLOCO",
     "production_beta_s":  6.0, "production_beta_c": -42.0},
    {"id": "S09-primary", "subject": "sub-09", "combo_label": "γALL|RDMV1|noLOCO",
     "production_beta_s":  2.0, "production_beta_c":  24.0},
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return json.loads(path.read_text())


def _load_perm_json_with_slices() -> dict:
    """Load primary PERM_JSON; if not found, attempt slice merging."""
    if PERM_JSON.exists():
        return _load_json(PERM_JSON)
    slices = sorted(REDTEAM_DIR.glob("null_label_permutation_v6_pca_p*-*.json"))
    if not slices:
        raise FileNotFoundError(
            f"No permutation JSON found: neither {PERM_JSON} "
            f"nor sliced *_p####-####.json")
    merged: dict = {"config": None, "cells": {}}
    for path in slices:
        d = _load_json(path)
        if merged["config"] is None:
            merged["config"] = d.get("config")
        for cid, cell in d.get("cells", {}).items():
            if cid not in merged["cells"]:
                merged["cells"][cid] = {"records": [], "summary": None}
            merged["cells"][cid]["records"].extend(cell.get("records", []))
            # keep last candidate metadata
            merged["cells"][cid]["candidate"] = cell.get("candidate")
    # Re-summarize merged cells
    for cid, cell in merged["cells"].items():
        valid = [r for r in cell["records"]
                 if "error" not in r and r.get("beta_s") is not None]
        if not valid:
            cell["summary"] = {"n": 0}
            continue
        losses = np.array([r["train_loss"] for r in valid
                            if r["train_loss"] is not None], dtype=float)
        bs = np.array([r["beta_s"] for r in valid], dtype=float)
        bc = np.array([r["beta_c"] for r in valid], dtype=float)
        cell["summary"] = {
            "n": len(valid),
            "beta_s_median": float(np.median(bs)),
            "beta_c_median": float(np.median(bc)),
            "raw_beta_s": bs.tolist(),
            "raw_beta_c": bc.tolist(),
            "raw_losses": losses.tolist(),
        }
    return merged


def _load_production_loss(subject: str, combo_label: str) -> Optional[float]:
    """Read median train_loss for combo from s10b_v6_pca_rdm_results_{subject}.json.

    Returns None if file or combo is missing.
    """
    path = INCLUSION_DIR / f"s10b_v6_pca_rdm_results_{subject}.json"
    if not path.exists():
        return None
    data = _load_json(path)
    storage = data.get("storage", {})
    block = storage.get(combo_label)
    if block is None:
        return None
    recs = block.get("2comp", [])
    losses = [r["train_loss"] for r in recs
              if r.get("train_loss") is not None]
    if not losses:
        return None
    return float(np.median(losses))


def _load_production_argmin(subject: str, combo_label: str) -> Optional[tuple]:
    """Read median (β_s, β_c) production argmin for the candidate."""
    path = INCLUSION_DIR / f"s10b_v6_pca_rdm_results_{subject}.json"
    if not path.exists():
        return None
    data = _load_json(path)
    storage = data.get("storage", {})
    block = storage.get(combo_label)
    if block is None:
        return None
    recs = block.get("2comp", [])
    if not recs:
        return None
    bs = np.array([r["beta_s"] for r in recs], dtype=float)
    bc = np.array([r["beta_c"] for r in recs], dtype=float)
    return float(np.median(bs)), float(np.median(bc))


# ---------------------------------------------------------------------------
# Verdict builders
# ---------------------------------------------------------------------------
def verdict_identifiability(recovery_data: dict, candidate: dict) -> dict:
    """Aggregate HC cells at magnitude=1.0 for candidate."""
    cells = [c for cid, c in recovery_data.get("cells", {}).items()
             if cid.startswith(f"{candidate['id']}_mag1.0_")]
    if not cells:
        return {"PASS": False, "reason": "no recovery cells @ mag=1.0",
                "frac_within_10deg_median": None}
    bias_bs = [c.get("bias_bs") for c in cells if c.get("bias_bs") is not None]
    bias_bc = [c.get("bias_bc") for c in cells if c.get("bias_bc") is not None]
    f10 = [c.get("frac_within_10deg") for c in cells
           if c.get("frac_within_10deg") is not None]
    f5 = [c.get("frac_within_5deg") for c in cells
          if c.get("frac_within_5deg") is not None]
    bs_med = float(np.median(bias_bs)) if bias_bs else None
    bc_med = float(np.median(bias_bc)) if bias_bc else None
    f10_med = float(np.median(f10)) if f10 else None
    f5_med = float(np.median(f5)) if f5 else None
    bs_iqr = (float(np.percentile(bias_bs, 75) - np.percentile(bias_bs, 25))
              if len(bias_bs) >= 2 else None)
    bc_iqr = (float(np.percentile(bias_bc, 75) - np.percentile(bias_bc, 25))
              if len(bias_bc) >= 2 else None)
    PASS = bool(f10_med is not None and f10_med >= 0.5
                and bs_med is not None and abs(bs_med) < 10.0
                and bc_med is not None and abs(bc_med) < 10.0)
    return {
        "PASS": PASS,
        "bias_bs_median": bs_med, "bias_bc_median": bc_med,
        "bias_bs_iqr": bs_iqr, "bias_bc_iqr": bc_iqr,
        "frac_within_10deg_median": f10_med,
        "frac_within_5deg_median": f5_med,
        "n_HC_carriers": len(cells),
    }


def verdict_within_subject_sig(perm_data: dict, candidate: dict) -> dict:
    """Compute one-sided p_perm against real production loss_at_argmin."""
    cell = perm_data.get("cells", {}).get(candidate["id"])
    if cell is None:
        return {"PASS": False, "reason": "no permutation cell",
                "p_perm": None}
    summary = cell.get("summary", {})
    perm_losses = np.array(summary.get("raw_losses", []), dtype=float)
    if perm_losses.size == 0:
        return {"PASS": False, "reason": "empty perm losses",
                "p_perm": None}
    real_loss = _load_production_loss(candidate["subject"],
                                       candidate["combo_label"])
    if real_loss is None:
        return {"PASS": False, "reason": "real production loss unavailable",
                "p_perm": None, "n_perm": int(perm_losses.size)}
    p_perm = float((np.sum(perm_losses <= real_loss) + 1) /
                   (perm_losses.size + 1))  # +1 conservative
    return {
        "PASS": bool(p_perm < 0.05),
        "p_perm": p_perm,
        "n_perm": int(perm_losses.size),
        "real_loss": float(real_loss),
        "perm_loss_5pct": float(np.percentile(perm_losses, 5)),
        "perm_loss_median": float(np.median(perm_losses)),
    }


def verdict_specificity(loo_data: dict, candidate: dict) -> dict:
    """Compare real CVD argmin to within-HC LOO null (B1) via percentile rank.

    For N_real=1: report rank_bs and rank_bc as fraction of HC null at or
    beyond the real value (one-sided high). PASS if real exceeds every HC
    null (i.e. rank ≤ 1/(n+1) under the alt-large convention).
    """
    cell = loo_data.get("cells", {}).get(candidate["id"])
    if cell is None:
        return {"PASS": False, "reason": "no LOO cell",
                "rank_bs": None, "rank_bc": None}
    summary = cell.get("B1_summary", {})
    bs_null = np.array(summary.get("raw_beta_s", []), dtype=float)
    bc_null = np.array(summary.get("raw_beta_c", []), dtype=float)
    if bs_null.size == 0:
        return {"PASS": False, "reason": "empty B1 raw"}
    real = _load_production_argmin(candidate["subject"],
                                    candidate["combo_label"])
    if real is None:
        return {"PASS": False, "reason": "real production argmin unavailable"}
    real_bs, real_bc = real
    real_dist = float(np.sqrt(real_bs ** 2 + real_bc ** 2))
    hc_dist = np.sqrt(bs_null ** 2 + bc_null ** 2)
    rank_bs = float((np.sum(bs_null >= real_bs) + 1) / (bs_null.size + 1))
    rank_bc_extreme = float(
        (np.sum(np.abs(bc_null) >= abs(real_bc)) + 1)
        / (bc_null.size + 1))
    rank_dist = float((np.sum(hc_dist >= real_dist) + 1)
                      / (hc_dist.size + 1))
    # Most-extreme verdict: real strictly exceeds every HC null on distance.
    PASS = bool(np.sum(hc_dist >= real_dist) == 0
                 and np.sum(bs_null >= real_bs) == 0)
    return {
        "PASS": PASS,
        "rank_bs": rank_bs, "rank_bc_extreme": rank_bc_extreme,
        "rank_distance": rank_dist,
        "real_bs": real_bs, "real_bc": real_bc, "real_distance": real_dist,
        "hc_bs_max": float(np.max(bs_null)),
        "hc_bc_absmax": float(np.max(np.abs(bc_null))),
        "hc_distance_max": float(np.max(hc_dist)),
        "n_hc_null": int(bs_null.size),
    }


def verdict_algo_validation(recovery_data: dict, candidate: dict) -> dict:
    """At magnitude=0, recovered should be ≈ (0,0). |bias|<5° in both axes."""
    cells = [c for cid, c in recovery_data.get("cells", {}).items()
             if cid.startswith(f"{candidate['id']}_mag0.0_")]
    if not cells:
        return {"PASS": False, "reason": "no recovery cells @ mag=0.0",
                "median_bs": None}
    bs = [c.get("recovered_bs_median") for c in cells
          if c.get("recovered_bs_median") is not None]
    bc = [c.get("recovered_bc_median") for c in cells
          if c.get("recovered_bc_median") is not None]
    bs_med = float(np.median(bs)) if bs else None
    bc_med = float(np.median(bc)) if bc else None
    PASS = bool(bs_med is not None and abs(bs_med) < 5.0
                and bc_med is not None and abs(bc_med) < 5.0)
    return {
        "PASS": PASS,
        "median_bs_at_gt0": bs_med,
        "median_bc_at_gt0": bc_med,
        "n_HC_carriers": len(cells),
    }


# ---------------------------------------------------------------------------
# FDR (Benjamini-Hochberg)
# ---------------------------------------------------------------------------
def bh_fdr(pvals: list, alpha: float = 0.05) -> list:
    """Return list of bool indicating BH-FDR significant at level α."""
    p_arr = np.array([np.nan if p is None else p for p in pvals], dtype=float)
    keep = ~np.isnan(p_arr)
    n = int(keep.sum())
    if n == 0:
        return [False] * len(pvals)
    order = np.argsort(p_arr[keep])
    sorted_p = p_arr[keep][order]
    thresh = alpha * (np.arange(1, n + 1) / n)
    passed = sorted_p <= thresh
    if not passed.any():
        sig_count = 0
    else:
        sig_count = int(np.max(np.where(passed)[0]) + 1)
    sig_mask = np.zeros(n, dtype=bool)
    sig_mask[order[:sig_count]] = True
    out = [False] * len(pvals)
    j = 0
    for i, k in enumerate(keep):
        if k:
            out[i] = bool(sig_mask[j])
            j += 1
    return out


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def render_markdown(verdicts: dict) -> str:
    lines = []
    lines.append("# Phase B v6 PCA RDM — Verification verdict matrix")
    lines.append("")
    lines.append(f"_Generated: {datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append("| Candidate | Identifiability | Within-subj SIG | Specificity vs HC | Algorithm validation | FDR-sig (BH α=0.05) |")
    lines.append("|---|---|---|---|---|---|")
    for cid, v in verdicts["per_candidate"].items():
        ident = v["identifiability"]
        sig = v["within_subject_sig"]
        spec = v["specificity"]
        algo = v["algorithm_validation"]
        ident_str = (f"{'PASS' if ident['PASS'] else 'FAIL'} "
                     f"(f10={ident.get('frac_within_10deg_median')}, "
                     f"bias=({ident.get('bias_bs_median')},"
                     f"{ident.get('bias_bc_median')}))")
        sig_str = (f"{'PASS' if sig['PASS'] else 'FAIL'} "
                   f"(p_perm={sig.get('p_perm')}, n={sig.get('n_perm')})")
        spec_str = (f"{'PASS' if spec['PASS'] else 'FAIL'} "
                    f"(rank_dist={spec.get('rank_distance')})")
        algo_str = (f"{'PASS' if algo['PASS'] else 'FAIL'} "
                    f"(median@GT0=({algo.get('median_bs_at_gt0')},"
                    f"{algo.get('median_bc_at_gt0')}))")
        fdr = v["fdr_significant"]
        fdr_str = (f"ident={fdr.get('identifiability')}, "
                   f"sig={fdr.get('within_subject_sig')}, "
                   f"spec={fdr.get('specificity')}")
        lines.append(f"| **{cid}** | {ident_str} | {sig_str} | {spec_str} | "
                     f"{algo_str} | {fdr_str} |")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Identifiability uses param_recovery_voxel @ mag=1.0; "
                 "PASS = frac_within_10° ≥ 0.5 AND |bias|<10° both axes.")
    lines.append("- Within-subject SIG uses null_label_permutation with the "
                 "production loss median as the real reference.")
    lines.append("- Specificity uses null_within_hc_loo B1 (real HC as fake CVD); "
                 "PASS = real CVD exceeds every HC null on β_s AND distance "
                 "from origin (one-sided high, percentile rank).")
    lines.append("- Algorithm validation uses param_recovery_voxel @ mag=0.0; "
                 "PASS = |bias|<5° in both axes.")
    lines.append("- FDR (Benjamini-Hochberg) is applied across 3 candidates × 3 "
                 "main tests (identifiability, sig, specificity) = 9 tests at α=0.05.")
    lines.append("")
    lines.append("## §0 framework reminder")
    lines.append("Specificity result is **descriptive only** per project §0; "
                 "the verdict matrix is a diagnostic, not a selection criterion.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fdr-alpha", type=float, default=0.05)
    parser.add_argument("--recovery-json", type=str, default=None,
                        help="Override recovery JSON path (e.g. v2 file).")
    parser.add_argument("--out-suffix", type=str, default="",
                        help="Suffix for output JSON/MD (e.g. _v2).")
    args = parser.parse_args(argv)

    global RECOVERY_JSON, OUT_JSON, OUT_MD  # noqa: PLW0603
    if args.recovery_json:
        RECOVERY_JSON = Path(args.recovery_json)
    if args.out_suffix:
        OUT_JSON = OUT_JSON.with_name(OUT_JSON.stem + args.out_suffix + OUT_JSON.suffix)
        OUT_MD = OUT_MD.with_name(OUT_MD.stem + args.out_suffix + OUT_MD.suffix)

    print("[analyze_verification] loading...", flush=True)
    print(f"  recovery: {RECOVERY_JSON}", flush=True)
    recovery = _load_json(RECOVERY_JSON) if RECOVERY_JSON.exists() else {"cells": {}}
    try:
        perm = _load_perm_json_with_slices()
    except FileNotFoundError:
        perm = {"cells": {}}
        print(f"  WARNING: no permutation results; SIG verdict will be FAIL/no-data.",
              flush=True)
    loo = _load_json(LOO_JSON) if LOO_JSON.exists() else {"cells": {}}

    verdicts = {
        "config": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "fdr_alpha": float(args.fdr_alpha),
            "candidates": [c["id"] for c in CANDIDATES],
            "inputs": {
                "recovery_json": str(RECOVERY_JSON),
                "permutation_json": str(PERM_JSON),
                "loo_json": str(LOO_JSON),
            },
        },
        "per_candidate": {},
    }

    # Collect p-values for FDR. We map verdicts onto "pseudo p-values" as follows:
    #   identifiability: p = 1 - frac_within_10deg_median (lower = better)
    #   within_subject_sig: p_perm directly
    #   specificity: rank_distance (lower = better; one-sided high test)
    pvals_for_fdr: list = []
    pvals_index: list = []   # (cid, test_name)

    for cand in CANDIDATES:
        ident = verdict_identifiability(recovery, cand)
        sig = verdict_within_subject_sig(perm, cand)
        spec = verdict_specificity(loo, cand)
        algo = verdict_algo_validation(recovery, cand)

        p_ident = (1.0 - ident["frac_within_10deg_median"]
                   if ident.get("frac_within_10deg_median") is not None
                   else None)
        p_sig = sig.get("p_perm")
        p_spec = spec.get("rank_distance")

        pvals_for_fdr.extend([p_ident, p_sig, p_spec])
        pvals_index.extend([(cand["id"], "identifiability"),
                            (cand["id"], "within_subject_sig"),
                            (cand["id"], "specificity")])

        verdicts["per_candidate"][cand["id"]] = {
            "candidate": cand,
            "identifiability": ident,
            "within_subject_sig": sig,
            "specificity": spec,
            "algorithm_validation": algo,
            "fdr_significant": {},  # filled below
        }

    # Apply BH-FDR across the 9 tests
    fdr_results = bh_fdr(pvals_for_fdr, alpha=args.fdr_alpha)
    for (cid, test), sig_bool, p in zip(pvals_index, fdr_results,
                                          pvals_for_fdr):
        verdicts["per_candidate"][cid]["fdr_significant"][test] = {
            "p_value_proxy": p, "BH_significant": bool(sig_bool),
        }

    # Save outputs
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(verdicts, indent=2, default=str))
    OUT_MD.write_text(render_markdown(verdicts))
    print(f"Saved: {OUT_JSON}", flush=True)
    print(f"Saved: {OUT_MD}", flush=True)


if __name__ == "__main__":
    main()
