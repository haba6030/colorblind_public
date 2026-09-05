#!/usr/bin/env python3
"""
exp2 sub-08 — 2x2 decoder x task test.

decoders : OLS (alpha=0, B&H pseudoinverse)  vs  ridge-GCV
tasks    : ENCODING (voxel-pattern prediction rho)  vs  DECODING (LOCO interpolation
           adjacent + exact accuracy)
conditions: HC reference (n=7, full 6-run) | no-filter (sub-08 exp1) | window | optimal
per ROI (V1,V2,V3,V4=hV4), FE-6, leave-one-color-out, per-run decode.

Confirms the decoder dissociation (analysis/decoder-comparison.md) ON the exp2 data,
and gives the correct (alpha=0) decoding-accuracy numbers for the filter conditions.

Run on server:  conda activate nilearn; python exp2_decoder_2x2.py [--variant native|matched]
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp2_decoder_2x2.py  (development repository, commit 53c81c2)
# Role      : OLS vs GCV decoder 2x2 check per condition.
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

import sys, json, argparse, os
import numpy as np
from pathlib import Path

ROOT = Path("/scratch/connectome/haba6030/colorBlind")
sys.path.insert(0, str(ROOT / "analysis" / "phase4_forward_model" / "scripts"))
from utils_forward_model import create_basis_matrix, HUE_ANGLES
from loco_canonical import loco_forward_readouts  # shared canonical impl

HC_SUBJS = [f"sub-0{i}" for i in range(1, 8)]
ROIS = ["V1", "V2", "V3", "V4"]
K = 6
N_COLORS = 8
BASIS360 = create_basis_matrix(np.arange(360), K, basis_type="fe")   # (360, K)
C = create_basis_matrix(HUE_ANGLES, K, basis_type="fe")              # (8, K)


def loco_2task(amp, decoder):
    """Leave-one-color-out, one decoder -> (rho_mean, adj_mean, exact_mean).
    Delegates to the shared canonical (loco_canonical) so this 2x2 test and the
    production exp2_hc_likeness use ONE implementation."""
    r = loco_forward_readouts(amp, C, BASIS360, decoder=decoder,
                              tasks=("rho", "adj", "exact"))
    return float(r["rho"].mean()), float(r["adj"].mean()), float(r["exact"].mean())


def load(path):
    return np.load(path) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="native", choices=["native", "matched"])
    ap.add_argument("--subject", default="08", help="CVD subject id, e.g. 08 / 09")
    args = ap.parse_args()
    subj = f"sub-{args.subject}"
    EXP2 = ROOT / "derivatives" / ("full_dataset_C010_exp2" if args.variant == "native"
                                   else "full_dataset_C010_exp2_matched")
    HC_C010 = ROOT / "derivatives" / "full_dataset_C010"
    OUT = ROOT / "analysis" / "phase6_behavioral_analysis" / "exp2_neural" / "results"
    print(f"SUBJECT={subj}  VARIANT={args.variant}  EXP2={EXP2.name}")

    results = {}
    for roi in ROIS:
        # condition -> amp (n_runs, 8, V)
        conds = {}
        # HC reference: list of per-subject amps (averaged at the end)
        hc_amps = []
        for s in HC_SUBJS:
            a = load(HC_C010 / s / roi / "amplitudes_procrustes.npy")
            if a is None: continue
            if roi == "V4" and a.shape[2] < 20:    # sub-07 hV4 sparse
                continue
            hc_amps.append(a)
        conds["nofilter"] = load(HC_C010 / subj / roi / "amplitudes_procrustes.npy")
        conds["window"]   = load(EXP2 / subj / "window"  / roi / "amplitudes_procrustes.npy")
        conds["optimal"]  = load(EXP2 / subj / "optimal" / roi / "amplitudes_procrustes.npy")

        roi_res = {"roi": roi, "hc_n": len(hc_amps), "conditions": {}}

        # HC = mean over subjects of each (rho/adj/exact) per decoder
        for decoder in ["ols", "gcv"]:
            vals = np.array([loco_2task(a, decoder) for a in hc_amps])  # (n_hc, 3)
            m = vals.mean(axis=0); sd = vals.std(axis=0, ddof=1)
            roi_res.setdefault("HC", {})[decoder] = {
                "rho": float(m[0]), "adj": float(m[1]), "exact": float(m[2]),
                "rho_sd": float(sd[0]), "adj_sd": float(sd[1]), "exact_sd": float(sd[2]),
            }
        # sub-08 conditions
        for cond, amp in conds.items():
            if amp is None:
                continue
            roi_res["conditions"][cond] = {"n_runs": int(amp.shape[0]),
                                           "n_voxels": int(amp.shape[2])}
            for decoder in ["ols", "gcv"]:
                rho, adj, exa = loco_2task(amp, decoder)
                roi_res["conditions"][cond][decoder] = {"rho": rho, "adj": adj, "exact": exa}
        results[roi] = roi_res

    OUT.mkdir(parents=True, exist_ok=True)
    out_f = OUT / f"exp2_decoder_2x2_{subj}_{args.variant}.json"
    with open(out_f, "w") as f:
        json.dump(results, f, indent=2)

    # ---- print: per ROI, conditions as rows, decoder as columns ----
    print(f"\n{'='*92}\n{subj} exp2 — 2x2 decoder(ols/gcv) x task(ENCODING rho / DECODING adj,exact)\n"
          f"chance: adj=0.375  exact=0.125   (ENCODING wants gcv high; DECODING wants ols high)\n{'='*92}")
    for roi, r in results.items():
        print(f"\n[{roi}]  (HC n={r['hc_n']})")
        print(f"{'condition':12} | {'rho_ols':>8}{'rho_gcv':>8} | "
              f"{'adj_ols':>8}{'adj_gcv':>8} | {'exa_ols':>8}{'exa_gcv':>8}")
        rows = [("HC", r["HC"]["ols"], r["HC"]["gcv"])]
        for cond in ["nofilter", "window", "optimal"]:
            c = r["conditions"].get(cond)
            if c: rows.append((cond, c["ols"], c["gcv"]))
        for name, o, g in rows:
            print(f"{name:12} | {o['rho']:>8.3f}{g['rho']:>8.3f} | "
                  f"{o['adj']:>8.3f}{g['adj']:>8.3f} | {o['exact']:>8.3f}{g['exact']:>8.3f}")
    print(f"\nSaved: {out_f}")


if __name__ == "__main__":
    main()
