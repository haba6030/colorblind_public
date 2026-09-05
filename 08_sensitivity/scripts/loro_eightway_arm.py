
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 08_sensitivity
# Manuscript: Supplementary S2 (tab:motion_arms, tab:interp_arms, session-2 endpoints), S13 sign stability, S18 (tab:color_specificity, head-motion column)
# Original  : analysis/future_phase1_sensitivity/scripts/_loro_eightway_arm.py  (development repository, commit 53c81c2)
# Role      : LORO eight-way accuracy with two-tailed Crawford-Howell on both arms -> loro_eightway_arms.json
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

# LORO eight-way classification across the two reported preprocessing pipelines.
#
# Companion to _perm_adjacent_arm.py, which does the LOCO half. S1 stated that
# "every neural endpoint was recomputed" under head-motion correction, but the
# LORO classification readout was never recomputed, so the categorical half of
# the dissociation had no arm table. This driver closes that gap and supplies
# the LORO panel of Supplementary tab:interp_arms.
#
# The readout is NOT reimplemented here. The published LORO column of
# tab:alignment comes from the phase3 decoder comparison
# (ForwardEncoding acc_exact, Procrustes-aligned, six folds), so this driver
# calls that same script on each arm tree and only aggregates. A hand-rolled
# FE-6/OLS reimplementation was tried first and does NOT reproduce the
# published values (HC V2 0.568 vs 0.607, deutan V3 0.354 vs 0.396), so the
# phase3 driver is the only correct source.
#
# Reproduction gate: the with_residuals arm must match the published
# tab:alignment LORO column exactly (HC 0.580/0.607/0.574/0.488;
# deutan 0.562/0.521/0.396/0.375; protan 0.562/0.562/0.458/0.375).
# assert_reproduces() fails loudly otherwise.
#
# Test is Crawford & Howell TWO-tailed, matching Methods: the hypothesis for
# classification is preservation, whereas the interpolation tests are one-tailed.
#
# Usage:
#   python _loro_eightway_arm.py            # runs both arms, writes results JSON
#   python _loro_eightway_arm.py --skip-run # aggregate existing per-arm output
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
ANALYSIS = ROOT / "analysis"
VIS = ANALYSIS / "phase1_procrustes_decoding/results/visualization"
DRIVER = ANALYSIS / "phase3_decoder_comparing/model_comparison_validation/scripts/loro_baseline.py"
OUT = Path(__file__).resolve().parent.parent / "results"
WORK = OUT / "_loro_arm_runs"

HC = ["01", "02", "03", "04", "05", "06", "07"]
CVD = {"08": "deutan", "09": "protan"}
ROIS = ["V1", "V2", "V3", "V4"]
ARMS = {"with_residuals": "Primary", "hmc_v2": "Realignment"}

PUBLISHED = {  # Supplementary tab:alignment, Procrustes-aligned LORO column
    "V1": (0.580, 0.562, 0.562),
    "V2": (0.607, 0.521, 0.562),
    "V3": (0.574, 0.396, 0.458),
    "V4": (0.488, 0.375, 0.375),
}


def run_arm(arm):
    for sub in HC + list(CVD):
        subprocess.run(
            [sys.executable, str(DRIVER),
             "--baseline_dir", str(VIS / f"full_dataset_C010_{arm}"),
             "--output_dir", str(WORK / arm),
             "--subject", sub, "--rois", *ROIS,
             "--models", "ForwardEncoding", "--alignment", "procrustes"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def acc(arm, sub, roi):
    p = WORK / arm / f"sub-{sub}_performance_raw.json"
    folds = json.load(open(p))["results"]["procrustes"][roi]["ForwardEncoding"]
    return float(np.mean([f["acc_exact"] for f in folds]))


def summarize(arm):
    out = {}
    for roi in ROIS:
        per = [acc(arm, s, roi) for s in HC]
        m, sd = float(np.mean(per)), float(np.std(per, ddof=1))
        cell = {"hc_mean": round(m, 4), "hc_sd": round(sd, 4)}
        for sub, label in CVD.items():
            v = acc(arm, sub, roi)
            t = (v - m) / (sd * np.sqrt((len(HC) + 1) / len(HC)))
            cell[label] = {"accuracy": round(v, 4), "t": round(float(t), 3),
                           "p_two_tailed": round(float(2 * stats.t.cdf(-abs(t), len(HC) - 1)), 4)}
        out["hV4" if roi == "V4" else roi] = cell
    return out


def assert_reproduces(summary):
    bad = []
    for roi, (hc, deu, pro) in PUBLISHED.items():
        c = summary["hV4" if roi == "V4" else roi]
        got = (c["hc_mean"], c["deutan"]["accuracy"], c["protan"]["accuracy"])
        if not np.allclose(got, (hc, deu, pro), atol=6e-4):
            bad.append((roi, got, (hc, deu, pro)))
    if bad:
        raise SystemExit(f"reproduction gate FAILED: {bad}")
    print("  reproduction gate: EXACT MATCH against tab:alignment", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-run", action="store_true")
    args = ap.parse_args()

    result = {
        "metric": "LORO eight-way classification accuracy "
                  "(ForwardEncoding acc_exact, Procrustes-aligned, 6 folds)",
        "chance": 0.125,
        "test": "Crawford-Howell two-tailed (hypothesis = preservation)",
        "source_driver": str(DRIVER.relative_to(ROOT)),
        "arms": {},
    }
    for arm, label in ARMS.items():
        print(f"=== arm: {arm} ({label}) ===", flush=True)
        if not args.skip_run:
            run_arm(arm)
        s = summarize(arm)
        if arm == "with_residuals":
            assert_reproduces(s)
        result["arms"][label] = s

    result["min_cvd_cell"] = {
        label: min(min(c[l]["accuracy"] for l in CVD.values()) for c in a.values())
        for label, a in result["arms"].items()
    }
    dest = OUT / "loro_eightway_arms.json"
    json.dump(result, open(dest, "w"), indent=1)
    print(f"written {dest}")
    print("min CVD cell:", result["min_cvd_cell"])
