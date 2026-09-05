#!/usr/bin/env python3
"""
Prior Analysis: Mean Activation Comparison (HC vs CVD)
======================================================
Run BEFORE phase2 SRM analysis to characterize activation-level differences.

Goals:
  (1) HC vs CVD group & individual differences in activation magnitude
  (2) Validate SRM: does activation level confound SRM disparity?

Outputs → results/activation_prior/
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase2_SRM_across_between/activation_prior_analysis.py  (development repository, commit 53c81c2)
# Role      : Activation-level metrics before SRM (S10).
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[3]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p


import numpy as np
import json
import os
from pathlib import Path
from scipy import stats
from itertools import combinations

# ── Config ──────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "phase1_preprocess_decoding" / "results" / "full_dataset_C010"
OUT_DIR = Path(__file__).parent / "results" / "activation_prior"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HC_SUBS = [f"sub-{i:02d}" for i in range(1, 8)]
CVD_SUBS = [f"sub-{i:02d}" for i in range(8, 11)]
ALL_SUBS = HC_SUBS + CVD_SUBS
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABELS = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
COLORS = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
CVD_TYPES = {"sub-08": "deutan", "sub-09": "protan", "sub-10": "deutan"}


def load_amplitudes(sub, roi, kind="raw"):
    """Load amplitude array (6 runs, 8 colors, n_voxels)."""
    return np.load(DATA_DIR / sub / roi / f"amplitudes_{kind}.npy")


def crawford_howell(patient_score, control_scores):
    """Crawford & Howell (1998) modified t-test for single case."""
    n = len(control_scores)
    mean_c = np.mean(control_scores)
    std_c = np.std(control_scores, ddof=1)
    if std_c == 0:
        return np.nan, np.nan
    t = (patient_score - mean_c) / (std_c * np.sqrt((n + 1) / n))
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)  # two-tailed
    effect = (patient_score - mean_c) / std_c  # Zcc
    return t, p, effect


def compute_metrics(amp):
    """Compute activation metrics from (6, 8, n_voxels) array.

    Returns dict with:
      - mean_act: grand mean activation
      - mean_abs_act: mean |activation| (signal strength)
      - color_means: per-color mean activation (8,)
      - color_selectivity: F-stat from 1-way ANOVA across colors
      - run_reliability: mean correlation of voxel patterns between runs
      - snr: mean / std across runs (per voxel, then averaged)
      - voxel_variance: mean variance across voxels (spatial spread)
    """
    n_runs, n_colors, n_vox = amp.shape

    # Basic activation
    mean_act = float(amp.mean())
    mean_abs_act = float(np.abs(amp).mean())

    # Per-color means (average over runs and voxels)
    color_means = amp.mean(axis=(0, 2))  # (8,)

    # Color selectivity: 1-way ANOVA across 8 colors
    # Use run-averaged voxel patterns
    run_avg = amp.mean(axis=0)  # (8, n_vox)
    color_groups = [run_avg[c, :] for c in range(n_colors)]
    try:
        f_stat, f_p = stats.f_oneway(*color_groups)
    except Exception:
        f_stat, f_p = np.nan, np.nan

    # Run reliability: mean pairwise correlation of color-mean patterns between runs
    run_color_means = amp.mean(axis=2)  # (6, 8) — color profile per run
    corrs = []
    for r1, r2 in combinations(range(n_runs), 2):
        c = np.corrcoef(run_color_means[r1], run_color_means[r2])[0, 1]
        if not np.isnan(c):
            corrs.append(c)
    run_reliability = float(np.mean(corrs)) if corrs else np.nan

    # SNR: per-voxel mean/std across runs, then median
    vox_run_means = amp.mean(axis=1)  # (6, n_vox) — mean across colors per run
    vox_mean = vox_run_means.mean(axis=0)  # (n_vox,)
    vox_std = vox_run_means.std(axis=0, ddof=1)  # (n_vox,)
    with np.errstate(divide='ignore', invalid='ignore'):
        vox_snr = np.abs(vox_mean) / vox_std
    snr = float(np.nanmedian(vox_snr))

    # Spatial variance: variance across voxels of the mean pattern
    spatial_var = float(run_avg.var(axis=1).mean())

    # Color modulation depth: max - min of color means
    modulation_depth = float(color_means.max() - color_means.min())

    return {
        "mean_act": mean_act,
        "mean_abs_act": mean_abs_act,
        "color_means": color_means.tolist(),
        "color_selectivity_F": float(f_stat) if not np.isnan(f_stat) else None,
        "color_selectivity_p": float(f_p) if not np.isnan(f_p) else None,
        "run_reliability": run_reliability,
        "snr_median": snr,
        "spatial_variance": spatial_var,
        "modulation_depth": modulation_depth,
        "n_voxels": int(n_vox),
    }


# ── Main ────────────────────────────────────────────────────────────────
print("=" * 70)
print("PRIOR ANALYSIS: Mean Activation Comparison (HC vs CVD)")
print("=" * 70)

all_results = {}

for roi in ROIS:
    label = ROI_LABELS[roi]
    print(f"\n{'─' * 50}")
    print(f"ROI: {label} (dir: {roi})")
    print(f"{'─' * 50}")

    # Compute metrics for all subjects
    subject_metrics = {}
    for sub in ALL_SUBS:
        amp = load_amplitudes(sub, roi, "raw")
        subject_metrics[sub] = compute_metrics(amp)

    # ── (1) Group comparison: HC vs CVD ──
    hc_vals = {k: [subject_metrics[s][k] for s in HC_SUBS]
               for k in ["mean_abs_act", "snr_median", "run_reliability",
                          "modulation_depth", "spatial_variance"]}
    cvd_vals = {k: [subject_metrics[s][k] for s in CVD_SUBS]
                for k in hc_vals}

    print(f"\n  {'Metric':<22} {'HC mean±sd':>16} {'CVD mean±sd':>16} {'t':>7} {'p':>7} {'d':>7}")
    print(f"  {'─'*22} {'─'*16} {'─'*16} {'─'*7} {'─'*7} {'─'*7}")

    group_tests = {}
    for k in hc_vals:
        hc = np.array(hc_vals[k])
        cvd = np.array(cvd_vals[k])
        # Welch's t-test (unequal variance, unequal n)
        t_val, p_val = stats.ttest_ind(hc, cvd, equal_var=False)
        # Cohen's d (pooled)
        pooled_std = np.sqrt(((len(hc)-1)*hc.std(ddof=1)**2 + (len(cvd)-1)*cvd.std(ddof=1)**2)
                             / (len(hc)+len(cvd)-2))
        d = (cvd.mean() - hc.mean()) / pooled_std if pooled_std > 0 else np.nan

        sig = "*" if p_val < 0.05 else "~" if p_val < 0.10 else ""
        print(f"  {k:<22} {hc.mean():>8.5f}±{hc.std():.4f} {cvd.mean():>8.5f}±{cvd.std():.4f} "
              f"{t_val:>7.3f} {p_val:>6.3f}{sig} {d:>6.2f}")

        group_tests[k] = {
            "hc_mean": float(hc.mean()), "hc_sd": float(hc.std(ddof=1)),
            "cvd_mean": float(cvd.mean()), "cvd_sd": float(cvd.std(ddof=1)),
            "t": float(t_val), "p": float(p_val), "d": float(d),
        }

    # ── (2) Individual CVD: Crawford & Howell ──
    print(f"\n  Individual CVD tests (Crawford & Howell 1998):")
    print(f"  {'Subject':<10} {'Type':<8} {'|Act|':>8} {'Zcc':>7} {'p':>7}  {'Modulation':>10} {'Zcc':>7} {'p':>7}")

    individual_tests = {}
    for sub in CVD_SUBS:
        cvd_type = CVD_TYPES[sub]
        # Test mean_abs_act
        hc_scores = [subject_metrics[s]["mean_abs_act"] for s in HC_SUBS]
        t1, p1, z1 = crawford_howell(subject_metrics[sub]["mean_abs_act"], hc_scores)
        # Test modulation_depth
        hc_mod = [subject_metrics[s]["modulation_depth"] for s in HC_SUBS]
        t2, p2, z2 = crawford_howell(subject_metrics[sub]["modulation_depth"], hc_mod)

        sig1 = "*" if p1 < 0.05 else "~" if p1 < 0.10 else ""
        sig2 = "*" if p2 < 0.05 else "~" if p2 < 0.10 else ""
        print(f"  {sub:<10} {cvd_type:<8} {subject_metrics[sub]['mean_abs_act']:>8.5f} "
              f"{z1:>7.2f} {p1:>6.3f}{sig1}  "
              f"{subject_metrics[sub]['modulation_depth']:>10.5f} {z2:>7.2f} {p2:>6.3f}{sig2}")

        individual_tests[sub] = {
            "cvd_type": cvd_type,
            "mean_abs_act": {"value": subject_metrics[sub]["mean_abs_act"],
                             "zcc": float(z1), "p": float(p1)},
            "modulation_depth": {"value": subject_metrics[sub]["modulation_depth"],
                                 "zcc": float(z2), "p": float(p2)},
        }

    # ── (3) Color tuning profiles ──
    print(f"\n  Color tuning profiles (mean activation × 10³):")
    print(f"  {'Subject':<10} " + " ".join(f"{c[:3]:>6}" for c in COLORS))
    for sub in ALL_SUBS:
        cm = np.array(subject_metrics[sub]["color_means"]) * 1000
        group_tag = "CVD" if sub in CVD_SUBS else "HC"
        print(f"  {sub:<10} " + " ".join(f"{v:>6.2f}" for v in cm) + f"  [{group_tag}]")

    # ── (4) Color selectivity (ANOVA F-stat) ──
    print(f"\n  Color selectivity (1-way ANOVA across voxels):")
    print(f"  {'Subject':<10} {'F':>8} {'p':>8} {'Sig':>5}")
    for sub in ALL_SUBS:
        f_s = subject_metrics[sub]["color_selectivity_F"]
        p_s = subject_metrics[sub]["color_selectivity_p"]
        if f_s is not None:
            sig = "***" if p_s < 0.001 else "**" if p_s < 0.01 else "*" if p_s < 0.05 else ""
            print(f"  {sub:<10} {f_s:>8.3f} {p_s:>8.4f} {sig:>5}")
        else:
            print(f"  {sub:<10} {'N/A':>8} {'N/A':>8}")

    # ── (5) Run reliability ──
    print(f"\n  Run reliability (mean pairwise r of color profiles):")
    for sub in ALL_SUBS:
        r = subject_metrics[sub]["run_reliability"]
        group_tag = "CVD" if sub in CVD_SUBS else "HC"
        print(f"  {sub:<10} r={r:>6.3f}  [{group_tag}]")

    all_results[label] = {
        "subject_metrics": subject_metrics,
        "group_tests": group_tests,
        "individual_tests": individual_tests,
    }

# ── (6) Cross-ROI: activation vs SRM disparity correlation ──
print(f"\n{'=' * 70}")
print("CROSS-VALIDATION: Activation metrics vs SRM disparity")
print(f"{'=' * 70}")

# Load SRM disparities from LOO-consistent results
srm_results_path = Path(__file__).parent / "results" / "loo_consistent" / "20260218_163819" / "loo_consistent_results.json"
if srm_results_path.exists():
    with open(srm_results_path) as f:
        srm_raw = json.load(f)
    srm_data = srm_raw["results"]  # keys: V1, V2, V3, hV4

    def get_all_disparities(roi_label):
        """Get per-subject SRM disparity from LOO-consistent results."""
        if roi_label not in srm_data:
            return None
        roi_res = srm_data[roi_label]
        disp = {}
        # HC: from hc_loo_disparities
        for s, v in roi_res.get("hc_loo_disparities", {}).items():
            disp[s] = v
        # CVD: from individual_cvd → cvd_score
        for s, v in roi_res.get("individual_cvd", {}).items():
            disp[s] = v.get("cvd_score", np.nan)
        return disp

    def correlate_metric_vs_srm(metric_name, metric_vals, roi_label):
        """Correlate a per-subject metric with SRM disparity."""
        disp = get_all_disparities(roi_label)
        if disp is None:
            return None
        act_arr, disp_arr = [], []
        for s in ALL_SUBS:
            if s in disp and not np.isnan(disp[s]):
                act_arr.append(metric_vals[s])
                disp_arr.append(disp[s])
        if len(act_arr) < 5:
            return None
        r, p = stats.pearsonr(act_arr, disp_arr)
        return {"r": float(r), "p": float(p), "n": len(act_arr)}

    for metric_name, metric_key in [
        ("|activation|", "mean_abs_act"),
        ("SNR", "snr_median"),
        ("n_voxels", "n_voxels"),
        ("modulation_depth", "modulation_depth"),
        ("run_reliability", "run_reliability"),
    ]:
        print(f"\n  Correlation: {metric_name} vs SRM disparity")
        print(f"  {'ROI':<6} {'r':>7} {'p':>7} {'n':>3} {'interpretation'}")
        for roi in ROIS:
            label = ROI_LABELS[roi]
            vals = {s: all_results[label]["subject_metrics"][s][metric_key] for s in ALL_SUBS}
            res = correlate_metric_vs_srm(metric_name, vals, label)
            if res:
                interp = "CONFOUND!" if res["p"] < 0.05 else "OK (independent)" if res["p"] > 0.10 else "borderline"
                sig = "*" if res["p"] < 0.05 else ""
                print(f"  {label:<6} {res['r']:>7.3f} {res['p']:>6.3f}{sig} {res['n']:>3} {interp}")
            else:
                print(f"  {label:<6} data not available")
else:
    print("  SRM results file not found — skipping cross-validation")

# ── (7) Multivariate: color profile similarity (correlation distance) ──
print(f"\n{'=' * 70}")
print("COLOR PROFILE SIMILARITY (correlation distance)")
print(f"{'=' * 70}")

for roi in ROIS:
    label = ROI_LABELS[roi]
    print(f"\n  {label}:")

    # Get color profiles (8-dim vector per subject)
    profiles = {}
    for sub in ALL_SUBS:
        profiles[sub] = np.array(all_results[label]["subject_metrics"][sub]["color_means"])

    # HC-HC, HC-CVD, CVD-CVD distances
    hc_hc_dists, hc_cvd_dists, cvd_cvd_dists = [], [], []

    for s1, s2 in combinations(HC_SUBS, 2):
        r = np.corrcoef(profiles[s1], profiles[s2])[0, 1]
        hc_hc_dists.append(1 - r)

    for s1 in HC_SUBS:
        for s2 in CVD_SUBS:
            r = np.corrcoef(profiles[s1], profiles[s2])[0, 1]
            hc_cvd_dists.append(1 - r)

    for s1, s2 in combinations(CVD_SUBS, 2):
        r = np.corrcoef(profiles[s1], profiles[s2])[0, 1]
        cvd_cvd_dists.append(1 - r)

    print(f"    HC-HC  distance: {np.mean(hc_hc_dists):.3f} ± {np.std(hc_hc_dists):.3f}")
    print(f"    HC-CVD distance: {np.mean(hc_cvd_dists):.3f} ± {np.std(hc_cvd_dists):.3f}")
    print(f"    CVD-CVD distance: {np.mean(cvd_cvd_dists):.3f} ± {np.std(cvd_cvd_dists):.3f}")

    # Permutation test: HC-CVD > HC-HC?
    obs_diff = np.mean(hc_cvd_dists) - np.mean(hc_hc_dists)
    all_dists = np.array(hc_hc_dists + hc_cvd_dists)
    n_hh = len(hc_hc_dists)
    rng = np.random.default_rng(42)
    n_perm = 10000
    perm_diffs = np.zeros(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(len(all_dists))
        perm_diffs[i] = all_dists[idx[n_hh:]].mean() - all_dists[idx[:n_hh]].mean()
    p_perm = (perm_diffs >= obs_diff).mean()
    sig = "*" if p_perm < 0.05 else ""
    print(f"    HC-CVD > HC-HC: diff={obs_diff:.3f}, p_perm={p_perm:.3f}{sig}")

# ── Save results ────────────────────────────────────────────────────────
# Convert for JSON serialization
save_data = {}
for label, data in all_results.items():
    save_data[label] = {
        "group_tests": data["group_tests"],
        "individual_tests": data["individual_tests"],
        "per_subject": {
            sub: {k: v for k, v in metrics.items()}
            for sub, metrics in data["subject_metrics"].items()
        },
    }

with open(OUT_DIR / "activation_prior_results.json", "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to: {OUT_DIR / 'activation_prior_results.json'}")

# ── Summary ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")
print("""
This prior analysis checks whether HC-CVD differences in activation MAGNITUDE
could confound the SRM geometric (pattern) analysis.

Key findings to check:
  1. If |activation| differs: CVD might have weaker/stronger signal
  2. If color selectivity differs: different tuning → different patterns
  3. If |activation| correlates with SRM disparity: potential confound
  4. If SNR differs: data quality disparity → SRM sensitivity difference
  5. If color profile similarity differs: activation-level group structure

Interpretation guide:
  - If activation metrics are SIMILAR: SRM disparity reflects genuine
    geometric (pattern) differences, not signal quality differences
  - If activation differs AND correlates with SRM: need to control for
    activation in SRM analysis (e.g., z-score normalization)
  - If color profiles differ: supports genuine perceptual differences
    at activation level, complementing SRM pattern findings
""")
