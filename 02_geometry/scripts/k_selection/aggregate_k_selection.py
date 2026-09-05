#!/usr/bin/env python3
"""
Formal k Aggregation for SRM Component Selection (2C)

Aggregates 28 fold-ROI k-selection results using mean rank method.
For each fold x ROI, ranks k candidates by each metric, then averages
ranks across 7 folds to select optimal k.

Input: 28 JSON files from 2C_optimal_k_selection/results/
Output: k_aggregation_results.json + console summary table

Usage:
    python aggregate_k_selection.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/phase2_SRM_across_between/validation/aggregate_k_selection.py  (development repository, commit 53c81c2)
# Role      : Mean-rank aggregation over folds and metrics -> k_aggregation_results.json.
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


import json
import numpy as np
from pathlib import Path
from scipy.stats import rankdata

# ============================================================================
# CONFIG
# ============================================================================

RESULTS_DIR = Path(__file__).resolve().parent / "2C_optimal_k_selection" / "results"
ROIS = ["V1", "V2", "V3", "hV4"]
ROI_FILE_MAP = {"V1": "V1", "V2": "V2", "V3": "V3", "hV4": "hV4"}
N_FOLDS = 7
K_CANDIDATES = [2, 3, 4, 5, 6]

# Metrics: higher is better for rdm_reliability and cross_subject_rdm_corr,
# lower is better for reconstruction_error
METRICS = {
    "rdm_reliability": "higher",
    "cross_subject_rdm_corr": "higher",
    "reconstruction_error": "lower"
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("Formal k Aggregation: Mean Rank Method")
    print(f"7 LOSO folds x 4 ROIs x k={{2,3,4,5,6}}")
    print("=" * 80)

    all_results = {}

    for roi in ROIS:
        print(f"\n--- {roi} ---")

        # Collect fold data
        fold_data = []
        for fold_idx in range(N_FOLDS):
            json_path = RESULTS_DIR / f"k_selection_fold{fold_idx}_{roi}_results.json"
            if not json_path.exists():
                print(f"  WARNING: Missing {json_path.name}")
                continue
            with open(json_path) as f:
                data = json.load(f)

            # Extract results for this ROI
            roi_key = roi if roi in data["results"] else list(data["results"].keys())[0]
            fold_results = data["results"][roi_key]["results_by_k"]
            fold_data.append(fold_results)

        if not fold_data:
            print(f"  SKIP: no data found")
            continue

        print(f"  Loaded {len(fold_data)}/{N_FOLDS} folds")

        # For each metric, compute ranks per fold, then average
        roi_result = {"n_folds": len(fold_data)}

        for metric_name, direction in METRICS.items():
            # Extract metric values per fold per k
            # shape: (n_folds, n_k_candidates)
            metric_matrix = np.zeros((len(fold_data), len(K_CANDIDATES)))

            for fi, fold in enumerate(fold_data):
                for ki, k in enumerate(K_CANDIDATES):
                    k_str = str(k)
                    if k_str in fold and metric_name in fold[k_str]:
                        metric_matrix[fi, ki] = fold[k_str][metric_name]
                    else:
                        metric_matrix[fi, ki] = np.nan

            # Rank per fold (1 = best)
            rank_matrix = np.zeros_like(metric_matrix)
            for fi in range(len(fold_data)):
                row = metric_matrix[fi]
                if direction == "higher":
                    # Higher is better -> rank descending (negate for rankdata)
                    rank_matrix[fi] = rankdata(-row, method='average')
                else:
                    # Lower is better -> rank ascending
                    rank_matrix[fi] = rankdata(row, method='average')

            # Mean rank across folds
            mean_ranks = np.mean(rank_matrix, axis=0)
            std_ranks = np.std(rank_matrix, axis=0, ddof=1)
            best_idx = np.argmin(mean_ranks)
            best_k = K_CANDIDATES[best_idx]

            # Also compute mean metric values across folds
            mean_values = np.nanmean(metric_matrix, axis=0)
            std_values = np.nanstd(metric_matrix, axis=0, ddof=1)

            print(f"\n  {metric_name} ({direction} is better):")
            for ki, k in enumerate(K_CANDIDATES):
                marker = " <-- BEST" if ki == best_idx else ""
                print(f"    k={k}: mean_rank={mean_ranks[ki]:.2f} (SD={std_ranks[ki]:.2f}), "
                      f"value={mean_values[ki]:.4f} (SD={std_values[ki]:.4f}){marker}")

            roi_result[metric_name] = {
                "best_k": int(best_k),
                "mean_ranks": {str(k): float(mean_ranks[ki]) for ki, k in enumerate(K_CANDIDATES)},
                "std_ranks": {str(k): float(std_ranks[ki]) for ki, k in enumerate(K_CANDIDATES)},
                "mean_values": {str(k): float(mean_values[ki]) for ki, k in enumerate(K_CANDIDATES)},
                "std_values": {str(k): float(std_values[ki]) for ki, k in enumerate(K_CANDIDATES)},
                "rank_matrix": rank_matrix.tolist()
            }

        # Composite rank: average of mean ranks across all 3 metrics
        composite_ranks = np.zeros(len(K_CANDIDATES))
        for metric_name in METRICS:
            for ki, k in enumerate(K_CANDIDATES):
                composite_ranks[ki] += roi_result[metric_name]["mean_ranks"][str(k)]
        composite_ranks /= len(METRICS)
        best_composite_idx = np.argmin(composite_ranks)
        best_composite_k = K_CANDIDATES[best_composite_idx]

        print(f"\n  COMPOSITE (mean of 3 metric ranks):")
        for ki, k in enumerate(K_CANDIDATES):
            marker = " <-- BEST" if ki == best_composite_idx else ""
            print(f"    k={k}: composite_rank={composite_ranks[ki]:.2f}{marker}")

        roi_result["composite"] = {
            "best_k": int(best_composite_k),
            "composite_ranks": {str(k): float(composite_ranks[ki]) for ki, k in enumerate(K_CANDIDATES)}
        }

        all_results[roi] = roi_result

    # ========================================================================
    # SUMMARY TABLE
    # ========================================================================

    print("\n" + "=" * 80)
    print("PUBLICATION TABLE: Optimal k Selection (Mean Rank, 7-fold LOSO CV)")
    print("=" * 80)

    header = (f"{'ROI':<5} | {'Best k':<7} | "
              f"{'RDM rel. rank':<15} | {'Cross-subj rank':<16} | {'Recon. rank':<14} | "
              f"{'Composite':<12} | {'Runner-up'}")
    print(header)
    print("-" * len(header))

    for roi in ROIS:
        if roi not in all_results:
            continue
        r = all_results[roi]
        best_k = r["composite"]["best_k"]
        comp = r["composite"]["composite_ranks"]

        # Find runner-up
        sorted_ks = sorted(comp.items(), key=lambda x: x[1])
        runner_up_k = int(sorted_ks[1][0])
        runner_up_rank = sorted_ks[1][1]

        rdm_rank = r["rdm_reliability"]["mean_ranks"][str(best_k)]
        cross_rank = r["cross_subject_rdm_corr"]["mean_ranks"][str(best_k)]
        recon_rank = r["reconstruction_error"]["mean_ranks"][str(best_k)]

        print(f"{roi:<5} | k={best_k:<5} | "
              f"{rdm_rank:.2f}            | {cross_rank:.2f}             | {recon_rank:.2f}           | "
              f"{comp[str(best_k)]:.2f}         | k={runner_up_k} ({runner_up_rank:.2f})")

    # Per-metric best k table
    print(f"\n{'ROI':<5} | {'rdm_reliability':<17} | {'cross_subj_rdm':<17} | {'recon_error':<15} | {'COMPOSITE':<12}")
    print("-" * 75)
    for roi in ROIS:
        if roi not in all_results:
            continue
        r = all_results[roi]
        rdm_k = r["rdm_reliability"]["best_k"]
        cross_k = r["cross_subject_rdm_corr"]["best_k"]
        recon_k = r["reconstruction_error"]["best_k"]
        comp_k = r["composite"]["best_k"]
        print(f"{roi:<5} | k={rdm_k:<15} | k={cross_k:<15} | k={recon_k:<13} | k={comp_k}")

    # Save results
    output_path = Path(__file__).resolve().parent / "k_aggregation_results.json"
    save_data = {
        "config": {
            "method": "mean_rank",
            "n_folds": N_FOLDS,
            "k_candidates": K_CANDIDATES,
            "metrics": list(METRICS.keys()),
            "source": str(RESULTS_DIR)
        },
        "results": all_results
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
