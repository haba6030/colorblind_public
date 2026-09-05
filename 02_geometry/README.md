# 02_geometry — Representational geometry: SRM, Procrustes disparity and its validity checks

**Manuscript:** Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)

`rerun_loo_consistent.py` fits the control-only shared response model (BrainIAK SRM, k = 4/4/3/3) with leave-one-control-out references and computes Procrustes disparity in the common space and under the symmetric leave-one-subject-out estimator, with Crawford-Howell single-case tests. The `validation/` scripts implement the alignment-independent distances (crossnobis, PCA, PCA-CCA), the colour-correspondence permutation with a frozen projection, the cyclic-shift analysis, and the activation-level comparison. BrainIAK requires `mpirun -np 1 python ...`; the notebook therefore loads the committed outputs and recomputes only the aggregate statistics (Spearman correlations, Hedges g, BH correction).

Open `02_geometry.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/rerun_loo_consistent.py` | Canonical SRM + disparity pipeline (common space and symmetric LOSO). Output: loo_consistent_results.json. Requires BrainIAK (mpirun -np 1). |
| `scripts/k_selection/run_k_selection_cv.py` | LOSO cross-validation over k = 2..6 (S4). |
| `scripts/k_selection/run_k_selection_cv.sbatch` | SLURM driver for the k selection. |
| `scripts/k_selection/aggregate_k_selection.py` | Mean-rank aggregation over folds and metrics -> k_aggregation_results.json. |
| `scripts/triangulation/compute_crossnobis_rdm.py` | Cross-validated Mahalanobis RDM in native voxel space (S9). Reads residuals_2nd_level.npy (not distributed). |
| `scripts/triangulation/compute_pca_cca_replication.py` | PCA and PCA-CCA pairwise alignment replication (S9). |
| `scripts/triangulation/compute_variance_explained.py` | SRM variance explained under the symmetric LOSO projection (S9, tab:variance_explained). |
| `scripts/activation/activation_prior_analysis.py` | Activation-level metrics before SRM (S10). |
| `scripts/activation/plot_overall_signal.py` | Consolidates the activation metrics with single-case tests -> overall_signal_results.json (S10). |
| `scripts/color_specificity/disparity_frozen_permutation.py` | Colour-correspondence permutation with frozen vs re-estimated projection (S18, tab:frozen_control, tab:color_specificity). |
| `scripts/color_specificity/color_correspondence_loro.py` | LORO-frozen colour-correspondence permutation. |
| `scripts/color_specificity/color_correspondence_heldout.py` | Split-half colour correspondence and within-participant RDM reliability (S18). |
| `scripts/color_specificity/frozen_disparity_label_check.py` | Label-permutation check of the frozen disparity. |
| `scripts/color_specificity/individual_color_label_permutation.py` | Individual-level colour-label permutation of disparity (both estimators). |
| `scripts/color_specificity/color_alignment_z.py` | Alignment z-scores across metrics. |
| `scripts/color_specificity/color_alignment_group_prereq.py` | Group-level prerequisite for the alignment test. |
| `scripts/color_specificity/within_hc_reliability.py` | Within-control RDM reliability in SRM, PCA and voxel spaces. |
| `scripts/color_specificity/basis_sensitivity_filter.py` | Sensitivity of the filter to the RDM basis. |
| `scripts/color_specificity/shift_gain_ch.py` | Cyclic hue-shift gain with Crawford-Howell test against the control gain distribution (S18). |
| `scripts/figure_generators/generate_fig3_geometry_r6.py` | Figure 5 (Procrustes disparity per participant and ROI). |
