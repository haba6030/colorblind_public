# Map: reported value -> result file -> producing script

For each stage, the committed result files the notebook reads and the script (in `<stage>/scripts/` unless a path is given) that produced them. Check ids are in `MANIFEST.md`; the notebook headers repeat this table.

## 00_preprocessing

| Result file | Producing script | Content |
|---|---|---|
| `results/preprocessing_qc_summary.json` | `scripts/motion_qc_summary.py, scripts/analyze_registration_quality.py, scripts/hmc_v2/analyze_hmc.sh` | framewise displacement, ROI coverage, ROI tSNR under both pipelines |
| `results/bbr_vs_mi_displacement.json` | `(fMRIPrep BBR vs header-initialised MI, server QC)` | slab centroid displacement between the two registration routes |

## 01_decoding

| Result file | Producing script | Content |
|---|---|---|
| `results/loro/procrustes/sub-*_performance_raw.json` | `scripts/loro_baseline.py` | LORO eight-way accuracy per fold, Procrustes-aligned amplitudes (main text, tab:alignment) |
| `results/loro/srm/sub-*_performance_raw.json` | `scripts/loro_baseline.py` | LORO per fold, SRM-aligned amplitudes, six decoders (tab:loro_decoders) |
| `results/loco/srm/sub-*_loco.json` | `scripts/loco_baseline.py` | LOCO adjacent accuracy, SRM-aligned amplitudes, six decoders (tab:loco_decoders) |
| `results/adjacent_accuracy_per_hue.json` | `common/loco_canonical.py via tools export` | per-hue LOCO adjacent / exact accuracy, Procrustes space, both pipelines (Results section 1, Figure 4B-C, tab:effect_sizes, tab:alignment) |
| `results/perm_adjacent_n7.json, perm_n7_null_*.npy` | `scripts/perm_adjacent_n7.py` | control interpolation gate: 1,000 colour-label permutations, n = 7 (tab:interp_arms, primary) |
| `results/cross_subject_generalization.json` | `scripts/validation_tests.py` | control-trained encoder applied to held-out controls (28 cells) and to CVD participants (12 cells incl. sub-10) |
| `results/nested_procrustes/nested_only/` | `scripts/loro_baseline.py (nested variant)` | leakage bound of the fixed-reference alignment (S7); ten participants incl. sub-10 |
| `results/lambda_stability.json` | `scripts/lambda_stability_loco.py` | ridge-penalty grid (S6) |

## 02_geometry

| Result file | Producing script | Content |
|---|---|---|
| `results/loo_consistent_results.json` | `scripts/rerun_loo_consistent.py (BrainIAK, mpirun -np 1)` | SRM k, common-space and symmetric-LOSO disparity with Crawford-Howell tests (Figure 5, tab:disparity_loso) |
| `results/k_aggregation_results.json` | `scripts/k_selection/aggregate_k_selection.py` | mean-rank aggregation over folds and metrics (S4) |
| `results/crossnobis_results.json, pca_cca_results.json, ve_results.json` | `scripts/triangulation/*.py` | alignment-independent distances and variance explained (S9); files include sub-10, the notebook drops it |
| `results/overall_signal_results.json, activation_prior_results.json` | `scripts/activation/*.py` | activation-level metrics with single-case tests (S10) |
| `results/disparity_frozen_permutation_primary.json` | `scripts/color_specificity/disparity_frozen_permutation.py` | colour-correspondence permutation, frozen vs re-estimated projection (S18) |
| `results/color_correspondence_heldout.json` | `scripts/color_specificity/color_correspondence_heldout.py` | split-half RDM reliability (S18) |
| `results/cyclic_shift_disparity.json, shift_gain_ch.json` | `scripts/color_specificity/shift_gain_ch.py` | cyclic hue-shift disparity and gain test (S18) |

## 03_psychophysics

| Result file | Producing script | Content |
|---|---|---|
| `results/hc_group_metrics.json` | `scripts/compute_hc_group_metrics.py` | control threshold mean / SD per pair and the deutan participant's session-1 thresholds |
| `results/sub-09_jnd_vs_hc.json / .csv` | `scripts/analyze_exp2_jnd_vs_hc.py` | protan session-1 thresholds ('nofilter') with z against the controls |
| `results/a2_staircase_diagnosis.json` | `scripts/a2_staircase_diagnosis.py` | every staircase: censoring at ceiling, high/low-start estimates and their spread (tab:staircase_pairs) |

## 04_distortion_model

| Result file | Producing script | Content |
|---|---|---|
| `results/s10b_v6_pca_rdm_summary_sub-0{8,9}.json` | `scripts/s10b_v6_pca_rdm.py` | every loss combination x model over N = 300 resamples, PCA basis (production); slimmed copy, `summary` verbatim |
| `results/s10b_v6_srm_rdm_summary_sub-0{8,9}.json` | `scripts/s10b_v6_srm_rdm.py` | same with the RDM atom in the SRM basis |
| `results/precondition_table.json` | `scripts/s10a_precondition.py` | Gate 1 separation d per atom and ROI |
| `results/s19_allcandidate_heldout.json, s18_heldout_predictive.json, s17_hc_loo_results.json` | `scripts/s19_allcandidate_heldout.py, s18_heldout_predictive.py, s17_hc_loo.py` | strict 7-fold held-out statistics and neural-term ablation (tab:fit_stability) |
| `results/beta_sign_three_arms.json` | `scripts/filter_robustness_arms.py` | selected combinations refitted on the head-motion-corrected pipeline (S13) |
| `scripts/two_comp.py, rc_1dof.py, s8_loo_train_test.py` | `(imported)` | grid definition, R+C gain range, Delta-lambda anchors |

## 05_identifiability

| Result file | Producing script | Content |
|---|---|---|
| `results/param_recovery_voxel_v6_pca_v2.json` | `scripts/param_recovery_voxel.py` | Test 1: 7 donors x 20 noise draws at the selected optimum |
| `results/null_within_hc_loo_v6_pca.json` | `scripts/null_within_hc_loo.py` | Test 2a (origin null, B2) and Test 2b (control pseudo-CVD, B1) |
| `results/null_label_permutation_v6_pca.json` | `scripts/null_label_permutation_block.py` | Test 2c: 1,000 colour-label permutations |
| `results/verdict_matrix_v6_pca_v2.json` | `scripts/analyze_verification.py` | criteria, verdicts and BH correction |

## 06_filter

| Result file | Producing script | Content |
|---|---|---|
| `results/sub-0{8,9}_2component_preimage.json` | `scripts/exp2_compute_preimage.py` | fitted parameters, stimulus angles, pre-image angles and per-hue corrections |
| `../04_distortion_model/scripts/two_comp.py` | `(imported)` | forward two-component model used to verify the pre-image |

## 07_filter_evaluation

| Result file | Producing script | Content |
|---|---|---|
| `results/behavior/sub-0{8,9}_summary.json` | `scripts/analyze_exp2_behavior.py` | JND means per condition, Wilcoxon between filters, 8AFC counts |
| `results/behavior/sub-0{8,9}_jnd_vs_hc.json` | `scripts/analyze_exp2_jnd_vs_hc.py` | z of every condition's threshold against the controls, per pair |
| `results/neural/exp2_hc_likeness_sub-0{8,9}_matched.json` | `scripts/exp2_hc_likeness.py` | LORO / LOCO readouts per condition vs the run-matched control reference (tab:exp2_loro, tab:exp2_geometry, Figure S3) |
| `results/neural/exp2_runmatched_geometry_sub-0{8,9}_matched.json` | `scripts/exp2_runmatched_geometry.py` | SRM disparity and RDM similarity, all C(6,4) subsets (tab:exp2_geometry) |
| `results/neural/exp2_convergent_sub-09_matched.json` | `scripts/exp2_convergent.py` | unmatched geometry (S14 matching comparison) |
| `results/adjacc_retention_summary.json` | `scripts/run_count_adjacc.py` | run-count adequacy (S14, Figure S2) |

## 08_sensitivity

| Result file | Producing script | Content |
|---|---|---|
| `results/loso_two_arm_summary.json` | `scripts/loso_arm_wrapper.py -> ../02_geometry/scripts/rerun_loo_consistent.py` | Crawford-Howell and LOSO disparity on both pipelines (tab:motion_arms) |
| `results/perm_adjacent_arm_{primary,head_motion_correction}.json + null arrays` | `scripts/perm_adjacent_arm.py` | control interpolation gate and single-case hV4 contrasts per pipeline (tab:interp_arms) |
| `results/loro_eightway_arms.json` | `scripts/loro_eightway_arm.py` | LORO eight-way accuracy per pipeline with two-tailed single-case tests (tab:interp_arms) |
| `results/disparity_frozen_permutation_head_motion_correction.json` | `../02_geometry/scripts/color_specificity/disparity_frozen_permutation.py` | colour-correspondence grid, head-motion column of tab:color_specificity |
| `results/exp2_endpoints_arms.json` | `(exp2 pipeline rerun on the harmonised + head-motion-corrected arm)` | session-2 endpoints under both arms (S2) |
| `results/disparity_individual_arms.json` | `(frozen wrapper stdout, parsed)` | disparity per arm, descriptive |
