# 00_preprocessing — Preprocessing, registration and response estimation

**Manuscript:** Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)

Functional runs cover occipital cortex only, so registration used header-initialised mutual information (`run_method3_header_mi_*.sbatch`). The primary pipeline resamples every volume once with a run-constant transform; the head-motion-corrected pipeline composes each volume's MCFLIRT matrix with that transform before the same single resampling (`hmc_reanalysis/`). `run_full_dataset_C010.py` fits the per-run GLM and writes the run x hue x voxel amplitude arrays (dataset token `C010`) that every later stage reads. The notebook loads the committed QC summary (framewise displacement, ROI coverage, ROI tSNR under both pipelines).

Open `00_preprocessing.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/run_method3_header_mi_all_subjects.sbatch` | Session-1 registration (header-initialised MI) for all participants. |
| `scripts/run_method3_header_mi_2nd.sbatch` | Session-2 registration, same method. |
| `scripts/run_method3_hmc_all_subjects.sbatch` | MCFLIRT motion-parameter estimation (session 1). |
| `scripts/run_method3_hmc_2nd.sbatch` | MCFLIRT motion-parameter estimation (session 2). |
| `scripts/add_motion_correction.sbatch` | Writes motion parameters for the QC summary. |
| `scripts/add_motion_correction_2nd.sbatch` | Same for session 2. |
| `scripts/deploy_2nd_preprocessing.sh` | Deploys the session-2 preprocessing jobs. |
| `scripts/motion_qc_summary.py` | Framewise displacement (Power 2012, r = 50 mm) per run / participant -> motion_qc_summary.json. |
| `scripts/analyze_registration_quality.py` | ROI coverage = atlas ROI intersected with the BOLD brain mask; GLM valid-voxel ratio. |
| `scripts/diagnose_registration_quality.py` | Registration diagnostics used for the visual inspection reported in S2. |
| `scripts/hmc_v2/run_c010_hmc_v2.sbatch` | Head-motion-corrected pipeline: compose MCFLIRT matrices with the normalisation transform, single resampling, then C010 amplitudes. |
| `scripts/hmc_v2/run_hmc_array.sbatch` | SLURM array driver for the head-motion-corrected pipeline (session 1). |
| `scripts/hmc_v2/run_hmc_array_exp2_harm.sbatch` | Session-2 harmonised + head-motion-corrected arm. |
| `scripts/hmc_v2/analyze_hmc.sh` | tSNR comparison between the two pipelines -> hmc_summary.csv. |
| `scripts/hmc_v2/identity_check.sh` | Checks that the composed transform reduces to the primary transform when the motion term is identity. |
| `scripts/hmc_v2/README.md` | Notes on the head-motion-corrected pipeline scripts. |
| `scripts/hmc_v2/run_c010_hmc_exp2.py` | Session-2 amplitude estimation on the head-motion-corrected arm. |
| `scripts/run_full_dataset_C010.py` | Per-run GLM -> (6 runs x 8 hues x voxels) amplitudes per ROI, plus run-to-run Procrustes alignment (dataset C010). |
| `scripts/roi_pipeline_selected_1202used.py` | ROI extraction (Wang atlas V1, V2, V3, hV4 in MNI space). |
| `scripts/run_C010_with_residuals.sbatch` | SLURM driver for the primary-pipeline amplitudes. |
| `scripts/run_all_C010_with_residuals.sh` | Batch wrapper for the SLURM driver. |
| `scripts/compute_srm_amplitudes.py` | SRM-aligned amplitudes (`amplitudes_srm.npy`) used by the cross-participant decoder comparison. |
