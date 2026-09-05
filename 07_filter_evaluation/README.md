# 07_filter_evaluation — Second-session evaluation of the filters (psychophysics and fMRI)

**Manuscript:** Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3

Both CVD participants repeated the JND and 8AFC tasks and were scanned under the deployed macOS accessibility filter and their individualized filter (four runs each, ABBA). `exp2_C010_conditions.py` estimates the condition-wise amplitudes; `exp2_hc_likeness.py` and `exp2_decoder_2x2.py` compute LORO / LOCO readouts against the run-matched control reference; `exp2_runmatched_geometry.py` rebuilds SRM disparity and RDM similarity over all C(6,4) run subsets; `analyze_exp2_behavior.py` and `analyze_exp2_jnd_vs_hc.py` handle the psychophysics; `run_count_adjacc.py` is the run-count adequacy analysis (Figure S2). All neural indices are single-case descriptive (d_cc); the notebook loads the committed condition-level outputs.

Open `07_filter_evaluation.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/analyze_exp2_behavior.py` | JND and 8AFC per condition, Wilcoxon between filters -> sub-*_summary.json |
| `scripts/analyze_exp2_jnd_vs_hc.py` | JND of every condition against the control distribution (z per pair) -> sub-*_jnd_vs_hc.json |
| `scripts/exp2_C010_conditions.py` | Condition-wise (window / optimal) amplitude estimation for session 2 (ABBA run map). |
| `scripts/exp2_hc_likeness.py` | LORO / LOCO readouts per condition vs the run-matched control reference -> exp2_hc_likeness_sub-*_matched.json |
| `scripts/exp2_decoder_2x2.py` | OLS vs GCV decoder 2x2 check per condition. |
| `scripts/exp2_convergent.py` | SRM disparity and shared-space RDM similarity per condition (BrainIAK) -> exp2_convergent_sub-*_matched.json |
| `scripts/exp2_runmatched_geometry.py` | Run-matched geometry over all C(6,4) subsets -> exp2_runmatched_geometry_sub-*_matched.json (tab:exp2_geometry) |
| `scripts/exp2_geometry_derived.py` | Derived geometry indices per condition. |
| `scripts/srm_disp_floor_loo.py` | Control leave-one-out floor for SRM disparity. |
| `scripts/srm_rdm_floor_loo.py` | Control leave-one-out self-consistency for RDM similarity. |
| `scripts/exp2_qc_figures_fixed.py` | Session-2 preprocessing QC figures. |
| `scripts/exp1_qc_figures.py` | Session-1 preprocessing QC figures. |
| `scripts/run_exp2_C010.sbatch` | SLURM driver (amplitudes). |
| `scripts/run_exp2_analysis.sbatch` | SLURM driver (hc_likeness). |
| `scripts/run_exp2_convergent.sbatch` | SLURM driver (convergent geometry). |
| `scripts/run_exp2_runmatch.sbatch` | SLURM driver (run-matched geometry). |
| `scripts/run_decoder_2x2.sbatch` | SLURM driver (decoder 2x2). |
| `scripts/run_count_adjacc.py` | Run-count adequacy of LOCO adjacent accuracy over all C(6,n) subsets (S14, Figure S2) -> adjacc_retention_summary.json |
| `scripts/figure_generators/generate_fig8.py` | Figure 7 (interpolation and geometry under each filter). |
| `scripts/figure_generators/generate_figS2_adjacc_saturation.py` | Figure S2 (adjacent accuracy vs run count). |
| `scripts/figure_generators/generate_figS3_forward_tuning.py` | Figure S3 (forward-tuning rho under each filter). |
