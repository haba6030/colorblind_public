# 04_distortion_model — Cortical distortion model: inverse fitting and parameter selection

**Manuscript:** Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability

The two-component model delta_theta = beta_s cos(theta - 90) + beta_c cos(theta - theta_conf) is fitted on a 1,326-cell grid by minimising a z-scored composite of loss atoms: psychophysical JND (`behav_loss.py`), Delta-RDM in a PCA(6) space (`neural_loss.py`) and hV4 LOCO voxel prediction. `s10b_v6_pca_rdm.py` runs every loss combination over N = 300 control 5-train/2-test resamples and applies the three gates; `s17_hc_loo.py` and `s18_heldout_predictive.py` / `s19_allcandidate_heldout.py` give the strict 7-fold leave-one-control-out statistics; `s10b_v6_srm_rdm.py` repeats the procedure with the RDM atom in the SRM basis; `rc_1dof.py` + `machado_simulator.py` implement the retinal-family comparison model. The committed `s10b_*_summary_*.json` files are slimmed copies of the 13-87 MB originals (the `summary` block is verbatim; per-resample fits are kept for the reported combinations only).

Open `04_distortion_model.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/two_comp.py` | Two-component forward model, grid definition (BS_GRID, BC_GRID), theta_conf per subtype. |
| `scripts/rc_1dof.py` | Retinal-plus-cortical-gain comparison model (one free gain g in [0, 3]). |
| `scripts/utils_distortion_models.py` | Distortion application helpers shared by both model classes. |
| `scripts/machado_simulator.py` | Machado (2009) cone-shift hue displacement on Stockman-Sharpe fundamentals. |
| `scripts/stockman_cone_shift.py` | Stockman & Sharpe cone fundamentals and the spectral-shift computation used by the Machado simulator. |
| `scripts/behav_loss.py` | Psychophysical JND loss atom L_gamma. |
| `scripts/neural_loss.py` | Delta-RDM loss atom L_RDM (cosine dissimilarity in PCA(6) space). |
| `scripts/diagnostic_delta_rdm.py` | Delta-RDM computation and simulated Delta-RDM under a candidate distortion (imported by neural_loss.py). |
| `scripts/s8_loo_train_test.py` | JND baseline from a control pool; Delta-lambda anchors per subtype. |
| `scripts/s10a_precondition.py` | Gate 1 separation precondition (Cohen's d of each atom vs the control LOO distribution) -> precondition_table.json |
| `scripts/s10b_v6_pca_rdm.py` | Production fit: all loss combinations x {2-component, R+C at three anchors} over N = 300 resamples -> s10b_v6_pca_rdm_results_sub-*.json |
| `scripts/s10b_v6_srm_rdm.py` | Same with the RDM atom in the SRM basis (requires BrainIAK). |
| `scripts/s17_hc_loo.py` | Strict 7-fold leave-one-control-out refit of the candidates -> s17_hc_loo_results.json |
| `scripts/s18_heldout_predictive.py` | Held-out predictive check and neural-term ablation -> s18_heldout_predictive.json |
| `scripts/s19_allcandidate_heldout.py` | Held-out statistics for every candidate (grid percentile, Delta-L, folds) -> s19_allcandidate_heldout.json |
| `scripts/filter_robustness_arms.py` | Refits the selected combinations on the head-motion-corrected pipeline (S13 sign stability) -> beta_sign_three_arms.json |
| `scripts/viz_closure_ground_plot.py` | Loss-landscape reconstruction on the full control pool (Figure S1). |
| `scripts/figure_generators/generate_figS1_landscape.py` | Figure S1 (loss landscape with resample argmins). |
