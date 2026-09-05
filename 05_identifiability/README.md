# 05_identifiability — Identifiability and recovery of the fitted parameters

**Manuscript:** Methods 'Identifiability and recovery'; Supplementary S12 (tab:identifiability); S13 control leave-one-out magnitude anchor

Four pre-specified checks on the selected fits with PCA-basis voxel synthesis: Test 1 parameter recovery at the selected optimum (`param_recovery_voxel.py`), Test 2a noise floor at a (0, 0) ground truth and Test 2b control pseudo-CVD refits (`null_within_hc_loo.py`), Test 2c colour-label permutation (`null_label_permutation_block.py`); `analyze_verification.py` assembles the verdict matrix with Benjamini-Hochberg correction. These scripts import the model and loss code of `04_distortion_model/scripts`.

Open `05_identifiability.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/forward_voxel_synth.py` | Synthetic voxel responses at a known distortion with donor-matched noise (PCA(20) spatial covariance, AR(1) = 0.3). |
| `scripts/param_recovery_voxel.py` | Test 1: recovery fraction within 10 deg and bias -> param_recovery_voxel_v6_pca_v2.json |
| `scripts/null_within_hc_loo.py` | Test 2a (origin null) and Test 2b (control pseudo-CVD) -> null_within_hc_loo_v6_pca.json |
| `scripts/null_label_permutation_block.py` | Test 2c: 1,000 colour-label permutations of the CVD data -> null_label_permutation_v6_pca.json |
| `scripts/analyze_verification.py` | Verdict matrix and BH correction over the six test-bearing checks -> verdict_matrix_v6_pca_v2.json |
| `scripts/run_param_recovery.sbatch` | SLURM driver (Test 1). |
| `scripts/run_null_within_hc_loo.sbatch` | SLURM driver (Tests 2a, 2b). |
| `scripts/run_null_label_perm.sbatch` | SLURM driver (Test 2c). |
