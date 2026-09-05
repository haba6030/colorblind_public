# 01_decoding — Hue classification (LORO) and hue interpolation (LOCO)

**Manuscript:** Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)

Leave-one-run-out (LORO) classification and leave-one-color-out (LOCO) interpolation with the hue-channel basis model (six half-wave-rectified squared-cosine channels). `loro_baseline.py` / `loco_baseline.py` run the six-decoder comparison of S5 on the SRM-aligned amplitudes; the main-text readouts use the Procrustes-aligned amplitudes through `common/loco_canonical.py`. `perm_adjacent_n7.py` is the 1,000-permutation colour-label null for the control interpolation gate. `validation_tests.py` holds the cross-participant encoder transfer (Mann-Whitney).

Open `01_decoding.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/loro_baseline.py` | Six-decoder LORO comparison (S5, tab:loro_decoders). Output: results/loro/{srm,procrustes}/sub-*_performance_raw.json |
| `scripts/loco_baseline.py` | Six-decoder LOCO comparison (S5, tab:loco_decoders). Output: results/loco/srm/sub-*_loco.json |
| `scripts/utils.py` | Decoder definitions and hyper-parameter defaults shared by the baselines. |
| `scripts/validation_tests.py` | Cross-participant encoder transfer HC->HC vs HC->CVD in SRM space (Results section 1, S5). Output: cross_subject_generalization.json |
| `scripts/run_cvd_cross_decoding.py` | Control-trained encoder applied to CVD participants (RT-7, HC-only SRM). |
| `scripts/loro_baseline_srm.sbatch` | SLURM driver (SRM space). |
| `scripts/loro_baseline_procrustes.sbatch` | SLURM driver (Procrustes space). |
| `scripts/loco_baseline_srm.sbatch` | SLURM driver (LOCO, SRM space). |
| `scripts/validation_tests.sbatch` | SLURM driver for validation_tests.py. |
| `scripts/perm_adjacent_n7.py` | Control interpolation gate: 1,000 per-participant colour-label permutations of LOCO adjacent accuracy, all four ROIs, n = 7. Output: results/perm_adjacent_n7.json + perm_n7_null_*.npy |
| `scripts/_repro_util.py` | Path and Crawford-Howell helpers used by perm_adjacent_n7.py (development-repository layout). |
| `scripts/_compute_paper_stats.py` | Single-case (Crawford-Howell) statistics for the decoding endpoints. |
| `scripts/analyze_per_color_loco.py` | Per-hue LOCO breakdown (hue vulnerability profile). |
| `scripts/permutation_test_loco.py` | Group-level subject-label permutation for LOCO (10,000 permutations). |
| `scripts/lambda_stability_loco.py` | Ridge-penalty grid stability (S6). Output: results/lambda_stability.json |
| `scripts/basis_comparison.py` | Basis-count ablation for the hue-channel model. |
| `scripts/basis_comparison_stats.py` | Statistics for the basis ablation. |
| `scripts/aggregate_basis_ablation.py` | Aggregates the basis ablation. |
| `scripts/validate_loro_loco_loso.py` | LORO / LOCO / LOSO validation of the ridge-GCV encoder (S7). |
| `scripts/step_a_fit_srm.py` | Forward-model pipeline step A (control SRM). |
| `scripts/step_b_group_prior.py` | Step B (group prior on channel weights). |
| `scripts/step_c_project_prior.py` | Step C (project prior to each participant). |
| `scripts/step_d_finetune.py` | Step D (participant-level fine-tuning). |
| `scripts/run_all.sh` | Runs steps A-D. |
| `scripts/figure_generators/generate_fig2.py` | Figure 4 (LORO / LOCO across V1-hV4). |
