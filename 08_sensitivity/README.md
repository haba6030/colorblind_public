# 08_sensitivity — Preprocessing sensitivity: the head-motion-corrected pipeline

**Manuscript:** Supplementary S2 (tab:motion_arms, tab:interp_arms, session-2 endpoints), S13 sign stability, S18 (tab:color_specificity, head-motion column)

Every neural endpoint of session 1 was recomputed on the head-motion-corrected pipeline (`00_preprocessing/scripts/hmc_v2/`) with every other element held fixed: the control interpolation gate (`perm_adjacent_arm.py`), eight-way classification (`loro_eightway_arm.py`), Procrustes disparity under both estimators (`loso_arm_wrapper.py` driving `02_geometry/scripts/rerun_loo_consistent.py`), the colour-correspondence permutation, and the session-2 endpoints (`exp2_endpoints_arms.json`). The manuscript reports both pipelines side by side and treats individual cells as descriptive.

Open `08_sensitivity.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/perm_adjacent_arm.py` | Control interpolation gate on a given pipeline arm -> perm_adjacent_arm_<arm>.json + null arrays. |
| `scripts/loro_eightway_arm.py` | LORO eight-way accuracy with two-tailed Crawford-Howell on both arms -> loro_eightway_arms.json |
| `scripts/loso_arm_wrapper.py` | Drives rerun_loo_consistent.py on an arm (DATA_DIR injection, sub-10 dropped) -> loso_two_arm_summary.json |
| `scripts/arm_agreement.py` | Agreement of the interpolation readout between the arms (Bland-Altman, ICC; descriptive only). |
| `scripts/boot_runs_arm.py` | Run-bootstrap confidence intervals of adjacent accuracy per arm. |
| `scripts/sub07_leaveout.py` | Control gate with and without the low-coverage control (sub-07). |
