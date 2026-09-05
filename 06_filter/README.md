# 06_filter — Per-participant stimulus-space filter (pre-image of the fitted distortion)

**Manuscript:** Results section 6; Figure 6; Methods 'Stimulus-space filter and its evaluation'

For each displayed hue theta_k the filter solves T(theta_pre) = theta_k for the fitted transform T (`exp2_compute_preimage.py`, Brent root finding); the per-hue correction is theta_pre - theta_k. `stim_lab_render.py` renders the corrected discs in CIELab for the PsychoPy display and for Figure 6. The notebook re-applies the forward model to the committed pre-image angles and confirms that every hue resolves to its target within 1e-3 degrees.

Open `06_filter.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/exp2_compute_preimage.py` | Numerical pre-image of the fitted two-component transform -> sub-*_2component_preimage.json |
| `scripts/stim_lab_render.py` | CIELab rendering of the original and filtered discs. |
| `scripts/figure_generators/generate_fig7_filter.py` | Figure 6 (original vs filtered stimuli, both participants). |
