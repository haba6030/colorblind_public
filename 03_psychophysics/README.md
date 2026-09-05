# 03_psychophysics — Session-1 hue-discrimination thresholds (JND) and identification (8AFC)

**Manuscript:** Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)

Interleaved staircases (two per hue pair) give the discrimination threshold t; gamma is the ratio to the control mean and z the deviation in control standard deviations (n = 7). `compute_hc_group_metrics.py` builds the control distribution, `a2_staircase_diagnosis.py` audits every staircase (censoring, track agreement) and `_staircase_pairs_table.py` prints tab:staircase_pairs from that audit. Trial-level behavioural files are not distributed; the notebook works from the per-participant summaries.

Open `03_psychophysics.ipynb` for the checks against the printed values. The table below lists the scripts in `scripts/` with the role each played; `results/` holds the committed outputs the notebook reads.

| Script | Role |
|---|---|
| `scripts/compute_hc_group_metrics.py` | Control JND distribution per hue pair -> hc_group_metrics.json. |
| `scripts/update_jnd_summary.py` | Tabulates hc_group_metrics.json -> jnd_summary.csv. |
| `scripts/compute_rsvp_metrics.py` | 8AFC identification accuracy (session 1) -> rsvp_summary.csv. |
| `scripts/a2_staircase_diagnosis.py` | Staircase audit: censoring at ceiling, track spread -> a2_staircase_diagnosis.json. |
| `scripts/staircase_pairs_table.py` | Renders tab:staircase_pairs (LaTeX) from the audit. |
| `scripts/jnd_noise_floor.py` | Split-half noise floor of the JND estimates. |
| `scripts/analyze_exp2_jnd_vs_hc.py` | JND of every condition against the control distribution (z per pair; also used in 07). |
