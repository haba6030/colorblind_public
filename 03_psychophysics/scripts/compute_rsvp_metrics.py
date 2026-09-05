"""
Compute RSVP 8AFC metrics from the unified behavior pilot dataset.

Data source: project_root/data/behavior/sub-{01,03,06,07,08}_rsvp_8afc_ses1_run1.csv
  - HC RSVP: sub-01, sub-03, sub-06, sub-07 (N=4; sub-02/04/05 have no RSVP run)
  - CVD: sub-08

Outputs:
  results/rsvp_summary.csv    — per-subject overall accuracy/RT/timeouts + HC group mean + CVD comparison
  results/rsvp_per_color.csv  — per-color accuracy (8 colors) with HC mean±SEM vs CVD

Notes:
  - `correct` column is 0/1; timeouts recorded with rt<=0 (project convention).
  - Color label uses `color_1..color_8` mapping to red/orange/yellow/green/cyan/blue/purple/magenta.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 03_psychophysics
# Manuscript: Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)
# Original  : analysis/phase6_behavioral_analysis/scripts/compute_rsvp_metrics.py  (development repository, commit 53c81c2)
# Role      : 8AFC identification accuracy (session 1) -> rsvp_summary.csv.
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[2]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p

import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent
PHASE_DIR = SCRIPT_DIR.parent
REPO_ROOT = PHASE_DIR.parent.parent
BEHAVIOR_DIR = REPO_ROOT / "data" / "behavior"
RESULTS_DIR = PHASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

HC_RSVP_SUBJECTS = ["sub-01", "sub-03", "sub-06", "sub-07"]
CVD_SUBJECT = "sub-08"

COLOR_NAMES = {
    1: 'red', 2: 'orange', 3: 'yellow', 4: 'green',
    5: 'cyan', 6: 'blue', 7: 'purple', 8: 'magenta',
}


def load_rsvp(subject_id):
    path = BEHAVIOR_DIR / f"{subject_id}_rsvp_8afc_ses1_run1.csv"
    return pd.read_csv(path)


def overall_metrics(df):
    n_total = len(df)
    n_timeout = int((df['rt'] <= 0).sum())
    n_negative_rt = int((df['rt'] < 0).sum())
    n_correct = int(df['correct'].sum())
    acc = 100.0 * n_correct / n_total if n_total > 0 else np.nan
    correct_df = df[(df['correct'] == 1) & (df['rt'] > 0)]
    mean_rt = float(correct_df['rt'].mean()) if len(correct_df) else np.nan
    return {
        'accuracy_pct': round(acc, 3),
        'n_correct': n_correct,
        'n_total': n_total,
        'mean_rt_correct_s': round(mean_rt, 4),
        'n_timeout': n_timeout,
        'n_negative_rt': n_negative_rt,
    }


def per_color_accuracy(df):
    # stimulus_label like 'color_3' → idx 3
    df = df.copy()
    df['color_idx'] = df['stimulus_label'].str.replace('color_', '', regex=False).astype(int)
    rows = []
    for cid in range(1, 9):
        sub = df[df['color_idx'] == cid]
        n = len(sub)
        n_correct = int(sub['correct'].sum())
        rows.append({
            'color_id': cid,
            'color_name': COLOR_NAMES[cid],
            'n_trials': n,
            'n_correct': n_correct,
            'accuracy': round(n_correct / n, 4) if n else np.nan,
        })
    return pd.DataFrame(rows)


hc_overall = {sid: overall_metrics(load_rsvp(sid)) for sid in HC_RSVP_SUBJECTS}
hc_per_color = {sid: per_color_accuracy(load_rsvp(sid)) for sid in HC_RSVP_SUBJECTS}

cvd_df = load_rsvp(CVD_SUBJECT)
cvd_overall = overall_metrics(cvd_df)
cvd_per_color = per_color_accuracy(cvd_df)

# ── rsvp_summary.csv ──
metrics_keys = ['accuracy_pct', 'n_correct', 'n_total',
                'mean_rt_correct_s', 'n_timeout', 'n_negative_rt']
summary_rows = []
for key in metrics_keys:
    row = {'metric': key}
    hc_vals = []
    for sid in HC_RSVP_SUBJECTS:
        v = hc_overall[sid][key]
        row[sid] = v
        if key in ('accuracy_pct', 'mean_rt_correct_s'):
            hc_vals.append(v)
    if key == 'accuracy_pct':
        row['hc_group_mean'] = round(float(np.mean(hc_vals)), 3)
        row['hc_group_sd'] = round(float(np.std(hc_vals, ddof=1)), 3)
    elif key == 'mean_rt_correct_s':
        row['hc_group_mean'] = round(float(np.mean(hc_vals)), 4)
        row['hc_group_sd'] = round(float(np.std(hc_vals, ddof=1)), 4)
    else:
        row['hc_group_mean'] = int(sum(hc_overall[s][key] for s in HC_RSVP_SUBJECTS))
        row['hc_group_sd'] = ''
    row[CVD_SUBJECT] = cvd_overall[key]
    # Diff column
    if key in ('accuracy_pct', 'mean_rt_correct_s'):
        row['diff_cvd_hc'] = round(cvd_overall[key] - row['hc_group_mean'], 4)
    else:
        row['diff_cvd_hc'] = ''
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
col_order = ['metric'] + HC_RSVP_SUBJECTS + ['hc_group_mean', 'hc_group_sd', CVD_SUBJECT, 'diff_cvd_hc']
summary_df = summary_df[col_order]
summary_path = RESULTS_DIR / "rsvp_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"Saved: {summary_path}")

# ── rsvp_per_color.csv ──
# Build wide table: color_id, color_name, per-subject accuracy + HC mean ± SEM + CVD + diff
rows = []
for cid in range(1, 9):
    row = {'color_id': cid, 'color_name': COLOR_NAMES[cid]}
    hc_accs = []
    for sid in HC_RSVP_SUBJECTS:
        df = hc_per_color[sid]
        v = float(df.loc[df['color_id'] == cid, 'accuracy'].iloc[0])
        row[f'{sid}_acc'] = round(v, 4)
        hc_accs.append(v)
    mean = float(np.mean(hc_accs))
    sd = float(np.std(hc_accs, ddof=1))
    sem = sd / np.sqrt(len(hc_accs))
    row['hc_mean'] = round(mean, 4)
    row['hc_sd'] = round(sd, 4)
    row['hc_sem'] = round(sem, 4)
    cvd_v = float(cvd_per_color.loc[cvd_per_color['color_id'] == cid, 'accuracy'].iloc[0])
    row['cvd'] = round(cvd_v, 4)
    row['diff_cvd_hc'] = round(cvd_v - mean, 4)
    rows.append(row)

per_color_df = pd.DataFrame(rows)
per_color_path = RESULTS_DIR / "rsvp_per_color.csv"
per_color_df.to_csv(per_color_path, index=False)
print(f"Saved: {per_color_path}")

# ── Console summary ──
print("\n" + "=" * 100)
print(f"RSVP 8AFC — HC N={len(HC_RSVP_SUBJECTS)} ({', '.join(HC_RSVP_SUBJECTS)}) vs CVD ({CVD_SUBJECT})")
print("=" * 100)
print(summary_df.to_string(index=False))
print("\nPer-color (HC mean ± SEM vs CVD):")
print(per_color_df[['color_id', 'color_name', 'hc_mean', 'hc_sem', 'cvd', 'diff_cvd_hc']].to_string(index=False))
print("=" * 100)
