"""
Compute HC group JND metrics from the unified behavior pilot dataset.

Data source: project_root/data/behavior/sub-{01..08}_jnd_ses1_no_filter_summary.csv
  - HC: sub-01..sub-07 (N=7)
  - CVD: sub-08 (deutan)

Saves: results/hc_group_metrics.json
Schema: one entry per color pair with fields
  hc_names, hc_mean, hc_std, hc_sem, n_hc, cvd, ratio, direction_hc_group,
  plus per-subject keys (sub-01, ..., sub-07).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 03_psychophysics
# Manuscript: Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)
# Original  : analysis/phase6_behavioral_analysis/scripts/compute_hc_group_metrics.py  (development repository, commit 53c81c2)
# Role      : Control JND distribution per hue pair -> hc_group_metrics.json.
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

PAIRS_ORDERED = [
    'red-orange', 'orange-yellow', 'yellow-green', 'green-blue',
    'yellow-purple', 'blue-purple', 'cyan-magenta', 'red-cyan',
]

HC_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 8)]
CVD_SUBJECT = "sub-08"


def load_jnd(subject_id):
    """Load per-pair JND (mean of both staircases) for a subject."""
    summary_file = BEHAVIOR_DIR / f"{subject_id}_jnd_ses1_no_filter_summary.csv"
    df = pd.read_csv(summary_file)
    return df.groupby('pair_name')['jnd_mean'].mean().to_dict()


hc_data = {sid: load_jnd(sid) for sid in HC_SUBJECTS}
cvd_data = load_jnd(CVD_SUBJECT)

for sid in HC_SUBJECTS + [CVD_SUBJECT]:
    print(f"  Loaded: {sid}")

print(f"\nHC N={len(HC_SUBJECTS)} ({', '.join(HC_SUBJECTS)}); CVD={CVD_SUBJECT}")

hc_metrics = {}
for pair in PAIRS_ORDERED:
    hc_values = [hc_data[sid].get(pair, np.nan) for sid in HC_SUBJECTS]
    hc_values_clean = [v for v in hc_values if not np.isnan(v)]

    hc_mean = float(np.mean(hc_values_clean))
    hc_std = float(np.std(hc_values_clean, ddof=1))
    hc_sem = hc_std / np.sqrt(len(hc_values_clean))

    cvd_val = float(cvd_data[pair])
    ratio = cvd_val / hc_mean if hc_mean > 0 else np.nan

    if ratio > 1.15:
        direction = 'HYPO'
    elif ratio < 0.85:
        direction = 'HYPER'
    else:
        direction = 'borderline'

    entry = {
        'hc_names': HC_SUBJECTS,
        'hc_mean': hc_mean,
        'hc_std': hc_std,
        'hc_sem': float(hc_sem),
        'n_hc': len(hc_values_clean),
        'cvd': cvd_val,
        'ratio': float(ratio),
        'direction_hc_group': direction,
    }
    for sid in HC_SUBJECTS:
        entry[sid] = float(hc_data[sid].get(pair, np.nan))

    hc_metrics[pair] = entry

output_json = RESULTS_DIR / "hc_group_metrics.json"
with open(output_json, 'w') as f:
    json.dump(hc_metrics, f, indent=2)
print(f"\nSaved: {output_json}")

print("\n" + "=" * 140)
print(f"HC GROUP METRICS (N={len(HC_SUBJECTS)})")
print("=" * 140)

header = f"{'Pair':<16}"
for sid in HC_SUBJECTS:
    header += f" {sid:<8}"
header += f" {'Mean':<7} {'SD':<7} {'CVD':<7} {'Ratio':<6} {'Dir(Group)'}"
print(header)
print("-" * 140)
for pair in PAIRS_ORDERED:
    m = hc_metrics[pair]
    row = f"{pair:<16}"
    for sid in HC_SUBJECTS:
        row += f" {m[sid]:<8.4f}"
    row += (f" {m['hc_mean']:<7.4f}"
            f" {m['hc_std']:<7.4f}"
            f" {m['cvd']:<7.4f}"
            f" {m['ratio']:<6.2f}"
            f" {m['direction_hc_group']}")
    print(row)
print("=" * 140)
