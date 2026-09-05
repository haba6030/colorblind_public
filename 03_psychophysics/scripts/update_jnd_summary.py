"""Update jnd_summary.csv from hc_group_metrics.json (supports variable N HC)."""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 03_psychophysics
# Manuscript: Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)
# Original  : analysis/phase6_behavioral_analysis/scripts/update_jnd_summary.py  (development repository, commit 53c81c2)
# Role      : Tabulates hc_group_metrics.json -> jnd_summary.csv.
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
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR.parent / "results"

# Load HC group metrics
with open(RESULTS_DIR / "hc_group_metrics.json", 'r') as f:
    metrics = json.load(f)

# Create updated summary
data = []
for pair in metrics:
    m = metrics[pair]
    hc_names = m['hc_names']

    row = {
        'pair': pair,
        'hc_group_mean': m['hc_mean'],
        'hc_group_std': m['hc_std'],
        'hc_group_sem': m['hc_sem'],
        'n_hc': m['n_hc'],
        'cvd_jnd_mean': m['cvd'],
        'ratio_cvd_hc_group': m['ratio'],
        'direction_hc_group': m['direction_hc_group'],
    }

    # Add individual HC columns
    for name in hc_names:
        row[f'{name}_jnd'] = m[name]

    data.append(row)

df = pd.DataFrame(data)

# Save
output_file = RESULTS_DIR / "jnd_summary.csv"
df.to_csv(output_file, index=False, float_format='%.4f')
print(f"Saved: {output_file}")
print(f"Columns: {', '.join(df.columns)}")
print(f"Rows: {len(df)}")
