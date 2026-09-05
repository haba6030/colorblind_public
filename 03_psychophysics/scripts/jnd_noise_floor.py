#!/usr/bin/env python3
"""JND measurement noise floor vs filter-induced change (patterson2022 standard).

Background
----------
Patterson et al. (2022, Opt. Express 30:31186) judged a commercial CVD filter
"functionally meaningful" only if its mean effect on CAD thresholds exceeded the
test-retest variability of the measurement itself (SD of baseline differences =
1.91 units). EnChroma's 1.16-unit mean effect fell below that floor; VINO's
8.03-unit effect (4.2x) cleared it.

Our JND paradigm runs two independent staircases per hue pair (start levels 0.8
and 0.5). Their disagreement |sc0 - sc1| gives a within-session measurement
noise floor in the same JND units as the filter effect, so the same comparison
can be made for the Window (deployed macOS Color Filter) and Optimal
(personalized 2-comp pre-image) conditions.

Caveat (stated in the output and in the write-up): |sc0 - sc1| is a *within*-
session floor. It excludes session-to-session drift, which patterson2022's
between-baseline SD does include. The ratios reported here are therefore
optimistic upper bounds, not like-for-like replications of their statistic.

Inputs
------
data/behavior/sub-0{1..7}_jnd_ses1_no_filter_summary.csv      (HC baseline)
data/behavior/sub-0{8,9}_jnd_ses1_no_filter_summary.csv       (CVD baseline)
data/behavior/2nd_exp/{sid}/jnd_ses2_run1_window_no_filter_summary.csv
data/behavior/2nd_exp/{sid}/jnd_ses2_run2_optimal_{sid}_summary.csv

Columns used: pair_name, staircase_id, start_level, jnd_mean

Output
------
results/exp2_behavior/jnd_noise_floor.json

Usage
-----
    conda activate srm
    python scripts/jnd_noise_floor.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 03_psychophysics
# Manuscript: Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)
# Original  : analysis/phase6_behavioral_analysis/scripts/jnd_noise_floor.py  (development repository, commit 53c81c2)
# Role      : Split-half noise floor of the JND estimates.
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
import os

import numpy as np
import pandas as pd

PAIR_ORDER = ['orange-yellow', 'yellow-green', 'green-blue', 'red-orange',
              'blue-purple', 'yellow-purple', 'cyan-magenta', 'red-cyan']
HC_IDS = [f'sub-0{i}' for i in range(1, 8)]
CVD_IDS = ['sub-08', 'sub-09']

_HERE = os.path.dirname(os.path.abspath(__file__))            # .../scripts
PHASE = os.path.dirname(_HERE)                                # .../phase6_behavioral_analysis
ROOT = os.path.dirname(os.path.dirname(PHASE))                # repo root
BEH = os.path.join(ROOT, 'data', 'behavior')
OUTDIR = os.path.join(PHASE, 'results', 'exp2_behavior')


def read_condition(path):
    """Return (threshold_per_pair, staircase_spread_per_pair) indexed by PAIR_ORDER.

    threshold = mean of the two staircases (matches analyze_exp2_behavior.py)
    spread    = |sc0 - sc1| = within-session measurement noise for that pair
    """
    d = pd.read_csv(path)
    g = d.groupby('pair_name')['jnd_mean']
    thr = g.mean().reindex(PAIR_ORDER)
    spread = (g.max() - g.min()).reindex(PAIR_ORDER)
    return thr, spread


def summarize(x):
    x = np.asarray(x, dtype=float)
    return {'mean': float(np.mean(x)), 'median': float(np.median(x)),
            'min': float(np.min(x)), 'max': float(np.max(x))}


def main():
    out = {
        'analysis': 'jnd_noise_floor',
        'standard': 'patterson2022 (effect vs measurement variability)',
        'noise_metric': 'abs(staircase_0 - staircase_1) per hue pair, same session',
        'caveat': ('within-session floor; excludes session-to-session drift that '
                   'patterson2022 between-baseline SD includes -> ratios are '
                   'optimistic upper bounds'),
        'reference_patterson2022': {
            'test_retest_sd_units': 1.91,
            'enchroma_mean_effect': 1.16, 'enchroma_ratio': round(1.16 / 1.91, 2),
            'vino_mean_effect': 8.03, 'vino_ratio': round(8.03 / 1.91, 2),
        },
        'hc': {}, 'cvd': {},
    }

    # ---- HC noise floor (baseline session only; HC has no repeat session) ----
    hc_spread = {}
    for sid in HC_IDS:
        _, sp = read_condition(f'{BEH}/{sid}_jnd_ses1_no_filter_summary.csv')
        hc_spread[sid] = sp
        out['hc'][sid] = {'noise_floor': summarize(sp)}
    pooled = pd.concat(hc_spread.values(), axis=1).values.ravel()
    out['hc']['pooled'] = {'n_subjects': len(HC_IDS), 'noise_floor': summarize(pooled)}

    # ---- CVD: noise floor per condition + filter-induced change ----
    for sid in CVD_IDS:
        base_t, base_s = read_condition(f'{BEH}/{sid}_jnd_ses1_no_filter_summary.csv')
        win_t, win_s = read_condition(
            f'{BEH}/2nd_exp/{sid}/jnd_ses2_run1_window_no_filter_summary.csv')
        opt_t, opt_s = read_condition(
            f'{BEH}/2nd_exp/{sid}/jnd_ses2_run2_optimal_{sid}_summary.csv')

        floor_pooled = float(pd.concat([base_s, win_s, opt_s], axis=1).values.mean())
        d_opt = (opt_t - base_t).abs()
        d_win = (win_t - base_t).abs()

        out['cvd'][sid] = {
            'noise_floor_by_condition': {
                'baseline': summarize(base_s),
                'window': summarize(win_s),
                'optimal': summarize(opt_s),
            },
            'noise_floor_pooled_mean': floor_pooled,
            'change_vs_baseline': {
                'optimal': {
                    'mean_abs_change': float(d_opt.mean()),
                    'ratio_to_floor': float(d_opt.mean() / floor_pooled),
                    'n_pairs_exceeding_floor': int((d_opt > floor_pooled).sum()),
                    'n_pairs': len(PAIR_ORDER),
                    'per_pair': {p: float(v) for p, v in d_opt.items()},
                },
                'window': {
                    'mean_abs_change': float(d_win.mean()),
                    'ratio_to_floor': float(d_win.mean() / floor_pooled),
                    'n_pairs_exceeding_floor': int((d_win > floor_pooled).sum()),
                    'n_pairs': len(PAIR_ORDER),
                    'per_pair': {p: float(v) for p, v in d_win.items()},
                },
            },
        }

    os.makedirs(OUTDIR, exist_ok=True)
    dest = os.path.join(OUTDIR, 'jnd_noise_floor.json')
    with open(dest, 'w') as f:
        json.dump(out, f, indent=2)

    # ---- console report ----
    print(f'HC pooled noise floor: mean {out["hc"]["pooled"]["noise_floor"]["mean"]:.4f}'
          f'  median {out["hc"]["pooled"]["noise_floor"]["median"]:.4f}')
    for sid in CVD_IDS:
        c = out['cvd'][sid]
        print(f'\n{sid}  pooled floor {c["noise_floor_pooled_mean"]:.4f}'
              f'  (base {c["noise_floor_by_condition"]["baseline"]["mean"]:.4f} /'
              f' win {c["noise_floor_by_condition"]["window"]["mean"]:.4f} /'
              f' opt {c["noise_floor_by_condition"]["optimal"]["mean"]:.4f})')
        for cond in ['optimal', 'window']:
            r = c['change_vs_baseline'][cond]
            print(f'  {cond:8s} |Δ| {r["mean_abs_change"]:.4f}'
                  f'  ratio {r["ratio_to_floor"]:.2f}x'
                  f'  exceeding {r["n_pairs_exceeding_floor"]}/{r["n_pairs"]}')
    print(f'\nwrote {dest}')


if __name__ == '__main__':
    main()
