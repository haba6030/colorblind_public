#!/usr/bin/env python3
"""A2 — sub-09 optimal-condition staircase instability: diagnosis.

Background
----------
jnd_noise_floor.py (A1) found the within-session staircase disagreement
|sc0 - sc1| in the sub-09 Optimal condition to be 0.0813, 5.2x its own baseline
(0.0156) and larger than the filter effect it is meant to bound (0.0566). Until
the cause is known, no quantitative claim about the sub-09 Optimal condition is
safe. TODO_additional_analysis.md A2 lists five candidate causes:

  1. pair-specific vs global
  2. staircase non-convergence (n_reversals / n_trials)
  3. start-level dependence (0.8 vs 0.5)
  4. filter pushing a pair out of the renderable gamut
  5. within-session block order / fatigue

This script decides among them from the summary and trial-level CSVs alone.
Candidate 4 makes a testable prediction that separates it from all the others:
a rendering failure changes the stimulus, so it degrades BOTH staircases of the
affected pair. A defect confined to one staircase excludes it.

Outputs results/exp2_behavior/a2_staircase_diagnosis.json and prints a report.

Usage
-----
    conda activate srm
    python scripts/a2_staircase_diagnosis.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 03_psychophysics
# Manuscript: Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)
# Original  : analysis/phase6_behavioral_analysis/scripts/a2_staircase_diagnosis.py  (development repository, commit 53c81c2)
# Role      : Staircase audit: censoring at ceiling, track spread -> a2_staircase_diagnosis.json.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
PHASE = os.path.dirname(_HERE)
ROOT = os.path.dirname(os.path.dirname(PHASE))
BEH = os.path.join(ROOT, 'data', 'behavior')
OUTDIR = os.path.join(PHASE, 'results', 'exp2_behavior')

# (subject, condition label, summary path, trials path)
CONDITIONS = []
for sid in ['sub-08', 'sub-09']:
    d = os.path.join(BEH, '2nd_exp', sid)
    CONDITIONS.append((sid, 'deployed', os.path.join(d, 'jnd_ses2_run1_window_no_filter_summary.csv'),
                       os.path.join(d, 'jnd_ses2_run1_window_no_filter_trials.csv')))
    CONDITIONS.append((sid, 'individualized', os.path.join(d, f'jnd_ses2_run2_optimal_{sid}_summary.csv'),
                       os.path.join(d, f'jnd_ses2_run2_optimal_{sid}_trials.csv')))
    CONDITIONS.append((sid, 'session1', os.path.join(BEH, f'{sid}_jnd_ses1_no_filter_summary.csv'), None))
for i in range(1, 8):
    sid = f'sub-0{i}'
    CONDITIONS.append((sid, 'session1', os.path.join(BEH, f'{sid}_jnd_ses1_no_filter_summary.csv'), None))

# Highest level the staircase can present. NOT the 0.8 maximum *start* level:
# levels rise above the start when the participant answers incorrectly, and the
# largest level appearing anywhere in the 13 trial files is 0.95. A staircase
# that answers incorrectly at 0.95 has run out of range, so its threshold is a
# lower bound rather than an estimate.
CEILING = 0.95


def spread_table(path):
    """|sc0 - sc1| and the two staircase thresholds per pair."""
    d = pd.read_csv(path)
    rows = {}
    for pair, g in d.groupby('pair_name'):
        g = g.sort_values('start_level', ascending=False)      # sc0 (0.8) first
        thr = g['jnd_mean'].to_numpy(float)
        rows[pair] = dict(
            sc_hi=float(thr[0]), sc_lo=float(thr[-1]),
            spread=float(abs(thr[0] - thr[-1])),
            threshold=float(np.mean(thr)),
            n_trials=[int(v) for v in g['n_trials']],
            n_reversals=[int(v) for v in g['n_reversals']],
        )
    return rows


def trial_diagnosis(path, pair):
    """Per-staircase trajectory summary for one pair."""
    d = pd.read_csv(path)
    d = d[d['pair_name'] == pair]
    out = {}
    for sc, g in d.groupby('staircase_id'):
        g = g.sort_values('trial')
        lvl = g['level'].to_numpy(float)
        cor = g['correct'].to_numpy(int)
        at_ceiling = lvl >= CEILING - 1e-9
        # last third of the staircase = where it is supposed to have settled
        tail = slice(int(len(lvl) * 2 / 3), None)
        out[sc] = dict(
            n=int(len(lvl)),
            level_min=float(lvl.min()), level_max=float(lvl.max()),
            level_final=float(lvl[-1]), level_tail_mean=float(lvl[tail].mean()),
            accuracy=float(cor.mean()),
            n_at_ceiling=int(at_ceiling.sum()),
            n_incorrect_at_ceiling=int(((~cor.astype(bool)) & at_ceiling).sum()),
            # smallest level answered correctly, and largest answered incorrectly:
            # if the second exceeds the first the staircase is internally inconsistent
            min_level_correct=float(lvl[cor == 1].min()) if (cor == 1).any() else None,
            max_level_incorrect=float(lvl[cor == 0].max()) if (cor == 0).any() else None,
        )
    return out


def census():
    """Dataset-wide scan for range-censored staircases.

    Applied symmetrically to every staircase in every trial file, so the
    exclusion rule it supports is not tailored to the condition under
    investigation. A staircase is censored when it answers incorrectly at the
    maximum presentable level: the true threshold then lies above the tested
    range and the reported value is a lower bound.
    """
    import glob
    files = sorted(glob.glob(os.path.join(BEH, '*_trials.csv'))) + \
        sorted(glob.glob(os.path.join(BEH, '2nd_exp', '*', '*_trials.csv')))
    rows = []
    for f in files:
        d = pd.read_csv(f)
        for (sc, pair), g in d.groupby(['staircase_id', 'pair_name']):
            g = g.sort_values('trial')
            lvl = g['level'].to_numpy(float)
            cor = g['correct'].to_numpy(int)
            tail = lvl[int(len(lvl) * 2 / 3):]
            at = lvl >= CEILING - 1e-9
            rows.append(dict(
                source=os.path.basename(f).replace('_trials.csv', ''),
                staircase=sc, pair=pair, n=int(len(lvl)),
                level_max=float(lvl.max()), tail_mean=float(tail.mean()),
                frac_upper_range=float((lvl >= 0.7).mean()),
                n_at_ceiling=int(at.sum()),
                n_incorrect_at_ceiling=int(((~cor.astype(bool)) & at).sum()),
            ))
    return pd.DataFrame(rows), len(files)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    report = {'ceiling': CEILING, 'conditions': {}}

    cen, n_files = census()
    censored = cen[cen.n_incorrect_at_ceiling > 0]
    report['census'] = dict(
        n_trial_files=int(n_files), n_staircases=int(len(cen)),
        n_censored=int(len(censored)),
        censored=censored.to_dict('records'),
    )
    print(f'=== range-censoring census: {len(cen)} staircases in {n_files} trial files ===')
    print(f'  censored (incorrect response at level {CEILING}): {len(censored)}')
    if len(censored):
        print(censored[['source', 'staircase', 'n', 'level_max', 'tail_mean',
                        'frac_upper_range', 'n_incorrect_at_ceiling']].to_string(index=False))
    print()

    print('=== per-condition staircase spread |sc0 - sc1| ===\n')
    print(f'{"subject":8s} {"condition":16s} {"mean":>7s} {"median":>7s} {"max":>7s}  worst pair')
    for sid, cond, sumpath, trialpath in CONDITIONS:
        if not os.path.exists(sumpath):
            continue
        rows = spread_table(sumpath)
        spreads = {p: r['spread'] for p, r in rows.items()}
        worst = max(spreads, key=spreads.get)
        rest = [v for p, v in spreads.items() if p != worst]
        key = f'{sid}/{cond}'
        report['conditions'][key] = dict(
            pairs=rows,
            mean_spread=float(np.mean(list(spreads.values()))),
            median_spread=float(np.median(list(spreads.values()))),
            max_spread=float(spreads[worst]), worst_pair=worst,
            mean_spread_excluding_worst=float(np.mean(rest)),
        )
        print(f'{sid:8s} {cond:16s} {np.mean(list(spreads.values())):7.4f} '
              f'{np.median(list(spreads.values())):7.4f} {spreads[worst]:7.4f}  '
              f'{worst} ({spreads[worst]:.3f})')

    # ---- focus: sub-09 individualized ------------------------------------
    key = 'sub-09/individualized'
    r = report['conditions'][key]
    worst = r['worst_pair']
    print(f'\n=== {key}: is the excess global or one pair? ===')
    print(f'  mean spread over 8 pairs          {r["mean_spread"]:.4f}')
    print(f'  mean spread excluding {worst:14s} {r["mean_spread_excluding_worst"]:.4f}')
    print(f'  sub-09 baseline (session1) mean    '
          f'{report["conditions"]["sub-09/session1"]["mean_spread"]:.4f}')
    print('\n  per-pair:')
    for p in PAIR_ORDER:
        v = r['pairs'][p]
        print(f'    {p:15s} sc_hi={v["sc_hi"]:.3f} sc_lo={v["sc_lo"]:.3f} '
              f'|d|={v["spread"]:.3f}  n_trials={v["n_trials"]} rev={v["n_reversals"]}')

    # ---- trial-level trajectory of the offending pair --------------------
    trialpath = [t for s, c, _, t in CONDITIONS if s == 'sub-09' and c == 'individualized'][0]
    traj = trial_diagnosis(trialpath, worst)
    report['trajectory'] = {key: {worst: traj}}
    print(f'\n=== {key} / {worst}: trial-level trajectory ===')
    for sc, v in sorted(traj.items()):
        print(f'  {sc}: n={v["n"]:3d} acc={v["accuracy"]:.2f} '
              f'levels {v["level_min"]:.2f}-{v["level_max"]:.2f} '
              f'tail_mean={v["level_tail_mean"]:.3f} final={v["level_final"]:.2f}')
        print(f'        smallest level answered CORRECT   = {v["min_level_correct"]}')
        print(f'        largest  level answered INCORRECT = {v["max_level_incorrect"]}')
        print(f'        trials at ceiling {CEILING}: {v["n_at_ceiling"]} '
              f'(incorrect at ceiling: {v["n_incorrect_at_ceiling"]})')

    # ---- candidate adjudication -----------------------------------------
    hi_sc = max(traj, key=lambda s: traj[s]['level_tail_mean'])
    lo_sc = min(traj, key=lambda s: traj[s]['level_tail_mean'])
    # A rendering/gamut failure changes the stimulus and so degrades both
    # staircases of the pair. One staircase settling low while the other settles
    # high excludes it.
    gamut_excluded = traj[lo_sc]['level_tail_mean'] < 0.3 and traj[hi_sc]['level_tail_mean'] > 0.6
    inconsistent = (traj[hi_sc]['max_level_incorrect'] or 0) > (traj[hi_sc]['min_level_correct'] or 1)
    censored = traj[hi_sc]['n_incorrect_at_ceiling'] > 0

    report['adjudication'] = dict(
        pair_specific=bool(r['mean_spread_excluding_worst']
                           < 2 * report['conditions']['sub-09/session1']['mean_spread']),
        start_level_dependent=bool(gamut_excluded),
        gamut_clipping_excluded=bool(gamut_excluded),
        internally_inconsistent=bool(inconsistent),
        censored_at_ceiling=bool(censored),
        high_staircase=hi_sc, low_staircase=lo_sc,
    )
    print('\n=== adjudication ===')
    for k, v in report['adjudication'].items():
        print(f'  {k:28s} {v}')

    out = os.path.join(OUTDIR, 'a2_staircase_diagnosis.json')
    with open(out, 'w') as f:
        json.dump(report, f, indent=1)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
