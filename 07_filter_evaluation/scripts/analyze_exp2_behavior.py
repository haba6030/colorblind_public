#!/usr/bin/env python3
"""Stage D / exp2 behavioral analysis — JND + RSVP-8AFC, per CVD subject.

Three conditions per subject:
  baseline  = exp1 (ses1) no-filter         data/behavior/sub-{ID}_*_ses1_*
  window    = exp2 (ses2) macOS Color Filter (deployed product, subject-tuned)
  optimal   = exp2 (ses2) personalized 2-comp pre-image filter (PsychoPy render)

Framing (per phase6_behavioral_analysis/CLAUDE.md, Path A):
  - JND/RSVP = OUTCOME comparison only. Window vs Optimal is subject to a
    rendering-pipeline confound (OS transform vs PsychoPy direct), so neither
    a win nor a tie here is interpretable as personalization (that lives in the
    cross-swap + fMRI prongs A-E).
  - Robust, reportable claim: both filters restore near-HC performance vs the
    no-filter baseline. Do NOT assert an optimal-vs-window null from this
    single-subject, single-run pilot (underpowered: 8 pairs, Wilcoxon).

Usage: python analyze_exp2_behavior.py sub-08
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/analyze_exp2_behavior.py  (development repository, commit 53c81c2)
# Role      : JND and 8AFC per condition, Wilcoxon between filters -> sub-*_summary.json
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

import sys, json, os
import pandas as pd, numpy as np
from scipy.stats import wilcoxon

HUE = {1:'red',2:'orange',3:'yellow',4:'green',5:'cyan',6:'blue',7:'purple',8:'magenta'}
PAIR_ORDER = ['orange-yellow','yellow-green','green-blue','red-orange',
              'blue-purple','yellow-purple','cyan-magenta','red-cyan']
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEH = os.path.join(ROOT, 'data', 'behavior')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'exp2_behavior')


def jnd_per_pair(path):
    """Mean JND threshold per color pair (average of the 2 staircases)."""
    return pd.read_csv(path).groupby('pair_name')['jnd_mean'].mean()


def rsvp_summary(path):
    d = pd.read_csv(path)
    errs = []
    for _, r in d[d['correct'] == 0].iterrows():
        s = HUE.get(int(str(r['stimulus_label']).split('_')[1]), r['stimulus_label'])
        rr = str(r['response_label'])
        rl = HUE.get(int(rr.split('_')[1]), rr) if ('_' in rr and rr.split('_')[1].isdigit()) else 'MISS'
        errs.append(f"{s}->{rl}")
    return {'acc': float(d['correct'].mean()), 'n': int(len(d)),
            'n_correct': int(d['correct'].sum()), 'errors': errs}


def analyze(sid):
    s2 = f"{BEH}/2nd_exp/{sid}"
    jnd = {
        'baseline': jnd_per_pair(f"{BEH}/{sid}_jnd_ses1_no_filter_summary.csv"),
        'window':   jnd_per_pair(f"{s2}/jnd_ses2_run1_window_no_filter_summary.csv"),
        'optimal':  jnd_per_pair(f"{s2}/jnd_ses2_run2_optimal_{sid}_summary.csv"),
    }
    T = pd.DataFrame(jnd).reindex(PAIR_ORDER)
    T['opt_minus_win']  = T['optimal'] - T['window']
    T['opt_minus_base'] = T['optimal'] - T['baseline']

    pairs = T.index
    stats = {
        'wilcoxon_opt_vs_win':  wilcoxon(T['optimal'][pairs], T['window'][pairs]),
        'wilcoxon_opt_vs_base': wilcoxon(T['optimal'][pairs], T['baseline'][pairs]),
        'wilcoxon_win_vs_base': wilcoxon(T['window'][pairs],  T['baseline'][pairs]),
    }
    rsvp = {
        'baseline': rsvp_summary(f"{BEH}/{sid}_rsvp_8afc_ses1_run1.csv"),
        'window':   rsvp_summary(f"{s2}/rsvp_8afc_ses2_run1_window_no_filter.csv"),
        'optimal':  rsvp_summary(f"{s2}/rsvp_8afc_ses2_run2_optimal_{sid}.csv"),
    }

    os.makedirs(OUTDIR, exist_ok=True)
    T.round(4).to_csv(f"{OUTDIR}/{sid}_jnd_compare.csv")
    out = {
        'subject': sid,
        'jnd_mean_by_condition': {k: round(float(T[k].mean()), 4)
                                  for k in ['baseline', 'window', 'optimal']},
        'jnd_opt_lt_win_npairs': int((T['opt_minus_win'] < 0).sum()),
        'jnd_opt_lt_base_npairs': int((T['opt_minus_base'] < 0).sum()),
        'wilcoxon': {k: {'stat': float(v.statistic), 'p': float(v.pvalue)}
                     for k, v in stats.items()},
        'rsvp': rsvp,
    }
    with open(f"{OUTDIR}/{sid}_summary.json", 'w') as f:
        json.dump(out, f, indent=2)

    pd.set_option('display.float_format', lambda x: f"{x:.4f}")
    print(f"\n=== {sid} JND per pair (lower = better discrimination) ===")
    print(T)
    print(f"\nMean JND: {out['jnd_mean_by_condition']}")
    print(f"optimal<window: {out['jnd_opt_lt_win_npairs']}/8   "
          f"optimal<baseline: {out['jnd_opt_lt_base_npairs']}/8")
    print("\nWilcoxon (n=8 pairs):")
    for k, v in stats.items():
        print(f"  {k:22s} W={v.statistic:.1f}  p={v.pvalue:.3f}")
    print("\n=== RSVP-8AFC ===")
    for k in ['baseline', 'window', 'optimal']:
        r = rsvp[k]
        print(f"  {k:9s} acc={r['acc']:.3f} ({r['n_correct']}/{r['n']})  errors={r['errors']}")
    print(f"\nSaved -> {OUTDIR}/{sid}_jnd_compare.csv, {sid}_summary.json")
    return out


if __name__ == '__main__':
    analyze(sys.argv[1] if len(sys.argv) > 1 else 'sub-08')
