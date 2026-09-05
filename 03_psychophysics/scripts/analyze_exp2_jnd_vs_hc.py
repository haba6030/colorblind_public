#!/usr/bin/env python3
"""Stage D / exp2 — JND evaluated by DISTANCE TO HC GROUP MEAN.

Per user request (2026-06-30): the filter's goal is to move CVD perception toward
HC. So evaluate each condition's JND not against the subject's own no-filter
baseline alone, but against the **HC group-mean JND** (sub-01..07, ses1 no-filter).

Reference (target) = HC mean JND per color pair (and overall).
Metric = signed diff (cond - HC_mean), |diff|, and z = diff / HC_sd(across HC).
"More HC-like" = smaller |cond - HC_mean|.

Conditions (same loaders as analyze_exp2_behavior.py):
  nofilter = exp1 ses1 (subject's own no-filter)
  window   = exp2 ses2 run1 macOS Color Filter
  optimal  = exp2 ses2 run2 personalized 2-comp pre-image filter

Usage: python analyze_exp2_jnd_vs_hc.py sub-09
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 03_psychophysics
# Manuscript: Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)
# Original  : analysis/phase6_behavioral_analysis/analyze_exp2_jnd_vs_hc.py  (development repository, commit 53c81c2)
# Role      : JND of every condition against the control distribution (z per pair; also used in 07).
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
from scipy.stats import t as tdist


def crawford_howell(x, control):
    """Crawford & Howell (1998) modified t-test: single case x vs control sample.
    Treats controls as a SAMPLE (not population) — correct for N=7 HC.
    Returns (t, df, p_two_tailed, z_cc) where z_cc=(x-M)/s is the case effect size."""
    control = np.asarray(control, float)
    control = control[~np.isnan(control)]
    n = control.size
    m, s = control.mean(), control.std(ddof=1)
    tval = (x - m) / (s * np.sqrt((n + 1) / n))
    df = n - 1
    p = float(2 * tdist.sf(abs(tval), df))
    return float(tval), int(df), p, float((x - m) / s)

HC_SUBJS = [f"sub-0{i}" for i in range(1, 8)]   # sub-01..07
PAIR_ORDER = ['orange-yellow', 'yellow-green', 'green-blue', 'red-orange',
              'blue-purple', 'yellow-purple', 'cyan-magenta', 'red-cyan']
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BEH = os.path.join(ROOT, 'data', 'behavior')
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'exp2_behavior')


def jnd_per_pair(path):
    """Mean JND threshold per color pair (average of the 2 staircases)."""
    return pd.read_csv(path).groupby('pair_name')['jnd_mean'].mean()


def hc_reference():
    """HC per-pair-per-subject JND matrix across sub-01..07 (ses1 no-filter)."""
    cols = {}
    for s in HC_SUBJS:
        p = f"{BEH}/{s}_jnd_ses1_no_filter_summary.csv"
        if os.path.exists(p):
            cols[s] = jnd_per_pair(p)
    H = pd.DataFrame(cols).reindex(PAIR_ORDER)   # rows=pair, cols=HC subj
    return H.mean(axis=1), H.std(axis=1, ddof=1), H.shape[1], H


def analyze(sid):
    s2 = f"{BEH}/2nd_exp/{sid}"
    cond = {
        'nofilter': jnd_per_pair(f"{BEH}/{sid}_jnd_ses1_no_filter_summary.csv"),
        'window':   jnd_per_pair(f"{s2}/jnd_ses2_run1_window_no_filter_summary.csv"),
        'optimal':  jnd_per_pair(f"{s2}/jnd_ses2_run2_optimal_{sid}_summary.csv"),
    }
    hc_mu, hc_sd, n_hc, H = hc_reference()
    T = pd.DataFrame(cond).reindex(PAIR_ORDER)
    T['HC_mean'] = hc_mu
    T['HC_sd'] = hc_sd

    # ---- per-pair Crawford-Howell: case (each condition) vs HC sample ----
    CH = {}      # pair -> {cond -> (t, df, p, z_cc)}
    for pair in PAIR_ORDER:
        ctrl = H.loc[pair].values
        CH[pair] = {c: crawford_howell(T.loc[pair, c], ctrl)
                    for c in ['nofilter', 'window', 'optimal']}

    # signed diff, |diff|, z relative to HC distribution, for each condition
    rows = {}
    for c in ['nofilter', 'window', 'optimal']:
        T[f'{c}-HC'] = T[c] - hc_mu
        T[f'|{c}-HC|'] = (T[c] - hc_mu).abs()
        T[f'z_{c}'] = (T[c] - hc_mu) / hc_sd

    # per-pair: is optimal closer to HC than window? than no-filter?
    opt_lt_win = int((T['|optimal-HC|'] < T['|window-HC|']).sum())
    opt_lt_nof = int((T['|optimal-HC|'] < T['|nofilter-HC|']).sum())
    win_lt_nof = int((T['|window-HC|'] < T['|nofilter-HC|']).sum())

    summ = {
        'subject': sid,
        'n_hc': int(n_hc),
        'hc_mean_overall': round(float(hc_mu.mean()), 4),
        'cond_mean_overall': {c: round(float(T[c].mean()), 4)
                              for c in ['nofilter', 'window', 'optimal']},
        # mean absolute distance to HC mean (lower = more HC-like), across 8 pairs
        'mean_abs_dist_to_HC': {c: round(float(T[f'|{c}-HC|'].mean()), 4)
                                for c in ['nofilter', 'window', 'optimal']},
        # mean |z| relative to HC spread
        'mean_abs_z_to_HC': {c: round(float(T[f'z_{c}'].abs().mean()), 4)
                             for c in ['nofilter', 'window', 'optimal']},
        'optimal_closer_to_HC_than_window_npairs': opt_lt_win,
        'optimal_closer_to_HC_than_nofilter_npairs': opt_lt_nof,
        'window_closer_to_HC_than_nofilter_npairs': win_lt_nof,
        # Crawford-Howell single-case-vs-HC per pair, per condition
        'crawford_howell': {
            pair: {c: {'t': round(CH[pair][c][0], 3), 'df': CH[pair][c][1],
                       'p': round(CH[pair][c][2], 4), 'z_cc': round(CH[pair][c][3], 3),
                       'sig05': bool(CH[pair][c][2] < 0.05)}
                   for c in ['nofilter', 'window', 'optimal']}
            for pair in PAIR_ORDER},
        'nofilter_sig_pairs': [p for p in PAIR_ORDER if CH[p]['nofilter'][2] < 0.05],
    }

    os.makedirs(OUTDIR, exist_ok=True)
    T.round(4).to_csv(f"{OUTDIR}/{sid}_jnd_vs_hc.csv")
    with open(f"{OUTDIR}/{sid}_jnd_vs_hc.json", 'w') as f:
        json.dump(summ, f, indent=2)

    pd.set_option('display.float_format', lambda x: f"{x:.4f}")
    pd.set_option('display.width', 200)
    show = ['HC_mean', 'HC_sd', 'nofilter', 'window', 'optimal',
            '|nofilter-HC|', '|window-HC|', '|optimal-HC|']
    print(f"\n=== {sid} JND vs HC group mean (N_HC={n_hc}) — lower |·-HC| = more HC-like ===")
    print(T[show])
    print(f"\nOverall mean JND   HC={summ['hc_mean_overall']}   "
          f"nofilter={summ['cond_mean_overall']['nofilter']}  "
          f"window={summ['cond_mean_overall']['window']}  "
          f"optimal={summ['cond_mean_overall']['optimal']}")
    print(f"Mean |dist to HC|  nofilter={summ['mean_abs_dist_to_HC']['nofilter']}  "
          f"window={summ['mean_abs_dist_to_HC']['window']}  "
          f"optimal={summ['mean_abs_dist_to_HC']['optimal']}  (lower=better)")
    print(f"Mean |z to HC|     nofilter={summ['mean_abs_z_to_HC']['nofilter']}  "
          f"window={summ['mean_abs_z_to_HC']['window']}  "
          f"optimal={summ['mean_abs_z_to_HC']['optimal']}")
    print(f"\noptimal closer to HC than window:   {opt_lt_win}/8 pairs")
    print(f"optimal closer to HC than no-filter: {opt_lt_nof}/8 pairs")
    print(f"window  closer to HC than no-filter: {win_lt_nof}/8 pairs")

    # ---- Crawford-Howell per-pair table (case vs HC sample) ----
    print(f"\n=== Crawford-Howell single-case vs HC (N={n_hc}) — per pair, per condition ===")
    print(f"  (t / p ; * = p<0.05 ; z_cc = (case-HCmean)/HCsd)")
    hdr = f"{'pair':14}" + "".join(f"{c:>22}" for c in ['nofilter', 'window', 'optimal'])
    print(hdr)
    for pair in PAIR_ORDER:
        cells = ""
        for c in ['nofilter', 'window', 'optimal']:
            t_, df_, p_, z_ = CH[pair][c]
            star = '*' if p_ < 0.05 else ' '
            cells += f"  t={t_:+5.2f} p={p_:.3f}{star}"
        print(f"{pair:14}{cells}")
    sigp = summ['nofilter_sig_pairs']
    print(f"\n>> NO-FILTER pairs significantly != HC (p<0.05): {sigp if sigp else 'none'}")
    for p in sigp:
        nf, wi, op = CH[p]['nofilter'], CH[p]['window'], CH[p]['optimal']
        def tag(x): return 'NORMALIZED (p>=.05)' if x[2] >= 0.05 else f'still sig (p={x[2]:.3f})'
        print(f"   {p}: no-filter z_cc={nf[3]:+.2f} p={nf[2]:.3f}  ->  "
              f"window: {tag(wi)} (z={wi[3]:+.2f}) | optimal: {tag(op)} (z={op[3]:+.2f})")
    print(f"\nSaved -> {OUTDIR}/{sid}_jnd_vs_hc.csv, {sid}_jnd_vs_hc.json")
    return summ


if __name__ == '__main__':
    analyze(sys.argv[1] if len(sys.argv) > 1 else 'sub-09')
