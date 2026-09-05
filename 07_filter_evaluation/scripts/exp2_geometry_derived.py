#!/usr/bin/env python3
"""Stage 2 -- derived geometry comparisons (File 2), cross-referencing File 1.

Reads exp2_embeddings_{subj}_{variant}.json (Stage 1) + JND per-pair csv, computes:
  (1) inter_embedding_agreement : Pearson/Spearman of the 28-vec distance between
      embedding PAIRS {srm~fe, srm~procrustes, fe~procrustes}, per condition,
      for eucl and corr distance. -> is the SRM<->LOCO dissociation real or a
      summary-statistic artifact?
  (2) relative_distance : mean pairwise separation per embedding/condition (+ ref).
  (3) displacement : procrustes_disparity(condition coords, reference coords),
      vs_nf (within-subject filter effect) and vs_hc (HC-likeness). srm/fe have
      coords for both; procrustes(voxel) has vs_nf only (same V under matched).
  (4) jnd_correlation : neural pairwise distance at the 8 measured JND pairs vs the
      JND distance for that condition (Pearson/Spearman, n=8, direction only).

Pure JSON arithmetic -> runs LOCAL (no brainiak/server).
Run:  python exp2_geometry_derived.py --subject 08 --variant matched
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp2_geometry_derived.py  (development repository, commit 53c81c2)
# Role      : Derived geometry indices per condition.
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

import csv
import json
import argparse
import numpy as np
from pathlib import Path
from scipy.linalg import orthogonal_procrustes
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"
BEHAV = HERE.parent / "results" / "exp2_behavior"
TRIU = np.triu_indices(8, k=1)
# (i,j)->position in the 28-vec upper-triangle order
PAIR_POS = {(int(i), int(j)): p for p, (i, j) in enumerate(zip(*TRIU))}
COND2COL = {'nofilter': 'baseline', 'window': 'window', 'optimal': 'optimal'}
EMB = ['procrustes', 'srm', 'fe_latent']


def pear_spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return {'pearson': None, 'spearman': None, 'n': int(len(a))}
    return {'pearson': float(pearsonr(a, b)[0]), 'spearman': float(spearmanr(a, b)[0]),
            'n': int(len(a))}


def procrustes_full(X, Y):
    """disparity + per-point residual norm after centre+unit-Frob+optimal rotation."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Xn = Xc / (np.linalg.norm(Xc, 'fro') + 1e-12)
    Yn = Yc / (np.linalg.norm(Yc, 'fro') + 1e-12)
    R, _ = orthogonal_procrustes(Xn, Yn)
    resid = Xn @ R - Yn
    return float(np.linalg.norm(resid, 'fro')), [float(np.linalg.norm(r)) for r in resid]


def load_jnd(subj):
    f = BEHAV / f"{subj}_jnd_compare.csv"
    if not f.exists():
        return None
    out = {}
    for row in csv.DictReader(open(f)):
        out[row['pair_name']] = {k: float(row[k]) for k in ('baseline', 'window', 'optimal')
                                 if row.get(k) not in (None, '')}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subject', default='08')
    ap.add_argument('--variant', default='matched', choices=['native', 'matched'])
    args = ap.parse_args()
    subj = f"sub-{args.subject}"
    f1 = json.load(open(RES / f"exp2_embeddings_{subj}_{args.variant}.json"))
    jnd = load_jnd(subj)
    jnd_pairs = f1['jnd_pairs']                       # [{name,i,j}]
    conditions = ['nofilter', 'window', 'optimal']

    out = {'subject': subj, 'variant': args.variant,
           'source': f"exp2_embeddings_{subj}_{args.variant}.json", 'rois': {}}

    for roi, R in f1['rois'].items():
        emb = R['embeddings']
        present = [c for c in conditions if c in emb['srm']['conditions']]
        rr = {'inter_embedding_agreement': {}, 'relative_distance': {},
              'displacement': {}, 'jnd_correlation': {}}

        # (1) inter-embedding agreement (per condition, eucl + corr)
        for c in present:
            rr['inter_embedding_agreement'][c] = {}
            for metric in ('dist_eucl', 'dist_corr'):
                for a, b in [('srm', 'fe_latent'), ('srm', 'procrustes'), ('fe_latent', 'procrustes')]:
                    va = emb[a]['conditions'][c][metric]
                    vb = emb[b]['conditions'][c][metric]
                    rr['inter_embedding_agreement'][c][f'{a}~{b}|{metric.split("_")[1]}'] = pear_spear(va, vb)

        # (2) relative distance (mean separation)
        for e in EMB:
            rr['relative_distance'][e] = {}
            for c in present:
                cc = emb[e]['conditions'][c]
                rr['relative_distance'][e][c] = {
                    'mean_sep_eucl': float(np.mean(cc['dist_eucl'])),
                    'mean_sep_corr': float(np.mean(cc['dist_corr'])),
                    'ref': f'embeddings.{e}.conditions.{c}.dist_eucl'}

        # (3) displacement (procrustes): vs_nf and vs_hc
        nf = 'nofilter' if 'nofilter' in present else None
        for e in EMB:
            rr['displacement'][e] = {}
            has_hc_coords = 'coords' in emb[e]['hc_ref']
            hc_coords = np.array(emb[e]['hc_ref']['coords']) if has_hc_coords else None
            nf_coords = np.array(emb[e]['conditions'][nf]['coords']) if nf else None
            for c in present:
                cc = np.array(emb[e]['conditions'][c]['coords'])
                d = {}
                if hc_coords is not None:
                    disp, pp = procrustes_full(cc, hc_coords)
                    d['vs_hc'] = {'disparity': disp, 'per_point': pp}
                if nf_coords is not None and c != nf and cc.shape == nf_coords.shape:
                    disp, pp = procrustes_full(cc, nf_coords)
                    d['vs_nf'] = {'disparity': disp, 'per_point': pp}
                rr['displacement'][e][c] = d

        # (4) jnd correlation (8 measured pairs)
        if jnd is not None:
            for c in present:
                col = COND2COL[c]
                jvals, idxs = [], []
                for p in jnd_pairs:
                    jd = jnd.get(p['name'], {}).get(col)
                    if jd is not None:
                        jvals.append(jd)
                        idxs.append(PAIR_POS[(min(p['i'], p['j']), max(p['i'], p['j']))])
                rr['jnd_correlation'][c] = {'n_pairs': len(jvals), 'jnd': jvals}
                for e in EMB:
                    for metric in ('dist_eucl', 'dist_corr'):
                        nd = [emb[e]['conditions'][c][metric][k] for k in idxs]
                        rr['jnd_correlation'][c][f'{e}|{metric.split("_")[1]}'] = pear_spear(nd, jvals)
        out['rois'][roi] = rr

    outp = RES / f"exp2_geometry_derived_{subj}_{args.variant}.json"
    outp.write_text(json.dumps(out, indent=1))
    print(f"SAVED {outp}")

    # ---- console summary ----
    print(f"\n{'='*78}\n{subj} {args.variant} — DERIVED\n{'='*78}")
    for roi, rr in out['rois'].items():
        print(f"\n--- {roi} ---")
        print("(1) srm~fe agreement (spearman, corr-dist):")
        for c in ['nofilter', 'window', 'optimal']:
            a = rr['inter_embedding_agreement'].get(c, {}).get('srm~fe_latent|corr')
            if a:
                print(f"    {c:9} r_s={a['spearman']}")
        print("(3) displacement vs_HC (srm / fe):")
        for c in ['nofilter', 'window', 'optimal']:
            s = rr['displacement']['srm'].get(c, {}).get('vs_hc', {}).get('disparity')
            f = rr['displacement']['fe_latent'].get(c, {}).get('vs_hc', {}).get('disparity')
            print(f"    {c:9} srm={s if s is None else round(s,3)}  fe={f if f is None else round(f,3)}")
        print("(4) fe~JND (spearman, eucl) / srm~JND:")
        for c in ['nofilter', 'window', 'optimal']:
            fj = rr['jnd_correlation'].get(c, {}).get('fe_latent|eucl', {})
            sj = rr['jnd_correlation'].get(c, {}).get('srm|eucl', {})
            print(f"    {c:9} fe={fj.get('spearman')}  srm={sj.get('spearman')}  (n={fj.get('n')})")


if __name__ == "__main__":
    main()
