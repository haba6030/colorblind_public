#!/usr/bin/env python
"""S19: held-out test-loss across ALL gate-passing candidates (not just the 3).

Purpose (user 2026-06-02): corroborate that the stability-selected candidate also
ranks well on held-out predictive goodness — i.e. selection was not "stability alone".
This is CHARACTERIZATION of the candidate landscape, NOT a re-selection (§0).

Method (reuses s18 machinery):
  - candidate = a loss combo (γ pairs × RDM ROI), noLOCO, with an RDM atom.
  - per fold (leave-one-HC-out, 7-fold): refit (β_s,β_c) on 6 HC (combined γ+RDM),
    evaluate on held-out HC.
  - PRIMARY cross-candidate rank metric = grid-null **percentile** (within-ROI rank,
    comparable across ROIs). ΔL vs (0,0) reported alongside BUT flagged: ΔL is NOT
    comparable across ROIs (floor=1.0 common, but achievable depth is ROI-dependent;
    S08-robust uses V2, S08-stable uses V1). Always show each candidate's ROI.
    (Note: this inverts the s18 roles — there ΔL was primary, percentile the floor
    de-confounder. For CROSS-ROI ranking percentile is the comparable one.)

Pre-committed interpretation (decided before running; §0 guard):
  - Expected (given Test 2a ~20° broad basin): candidates CLUSTER → held-out goodness
    is non-discriminative among gate-passing candidates → stability was a reasonable
    tiebreak among goodness-equivalent candidates. (This already answers "stability
    alone, OK?".)
  - If a non-selected combo clearly tops the stability-pick → reportable limitation
    ("goodness rank ≠ stability rank"), NOT a re-selection (§0 forbids).
  - A sharp clean ranking crowning the selected candidate is SUSPECT (likely ROI
    difficulty confound, not value quality).

Scope: only noLOCO + has-RDM combos are cleanly rankable (S08 ~30, S09 3). γ-only
combos → separate γ-ΔL section. LOCO combos → excluded (held-out LOCO degenerate).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/s19_allcandidate_heldout.py  (development repository, commit 53c81c2)
# Role      : Held-out statistics for every candidate (grid percentile, Delta-L, folds) -> s19_allcandidate_heldout.json
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
import itertools
import numpy as np

import s18_heldout_predictive as s18
import s17_hc_loo as s17

HC_SUBJS = s18.HC_SUBJS
OUT = "../results/s10_inclusion/s19_allcandidate_heldout"

REFERENCE = {  # combo_label -> role tag
    ('sub-08', 'γOY|RDMV2|noLOCO'): 'S08-robust ★ (selected, stability-pick)',
    ('sub-08', 'γALL|RDMV1|noLOCO'): 'S08-stable/βs-dom (dropped 2026-06)',
    ('sub-09', 'γALL|RDMV1|noLOCO'): 'S09-primary ★ (selected)',
}


# --- combo enumeration (copied from s10b_v6_pca_rdm to avoid import side-effects) ---
def enumerate_combos_sub08():
    gamma_opts = [[], ['OY'], ['YG'], ['YP'], ['OY', 'YG', 'YP'], ['ALL']]
    rdm_opts = [[], ['V1'], ['V2'], ['V3'], ['V4'], ['V1', 'V4']]
    loco_opts = [[], ['V4']]
    out = []
    for g_a, r_r, l in itertools.product(gamma_opts, rdm_opts, loco_opts):
        if not g_a and not r_r and not l:
            continue
        g_label = ','.join(g_a) if g_a else '_'
        r_label = '+'.join(r_r) if r_r else '_'
        l_label = 'LOCO' if l else 'noLOCO'
        out.append({'gamma_pairs': g_a, 'rdm_rois': r_r, 'loco_v4': bool(l),
                    'label': f"γ{g_label}|RDM{r_label}|{l_label}"})
    return out


def enumerate_combos_sub09():
    out = []
    for inc_g, inc_r, inc_l in itertools.product([False, True], repeat=3):
        if not (inc_g or inc_r or inc_l):
            continue
        out.append({'gamma_pairs': ['GB'] if inc_g else [],
                    'rdm_rois': ['V1'] if inc_r else [],
                    'loco_v4': inc_l,
                    'label': f"γ{'GB' if inc_g else '_'}|"
                             f"RDM{'V1' if inc_r else '_'}|"
                             f"{'LOCO' if inc_l else 'noLOCO'}"})
    for inc_r, inc_l in itertools.product([False, True], repeat=2):
        out.append({'gamma_pairs': ['ALL'],
                    'rdm_rois': ['V1'] if inc_r else [],
                    'loco_v4': inc_l,
                    'label': f"γALL|RDM{'V1' if inc_r else '_'}|"
                             f"{'LOCO' if inc_l else 'noLOCO'}"})
    return out


def eval_combo(combo, subject, family, cvd_amps, hc_amps_by_roi,
               K_by_roi, C_by_roi, cvd_jnd):
    """Combined-variant held-out eval for one combo. Returns summary dict."""
    cand = {'gamma_pairs': combo['gamma_pairs'], 'rdm_rois': combo['rdm_rois'],
            'family': family}
    has_rdm = bool(combo['rdm_rois'])
    has_gamma = bool(combo['gamma_pairs'])
    roi = combo['rdm_rois'][0] if has_rdm else None

    # full-pool fit (all 7 HC)
    atoms_full = s18.build_atoms('combined', cand, HC_SUBJS, cvd_amps,
                                 hc_amps_by_roi, K_by_roi, C_by_roi, cvd_jnd)
    full_fit = s18.composite_argmin(atoms_full, family) if atoms_full else None

    pcts, dL_rdm, gdeltas = [], [], []
    for held in HC_SUBJS:
        train = [h for h in HC_SUBJS if h != held]
        atoms = s18.build_atoms('combined', cand, train, cvd_amps,
                                hc_amps_by_roi, K_by_roi, C_by_roi, cvd_jnd)
        fit = s18.composite_argmin(atoms, family) if atoms else None
        if fit is None:
            continue
        delta = s18.forward_2comp(fit['beta_s'], fit['beta_c'], family)
        if has_rdm and roi in cvd_amps:
            rr = s18.rdm_heldout_eval(fit, roi, cvd_amps[roi], held,
                                      hc_amps_by_roi, C_by_roi[roi],
                                      K_by_roi[roi], family)
            if rr is not None:
                pcts.append(rr['grid_null_percentile'])
                dL_rdm.append(rr['L_rdm_test'] - rr['L_rdm_test_at_00'])
        if has_gamma:
            train_jnd = [h for h in train if h in s18.HC_JND_SUBJS]
            _, train_sd = s18.jnd_baseline_from_pool(train_jnd)
            heldout_jnd = (s18.load_jnd_per_pair(held)
                           if held in s18.HC_JND_SUBJS else None)
            Lg = s18.gamma_heldout_loss(delta, combo['gamma_pairs'],
                                        cvd_jnd, heldout_jnd, train_sd)
            Lg0 = s18.gamma_heldout_loss(np.zeros(8), combo['gamma_pairs'],
                                         cvd_jnd, heldout_jnd, train_sd)
            if Lg is not None and Lg0 is not None:
                gdeltas.append(Lg - Lg0)

    def med(x):
        return float(np.median(x)) if x else None

    return {
        'label': combo['label'], 'subject': subject,
        'gamma_pairs': combo['gamma_pairs'], 'rdm_rois': combo['rdm_rois'],
        'loco': combo['loco_v4'], 'roi': roi,
        'has_rdm': has_rdm, 'has_gamma': has_gamma,
        'full_fit': ({'beta_s': full_fit['beta_s'], 'beta_c': full_fit['beta_c']}
                     if full_fit else None),
        'rdm_pct_med': med(pcts),               # PRIMARY cross-candidate rank (within-ROI)
        'rdm_dL_med': med(dL_rdm),              # ΔL vs (0,0); NOT cross-ROI comparable
        'rdm_folds_beat00': float(np.mean(np.array(dL_rdm) < 0)) if dL_rdm else None,
        'gamma_dL_med': med(gdeltas),
        'gamma_folds_beat00': float(np.mean(np.array(gdeltas) < 0)) if gdeltas else None,
        'n_folds_rdm': len(pcts),
        'role': REFERENCE.get((subject, combo['label']), ''),
    }


def make_md(results):
    L = ["# S19 — Held-out test-loss across ALL gate-passing candidates", "",
         "_Characterization of the candidate landscape (NOT re-selection, §0). "
         "Answers: did stability-pick also rank well on held-out goodness?_", "",
         "**Rank metric = grid-null percentile** (within-ROI, cross-ROI comparable; "
         "LOW = value beats most arbitrary shifts for that ROI). ΔL vs (0,0) shown "
         "alongside but is **NOT cross-ROI comparable** (ROI-difficulty confound — "
         "achievable depth below the 1.0 floor differs by ROI; ROI shown per row).", ""]
    for subject in ['sub-08', 'sub-09']:
        rows = [r for r in results if r['subject'] == subject
                and not r['loco'] and r['has_rdm'] and r['rdm_pct_med'] is not None]
        rows.sort(key=lambda r: r['rdm_pct_med'])
        L += [f"## {subject} — noLOCO + has-RDM combos ranked by held-out percentile "
              f"(n={len(rows)})", "",
              "| rank | combo | ROI | full-fit (β_s,β_c) | RDM pct med | folds beat(0,0) | ΔL vs(0,0) med | γ ΔL med | role |",
              "|---|---|---|---|---|---|---|---|---|"]
        for i, r in enumerate(rows, 1):
            ff = (f"({r['full_fit']['beta_s']:.0f},{r['full_fit']['beta_c']:.0f})"
                  if r['full_fit'] else "—")
            g = f"{r['gamma_dL_med']:+.2f}" if r['gamma_dL_med'] is not None else "—"
            fb = f"{r['rdm_folds_beat00']:.2f}" if r['rdm_folds_beat00'] is not None else "—"
            mark = "**" if r['role'] else ""
            L.append(f"| {i} | {mark}{r['label']}{mark} | {r['roi']} | {ff} | "
                     f"{r['rdm_pct_med']:.3f} | {fb} | {r['rdm_dL_med']:+.3f} | {g} | {r['role']} |")
        # gamma-only noLOCO
        gonly = [r for r in results if r['subject'] == subject and not r['loco']
                 and r['has_gamma'] and not r['has_rdm']]
        gonly.sort(key=lambda r: (r['gamma_dL_med'] if r['gamma_dL_med'] is not None else 9e9))
        if gonly:
            L += ["", f"### {subject} — γ-only noLOCO (no RDM target; γ ΔL only)", "",
                  "| combo | γ ΔL med | folds beat(0,0) |", "|---|---|---|"]
            for r in gonly:
                g = f"{r['gamma_dL_med']:+.2f}" if r['gamma_dL_med'] is not None else "—"
                fb = f"{r['gamma_folds_beat00']:.2f}" if r['gamma_folds_beat00'] is not None else "—"
                L.append(f"| {r['label']} | {g} | {fb} |")
        n_loco = sum(1 for r in results if r['subject'] == subject and r['loco'])
        L += ["", f"_{subject}: {n_loco} LOCO combos excluded from held-out rank "
              f"(LOCO held-out degenerate)._", ""]
    L += ["## Pre-committed interpretation", "",
          "- **Clustering of percentiles** = held-out goodness non-discriminative among "
          "gate-passing candidates (consistent with Test 2a ~20° broad basin) → "
          "**stability was a reasonable tiebreak among goodness-equivalent candidates**.",
          "- A non-selected combo clearly topping the selected one = reportable "
          "limitation (goodness rank ≠ stability rank), NOT a re-selection (§0).",
          "- A sharp clean win for the selected candidate is suspect (ROI confound)."]
    return "\n".join(L)


def main():
    print("S19 — all-candidate held-out test-loss", flush=True)
    cvd_amps_by_subj, hc_amps_by_roi, K_by_roi, C_by_roi, cvd_jnd_by_subj = \
        s17.preload_data(['sub-08', 'sub-09'])
    fam = {'sub-08': 'deutan', 'sub-09': 'protan'}
    enum = {'sub-08': enumerate_combos_sub08(), 'sub-09': enumerate_combos_sub09()}
    results = []
    for subject in ['sub-08', 'sub-09']:
        combos = enum[subject]
        print(f"[{subject}] {len(combos)} combos", flush=True)
        for combo in combos:
            r = eval_combo(combo, subject, fam[subject],
                           cvd_amps_by_subj[subject], hc_amps_by_roi,
                           K_by_roi, C_by_roi, cvd_jnd_by_subj.get(subject))
            results.append(r)
            tag = f"  {combo['label']:24s} pct={r['rdm_pct_med']}" \
                  if r['rdm_pct_med'] is not None else f"  {combo['label']:24s} (no RDM rank)"
            print(tag + (f"  <- {r['role']}" if r['role'] else ""), flush=True)
    with open(OUT + ".json", "w") as f:
        json.dump({'candidates': results}, f, indent=2)
    with open(OUT + ".md", "w") as f:
        f.write(make_md(results))
    print("saved", OUT + ".{json,md}", flush=True)


if __name__ == '__main__':
    main()
