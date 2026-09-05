"""filter_robustness_arms.py — Intervention identifiability: primary vs motion-regressed FILTER.

External review (2026-08-15) raised the question that supersedes parameter
identifiability: if a reasonable preprocessing choice changes the fitted beta,
does the PHYSICAL stimulus transformation the participant actually saw change too?

Compares, for each participant, the inverse filter derived from the primary
(published) estimate against the one derived from the motion-regressed estimate:

    deutan   P = (beta_s= 6, beta_c=-42)    M = (20, -48)
    protan   P = ( 2, +24)                  M = (22, -24)

Reports
  1. injectivity of the forward map (pre-image uniqueness -- closure invariant A5)
  2. per-hue delta-theta of both filters, signed difference, sign reversals
  3. agreement statistics (mean/max |diff|, Pearson r, cosine, circular corr)
  4. 2x2 cross-evaluation  {F_P, F_M} x {M_P, M_M}  with the no-filter reference

Nothing here refits anything. Both beta pairs are taken as given from
U2_BETA_SIGN_PRESPEC results; this is a deterministic consequence of them.

Output: results/filter_robustness_arms/filter_robustness_arms.json
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 04_distortion_model
# Manuscript: Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability
# Original  : analysis/phase5_filter_optimization/scripts/filter_robustness_arms.py  (development repository, commit 53c81c2)
# Role      : Refits the selected combinations on the head-motion-corrected pipeline (S13 sign stability) -> beta_sign_three_arms.json
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
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from two_comp import forward_2comp, HUE_CANON, THETA_CONF  # noqa: E402

HUE_NAMES = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']

CASES = [
    dict(subject='sub-08', family='deutan',
         primary=(6.0, -42.0), motion=(20.0, -48.0)),
    dict(subject='sub-09', family='protan',
         primary=(2.0, 24.0), motion=(22.0, -24.0)),
]


def wrap180(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0


def display_map(theta, beta_s, beta_c, family):
    """theta_perceived(theta_displayed) = theta + delta-theta(theta)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    return theta + forward_2comp(beta_s, beta_c, family, hues=theta)


def injectivity(beta_s, beta_c, family, n=36000):
    """Is theta -> theta + delta-theta(theta) strictly increasing over the circle?

    If not, the forward map folds and the pre-image is not unique: the closure
    invariant A5 (exact numerical inverse) is not well posed.
    """
    theta = np.linspace(0.0, 360.0, n, endpoint=False)
    mapped = display_map(theta, beta_s, beta_c, family)
    d = np.diff(mapped)
    return dict(
        monotone=bool(np.all(d > 0)),
        min_derivative=float(np.min(d) / (360.0 / n)),
        n_fold_points=int(np.sum(d <= 0)),
    )


def all_preimages(target, beta_s, beta_c, family, n=36000):
    """Every theta_d in [0,360) with theta_d + delta-theta(theta_d) == target (mod 360)."""
    theta = np.linspace(0.0, 360.0, n, endpoint=False)
    res = wrap180(display_map(theta, beta_s, beta_c, family) - target)
    roots = []
    for i in range(n):
        j = (i + 1) % n
        a, b = res[i], res[j]
        if a == 0.0:
            roots.append(theta[i])
        elif a * b < 0 and abs(a - b) < 180.0:  # exclude the +-180 wrap seam
            lo, hi = theta[i], theta[i] + 360.0 / n

            def f(t):
                return wrap180(display_map(t, beta_s, beta_c, family) - target)[0]

            roots.append(float(brentq(f, lo, hi, xtol=1e-9)))
    # deduplicate modulo 360
    out = []
    for r in roots:
        r = r % 360.0
        if not any(abs(wrap180(r - o)) < 1e-6 for o in out):
            out.append(r)
    return sorted(out)


def build_filter(beta_s, beta_c, family):
    """Pre-image filter: delta-theta applied to each rendered canonical hue.

    Picks the root nearest the target when several exist, matching the +-60 deg
    bracketed search used by exp2_compute_preimage.py.
    """
    pre, n_roots = [], []
    for t in HUE_CANON:
        roots = all_preimages(t, beta_s, beta_c, family)
        n_roots.append(len(roots))
        nearest = min(roots, key=lambda r: abs(wrap180(r - t)))
        pre.append(nearest)
    pre = np.array(pre)
    perceived = display_map(pre, beta_s, beta_c, family)
    residual = wrap180(perceived - HUE_CANON)
    return dict(
        beta=(float(beta_s), float(beta_c)),
        preimage=pre.tolist(),
        delta_apply=wrap180(pre - HUE_CANON).tolist(),
        max_residual=float(np.max(np.abs(residual))),
        n_preimage_roots=n_roots,
        unique_preimage=bool(max(n_roots) == 1),
    )


def circ_corr(a_deg, b_deg):
    """Jammalamadaka circular correlation between two angular series."""
    a, b = np.deg2rad(a_deg), np.deg2rad(b_deg)

    def cmean(x):
        return np.arctan2(np.mean(np.sin(x)), np.mean(np.cos(x)))

    da, db = a - cmean(a), b - cmean(b)
    num = np.sum(np.sin(da) * np.sin(db))
    den = np.sqrt(np.sum(np.sin(da) ** 2) * np.sum(np.sin(db) ** 2))
    return float(num / den) if den > 0 else float('nan')


def cross_evaluate(filt, beta_s, beta_c, family):
    """Apply a filter's pre-image under a (possibly different) model.

    Returns the residual hue error the observer would experience, in degrees.
    """
    pre = np.array(filt['preimage'])
    return wrap180(display_map(pre, beta_s, beta_c, family) - HUE_CANON)


def main():
    out_dir = Path(__file__).parent.parent / 'results' / 'filter_robustness_arms'
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    for case in CASES:
        sub, fam = case['subject'], case['family']
        bP, bM = case['primary'], case['motion']

        injP, injM = injectivity(*bP, fam), injectivity(*bM, fam)
        fP, fM = build_filter(*bP, fam), build_filter(*bM, fam)

        dP = np.array(fP['delta_apply'])
        dM = np.array(fM['delta_apply'])
        diff = wrap180(dM - dP)
        both_nonzero = (np.abs(dP) > 1e-9) & (np.abs(dM) > 1e-9)
        reversed_mask = both_nonzero & (np.sign(dP) != np.sign(dM))

        # 2x2 cross-evaluation + no-filter reference
        no_filter_P = wrap180(forward_2comp(*bP, fam))
        no_filter_M = wrap180(forward_2comp(*bM, fam))
        e_PP = cross_evaluate(fP, *bP, fam)
        e_PM = cross_evaluate(fP, *bM, fam)   # deployed filter, alternative model
        e_MP = cross_evaluate(fM, *bP, fam)
        e_MM = cross_evaluate(fM, *bM, fam)

        report[sub] = dict(
            family=fam,
            beta_primary=list(bP), beta_motion=list(bM),
            injectivity=dict(primary=injP, motion=injM),
            filter_primary=fP, filter_motion=fM,
            per_hue=[
                dict(hue=HUE_NAMES[i], theta=float(HUE_CANON[i]),
                     delta_primary=float(dP[i]), delta_motion=float(dM[i]),
                     difference=float(diff[i]), sign_reversed=bool(reversed_mask[i]))
                for i in range(8)
            ],
            agreement=dict(
                n_sign_reversed=int(reversed_mask.sum()),
                mean_abs_difference=float(np.mean(np.abs(diff))),
                max_abs_difference=float(np.max(np.abs(diff))),
                pearson_r=float(np.corrcoef(dP, dM)[0, 1]),
                cosine=float(dP @ dM / (np.linalg.norm(dP) * np.linalg.norm(dM))),
                circular_r=circ_corr(np.array(fP['preimage']), np.array(fM['preimage'])),
                mean_abs_delta_primary=float(np.mean(np.abs(dP))),
                mean_abs_delta_motion=float(np.mean(np.abs(dM))),
            ),
            cross_evaluation=dict(
                no_filter_under_MP=dict(mean_abs=float(np.mean(np.abs(no_filter_P))),
                                        max_abs=float(np.max(np.abs(no_filter_P)))),
                no_filter_under_MM=dict(mean_abs=float(np.mean(np.abs(no_filter_M))),
                                        max_abs=float(np.max(np.abs(no_filter_M)))),
                F_P_under_M_P=dict(mean_abs=float(np.mean(np.abs(e_PP))),
                                   max_abs=float(np.max(np.abs(e_PP)))),
                F_P_under_M_M=dict(mean_abs=float(np.mean(np.abs(e_PM))),
                                   max_abs=float(np.max(np.abs(e_PM))),
                                   per_hue=e_PM.round(3).tolist()),
                F_M_under_M_P=dict(mean_abs=float(np.mean(np.abs(e_MP))),
                                   max_abs=float(np.max(np.abs(e_MP))),
                                   per_hue=e_MP.round(3).tolist()),
                F_M_under_M_M=dict(mean_abs=float(np.mean(np.abs(e_MM))),
                                   max_abs=float(np.max(np.abs(e_MM)))),
            ),
        )

        # ---- console ----
        print(f"\n{'=' * 74}\n{sub}  ({fam})   primary {bP}   motion {bM}\n{'=' * 74}")
        print(f"  injectivity   primary: monotone={injP['monotone']}  "
              f"min d(theta_perc)/d(theta_disp)={injP['min_derivative']:+.3f}")
        print(f"                motion : monotone={injM['monotone']}  "
              f"min d(theta_perc)/d(theta_disp)={injM['min_derivative']:+.3f}")
        print(f"  pre-image unique   primary={fP['unique_preimage']}  motion={fM['unique_preimage']}"
              f"   (roots: {fP['n_preimage_roots']} / {fM['n_preimage_roots']})")
        print(f"\n  {'hue':9s} {'dtheta_P':>10s} {'dtheta_M':>10s} {'diff':>9s}   reversed")
        for i in range(8):
            print(f"  {HUE_NAMES[i]:9s} {dP[i]:+10.2f} {dM[i]:+10.2f} {diff[i]:+9.2f}   "
                  f"{'YES' if reversed_mask[i] else ''}")
        a = report[sub]['agreement']
        print(f"\n  sign reversals {a['n_sign_reversed']}/8   "
              f"mean|diff| {a['mean_abs_difference']:.2f} deg   max|diff| {a['max_abs_difference']:.2f} deg")
        print(f"  Pearson r {a['pearson_r']:+.3f}   cosine {a['cosine']:+.3f}   "
              f"circular r {a['circular_r']:+.3f}")
        print(f"  filter magnitude  mean|dtheta_P| {a['mean_abs_delta_primary']:.2f} deg   "
              f"mean|dtheta_M| {a['mean_abs_delta_motion']:.2f} deg")
        c = report[sub]['cross_evaluation']
        print(f"\n  2x2 cross-evaluation -- residual hue error (mean|.| deg, max|.| deg)")
        print(f"    {'':16s} {'model M_P':>20s} {'model M_M':>20s}")
        print(f"    {'no filter':16s} "
              f"{c['no_filter_under_MP']['mean_abs']:9.2f} /{c['no_filter_under_MP']['max_abs']:8.2f} "
              f"{c['no_filter_under_MM']['mean_abs']:9.2f} /{c['no_filter_under_MM']['max_abs']:8.2f}")
        print(f"    {'filter F_P':16s} "
              f"{c['F_P_under_M_P']['mean_abs']:9.2f} /{c['F_P_under_M_P']['max_abs']:8.2f} "
              f"{c['F_P_under_M_M']['mean_abs']:9.2f} /{c['F_P_under_M_M']['max_abs']:8.2f}")
        print(f"    {'filter F_M':16s} "
              f"{c['F_M_under_M_P']['mean_abs']:9.2f} /{c['F_M_under_M_P']['max_abs']:8.2f} "
              f"{c['F_M_under_M_M']['mean_abs']:9.2f} /{c['F_M_under_M_M']['max_abs']:8.2f}")

    (out_dir / 'filter_robustness_arms.json').write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out_dir / 'filter_robustness_arms.json'}")


if __name__ == '__main__':
    main()
