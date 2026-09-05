"""_arm_agreement.py — how well does the LOCO metric agree across preprocessing arms?

Three questions, all raised 2026-08-15:

1. Bland-Altman agreement between the primary and hmc arms, per participant, so the
   two CVD cases can be read against the HC agreement distribution. Preferred over
   regressing the change on the primary value, which is biased by regression to the
   mean.
2. ICC / correlation between arms: does the metric preserve participant ranking?
3. Per-hue decomposition: adjacent accuracy scores a prediction correct within 45
   deg, so a hue sitting near a decision boundary can flip a whole 1/8 of the score
   for an arbitrarily small change in the voxel pattern. We therefore also report
   the criterion-free mean absolute angular error and a tighter 22.5 deg criterion.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 08_sensitivity
# Manuscript: Supplementary S2 (tab:motion_arms, tab:interp_arms, session-2 endpoints), S13 sign stability, S18 (tab:color_specificity, head-motion column)
# Original  : analysis/future_phase1_sensitivity/scripts/_arm_agreement.py  (development repository, commit 53c81c2)
# Role      : Agreement of the interpolation readout between the arms (Bland-Altman, ICC; descriptive only).
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

import sys as _sys
from pathlib import Path as _Path

# Moved out of docs/PAPER/repro on 2026-08-17 (analysis/future_phase1_sensitivity).
# _repro_util still lives there and stays the single definition of the data roots.
_REPRO = _Path(__file__).resolve().parents[3] / "docs" / "PAPER" / "repro"
_sys.path.insert(0, str(_REPRO))
OUT = _Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

import numpy as np

import _repro_util as U
from _perm_adjacent_arm import (  # noqa: E402
    BASIS_Z, BASIS_OK, CVD, HC, HUES, K, N_COLORS, ROI_DIR, STEP,
    arm_root, fold_pinvs, load,
)

ARMS = ["with_residuals", "hmc_v2"]
ALL = HC + list(CVD)


def loco_detail(amp, pinvs):
    """Per-colour predictions -> adjacent acc, tight acc, absolute angular error."""
    n_runs, n_colors, V = amp.shape
    adj = np.zeros(n_colors)
    tight = np.zeros(n_colors)
    abserr = np.full(n_colors, np.nan)
    for c in range(n_colors):
        train, P = pinvs[c]
        W = P @ amp[:, train, :].reshape(-1, V)
        resp = W @ amp[:, c, :].T
        rz = resp - resp.mean(axis=0, keepdims=True)
        rsd = rz.std(axis=0, keepdims=True)
        ok = rsd[0] > 0
        if not ok.any():
            continue
        rz = np.divide(rz, rsd, out=np.zeros_like(rz), where=rsd > 0)
        corrs = (BASIS_Z @ rz) / K
        corrs[~BASIS_OK, :] = np.nan
        pred = np.nanargmax(corrs, axis=0).astype(float)[ok]
        err = np.abs(pred - HUES[c])
        err = np.minimum(err, 360.0 - err)
        adj[c] = float(np.mean(err <= STEP))
        tight[c] = float(np.mean(err <= STEP / 2))
        abserr[c] = float(np.mean(err))
    return adj, tight, abserr


def icc21(x, y):
    """ICC(2,1) absolute agreement, two arms treated as random raters."""
    m = np.vstack([x, y]).T                     # (n, 2)
    n, k = m.shape
    gm = m.mean()
    ms_r = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
    ms_c = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    ms_e = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum() \
        / ((n - 1) * (k - 1))
    return float((ms_r - ms_e) / (ms_r + (k - 1) * ms_e + k * (ms_c - ms_e) / n))


def main():
    out = {"arms": ARMS, "rois": {}}
    for roi, rdir in ROI_DIR.items():
        det = {}
        for arm in ARMS:
            root = arm_root(arm)
            amps = {s: load(root, s, rdir) for s in ALL}
            pin = fold_pinvs(next(iter(amps.values())).shape[0])
            det[arm] = {s: loco_detail(a, pin) for s, a in amps.items()}

        a0 = np.array([det[ARMS[0]][s][0].mean() for s in ALL])
        a1 = np.array([det[ARMS[1]][s][0].mean() for s in ALL])
        diff, mean = a1 - a0, (a0 + a1) / 2
        hc_d = diff[:len(HC)]
        bias, sd = float(hc_d.mean()), float(hc_d.std(ddof=1))

        e = {"bland_altman": {"hc_bias": round(bias, 4), "hc_sd": round(sd, 4),
                              "hc_loa": [round(bias - 1.96 * sd, 4), round(bias + 1.96 * sd, 4)],
                              "per_subject": {}},
             "agreement": {"pearson_r": round(float(np.corrcoef(a0, a1)[0, 1]), 3),
                           "spearman_rho": round(float(np.corrcoef(
                               np.argsort(np.argsort(a0)), np.argsort(np.argsort(a1)))[0, 1]), 3),
                           "icc_2_1": round(icc21(a0, a1), 3)},
             "per_hue": {}}
        for i, s in enumerate(ALL):
            e["bland_altman"]["per_subject"][s] = {
                "mean": round(float(mean[i]), 4), "diff": round(float(diff[i]), 4),
                "z_vs_hc_loa": round(float((diff[i] - bias) / sd), 2),
                "outside_loa": bool(abs(diff[i] - bias) > 1.96 * sd),
                "group": CVD.get(s, "HC")}

        for s in ALL:
            adj0, t0, er0 = det[ARMS[0]][s]
            adj1, t1, er1 = det[ARMS[1]][s]
            e["per_hue"][s] = {
                "group": CVD.get(s, "HC"),
                "adj_primary": [round(float(v), 3) for v in adj0],
                "adj_hmc": [round(float(v), 3) for v in adj1],
                "adj_delta": [round(float(v), 3) for v in adj1 - adj0],
                "n_hues_changed": int((np.abs(adj1 - adj0) > 1e-9).sum()),
                "tight22_primary": round(float(t0.mean()), 4),
                "tight22_hmc": round(float(t1.mean()), 4),
                "abserr_primary": round(float(np.nanmean(er0)), 2),
                "abserr_hmc": round(float(np.nanmean(er1)), 2)}
        out["rois"][roi] = e

        print(f"\n=== {roi} ===")
        ag = e["agreement"]
        print(f"  arm 일치도: Pearson r={ag['pearson_r']:+.3f}  "
              f"Spearman rho={ag['spearman_rho']:+.3f}  ICC(2,1)={ag['icc_2_1']:+.3f}")
        print(f"  Bland-Altman (HC): bias {bias:+.3f}  SD {sd:.3f}  "
              f"LoA [{bias - 1.96 * sd:+.3f}, {bias + 1.96 * sd:+.3f}]")
        for s in CVD:
            b = e["bland_altman"]["per_subject"][s]
            print(f"    {CVD[s]:6s} diff {b['diff']:+.3f}  z={b['z_vs_hc_loa']:+.2f}  "
                  f"LoA 밖={b['outside_loa']}")

    with open(OUT / "arm_agreement.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote arm_agreement.json")

    for who, sid in [("protan", "09"), ("deutan", "08")]:
        print(f"\n=== {who} hV4 색별 분해 ===")
        _ph = out["rois"]["hV4"]["per_hue"][sid]
        _n = ["red","orange","yellow","green","cyan","blue","purple","magenta"]
        print(f"  {'hue':9s} {'primary':>8s} {'hmc':>8s} {'delta':>8s}")
        for i, nm in enumerate(_n):
            print(f"  {nm:9s} {_ph['adj_primary'][i]:8.3f} {_ph['adj_hmc'][i]:8.3f} {_ph['adj_delta'][i]:+8.3f}")
        print(f"  변한 색 {_ph['n_hues_changed']}/8")
        print(f"  22.5deg 기준 평균: {_ph['tight22_primary']:.3f} -> {_ph['tight22_hmc']:.3f}")
        print(f"  평균 절대 각오차(기준 무관): {_ph['abserr_primary']:.1f} -> {_ph['abserr_hmc']:.1f} deg")
    print("\n=== (구) protan 상세 ===")
    ph = out["rois"]["hV4"]["per_hue"]["09"]
    names = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
    print(f"  {'hue':9s} {'primary':>8s} {'hmc':>8s} {'Δ':>8s}")
    for i, n in enumerate(names):
        print(f"  {n:9s} {ph['adj_primary'][i]:8.3f} {ph['adj_hmc'][i]:8.3f} {ph['adj_delta'][i]:+8.3f}")
    print(f"  변한 색 {ph['n_hues_changed']}/8")
    print(f"  22.5deg 기준: {ph['tight22_primary']:.3f} -> {ph['tight22_hmc']:.3f}")
    print(f"  평균 절대 각오차(기준 무관): {ph['abserr_primary']:.1f}도 -> {ph['abserr_hmc']:.1f}도")


if __name__ == "__main__":
    main()
