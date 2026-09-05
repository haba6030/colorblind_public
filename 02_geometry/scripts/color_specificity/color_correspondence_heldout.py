#!/usr/bin/env python3
"""
color_correspondence_heldout.py — 정렬이 색 대응을 '발견'하는가, '입력받는가' (2026-08-05)

배경
----
`color_alignment_z.py` / `color_alignment_group_prereq.py` 결과:
  · SRM 공간 RDM 대응 r = +.42~+.47 인데 **색 라벨을 섞어도 +.44~+.49** (z≈0)
  · native voxel 공간에서 HC 21 쌍 평균 r = +.033 ~ -.046, 라벨 순열 귀무와 무구분 (p=.23~.72)
그런데 논문은 SRM 공간 cross-subject 8-way decoding 0.665 (chance .125) 를 보고한다.

이 둘은 양립하기 어렵다. 양립하는 유일한 설명은 **정렬 절차가 색 대응을 입력으로 받는
것**이다. SRM 을 시계열이 아니라 조건평균 beta (V, 8) 에 적합하면 열이 이미 색으로
정렬돼 있으므로, 공유공간이 색 대응을 *발견*하는 게 아니라 *가정*한다. 대상 피험자를
붙일 때 쓰는 SVD 투영도 대상의 라벨된 데이터로 적합되므로 같은 문제를 갖는다.

검정
----
run 을 반으로 나눠(A = runs 1--3, B = runs 4--6) **투영을 A 에서 적합해 동결한 뒤
B 에서만 평가**한다. 투영이 동결돼 있으면 B 의 색 라벨 순열을 흡수할 수 없다.

  frozen_projection  (깨끗한 검정)
      학습 6 HC: SRM 을 A 로 적합 → w_j 동결 → aligned_j^B = (w_j^T X_j^B)^T
      참조: ref_rdm^B = mean_j rdm(aligned_j^B)
      대상: P_t = SVD(X_t^A^T · pinv(S)) 로 A 에서 적합 → 동결 → proj_t^B = X_t^B · P_t
      귀무: 대상의 **B 라벨만** 순열 (P_t 는 그대로) → 흡수 불가

  refit_projection   (논문 절차에 대응)
      P_t 를 B 에서 다시 적합. 라벨 순열이 투영 재적합에 흡수될 수 있다.

  frozen 에서 z > 0 이고 refit 에서 z ≈ 0 이면 → 대응은 실재하나 **재적합이 그것을
  지운다**(= 논문 절차가 대응을 측정하지 못한다).
  둘 다 z ≈ 0 이면 → 피험자 간 색 대응 자체가 이 자료에서 검출되지 않는다.

양성 대조
--------
피험자별 **within-subject split-half** RDM 상관 (A vs B) 을 함께 보고한다. 이것이
높은데 between-subject frozen 이 0 이면, 검정력 부족이 아니라 대응 부재다.

출력: analysis/validation/results/color_correspondence_heldout.json
실행: conda run -n srm python analysis/validation/scripts/color_correspondence_heldout.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/color_correspondence_heldout.py  (development repository, commit 53c81c2)
# Role      : Split-half colour correspondence and within-participant RDM reliability (S18).
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[3]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p


import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from brainiak.funcalign.srm import SRM
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "analysis/phase1_procrustes_decoding/results/full_dataset_C010"
OUT = ROOT / "analysis/validation/results/color_correspondence_heldout.json"

HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
K_SRM = {"V1": 4, "V2": 4, "V3": 3, "V4": 3}
MIN_VOX = 20
N_COLOR = 8
HALF_A, HALF_B = [0, 1, 2], [3, 4, 5]
SEED = 42
MODES = ["frozen_projection", "refit_projection"]


def load_halves(subject, roi):
    p = DATA_DIR / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)                                    # (6 runs, 8 colors, V)
    if a.shape[2] < MIN_VOX or a.shape[0] < 6:
        return None
    return a[HALF_A].mean(0), a[HALF_B].mean(0)       # each (8, V)


def normalize(X):
    Xc = X - X.mean(0)
    n = np.linalg.norm(Xc, "fro")
    return Xc / n if n > 0 else Xc


def rdm(X):
    return pdist(normalize(X), metric="euclidean")


def rdm_corr(X, ref):
    return float(np.corrcoef(rdm(X), ref)[0, 1])


def disparity(X, Y):
    Xn, Yn = normalize(X), normalize(Y)
    R, _ = orthogonal_procrustes(Xn, Yn)
    return float(np.linalg.norm(Xn @ R - Yn, "fro"))


def fit_projection(pat, s_pinv):
    u, _, vt = np.linalg.svd(pat.T @ s_pinv, full_matrices=False)
    return u @ vt                                     # (V, k), 직교열


def crawford_howell(score, controls):
    n = len(controls)
    m, s = float(np.mean(controls)), float(np.std(controls, ddof=1))
    if s == 0 or not np.isfinite(s):
        return float("nan"), float("nan")
    t = (score - m) / (s * np.sqrt((n + 1) / n))
    return float(t), float(t_dist.cdf(t, n - 1))


def self_null_z(obs, null):
    nv = np.asarray(null, float)
    nv = nv[np.isfinite(nv)]
    m, sd = float(nv.mean()), float(nv.std(ddof=1))
    if sd == 0 or not np.isfinite(sd) or not np.isfinite(obs):
        return float("nan"), m, sd, float("nan")
    return float((obs - m) / sd), m, sd, float((1 + (nv >= obs).sum()) / (1 + len(nv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(N_COLOR) for _ in range(args.n_perm)]

    out = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "analysis/validation/scripts/color_correspondence_heldout.py",
        "n_perm": args.n_perm, "seed": SEED, "half_a": HALF_A, "half_b": HALF_B,
        "k_srm": K_SRM, "min_voxels": MIN_VOX,
        "question": ("정렬(SRM w_ + SVD 투영)이 피험자 간 색 대응을 발견하는가, "
                     "라벨된 데이터로부터 입력받는가."),
        "modes": {"frozen_projection": "투영을 half A 에서 적합·동결, half B 에서만 평가 (라벨 순열 흡수 불가)",
                  "refit_projection": "투영을 half B 에서 재적합 (논문 절차 대응; 순열 흡수 가능)"},
        "metric": "SRM 공간 RDM 상관 (중심화+Frobenius 정규화 후 유클리드 RDM)",
    }, "results": {}}

    for roi in ROIS:
        lab = ROI_LABEL[roi]
        hc_raw = [(s, load_halves(s, roi)) for s in HC]
        hc_used = [s for s, v in hc_raw if v is not None]
        hc_h = [v for _, v in hc_raw if v is not None]
        cvd_h = {s: v for s, v in ((s, load_halves(s, roi)) for s in CVD) if v is not None}
        k, n = K_SRM[roi], len(hc_h)

        # 양성 대조: within-subject split-half RDM 신뢰도
        rel = {s: round(float(np.corrcoef(rdm(a), rdm(b))[0, 1]), 3)
               for s, (a, b) in zip(hc_used, hc_h)}
        rel.update({s: round(float(np.corrcoef(rdm(a), rdm(b))[0, 1]), 3)
                    for s, (a, b) in cvd_h.items()})

        hc_z = {m: {} for m in MODES}
        cvd_z = {m: {s: [] for s in cvd_h} for m in MODES}

        for i in range(n):
            tr = [j for j in range(n) if j != i]
            srm = SRM(n_iter=10, features=k)
            srm.fit([hc_h[j][0].T for j in tr])                    # half A 로 적합
            s_pinv = np.linalg.pinv(srm.s_)
            aligned_B = [(srm.w_[t].T @ hc_h[j][1].T).T for t, j in enumerate(tr)]
            ref_rdm_B = np.mean([rdm(a) for a in aligned_B], axis=0)

            targets = [(hc_used[i], hc_h[i], "hc")] + [(s, v, "cvd") for s, v in cvd_h.items()]
            for name, (XA, XB), kind in targets:
                for mode in MODES:
                    if mode == "frozen_projection":
                        P = fit_projection(XA, s_pinv)
                        obs = rdm_corr(XB @ P, ref_rdm_B)
                        null = [rdm_corr(XB[pi] @ P, ref_rdm_B) for pi in perms]
                    else:
                        obs = rdm_corr(XB @ fit_projection(XB, s_pinv), ref_rdm_B)
                        null = [rdm_corr(XB[pi] @ fit_projection(XB[pi], s_pinv), ref_rdm_B)
                                for pi in perms]
                    z, m, sd, pp = self_null_z(obs, null)
                    rec = {"z": round(z, 3), "observed": round(obs, 4),
                           "null_mean": round(m, 4), "null_sd": round(sd, 4),
                           "p_perm": round(pp, 4)}
                    if kind == "hc":
                        hc_z[mode][name] = rec
                    else:
                        cvd_z[mode][name].append(rec)

        entry = {"k": k, "n_hc": n, "hc_used": hc_used,
                 "within_subject_split_half_rdm_r": rel, "modes": {}}
        for mode in MODES:
            vals = np.array([hc_z[mode][s]["z"] for s in hc_used], float)
            t_hc = float(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals))))
            p_hc = float(1 - t_dist.cdf(t_hc, len(vals) - 1))
            cvd_out = {}
            for s in cvd_h:
                zs = np.array([r["z"] for r in cvd_z[mode][s]], float)
                t_ch, p_ch = crawford_howell(float(zs.mean()), vals)
                cvd_out[s] = {"cvd_type": CVD[s], "z_mean": round(float(zs.mean()), 3),
                              "z_sd": round(float(zs.std(ddof=1)), 3),
                              "t_crawford_howell": round(t_ch, 3),
                              "p_one_tailed_lower": round(p_ch, 4)}
            entry["modes"][mode] = {
                "hc_z": {"mean": round(float(vals.mean()), 3),
                         "sd": round(float(vals.std(ddof=1)), 3),
                         "t_vs_zero": round(t_hc, 3), "p_vs_zero": round(p_hc, 4),
                         "n_p_perm_lt_05": int(sum(hc_z[mode][s]["p_perm"] < 0.05
                                                   for s in hc_used)),
                         "observed_mean": round(float(np.mean(
                             [hc_z[mode][s]["observed"] for s in hc_used])), 4),
                         "null_mean_mean": round(float(np.mean(
                             [hc_z[mode][s]["null_mean"] for s in hc_used])), 4),
                         "per_subject": hc_z[mode]},
                "cvd": cvd_out}
        out["results"][lab] = entry
        print(f"[{lab}] done  (within-subject split-half r: "
              f"{', '.join(f'{s}={v}' for s, v in rel.items())})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSAVED {OUT}\n")

    for mode in MODES:
        print(f"\n=== {mode} — {out['meta']['modes'][mode]} ===")
        print(f"{'ROI':5s} {'HC obs r':>9s} {'HC null r':>10s} {'HC z':>13s} "
              f"{'p(z>0)':>8s} {'perm sig':>9s}")
        for lab, e in out["results"].items():
            h = e["modes"][mode]["hc_z"]
            print(f"{lab:5s} {h['observed_mean']:+9.4f} {h['null_mean_mean']:+10.4f} "
                  f"{h['mean']:+6.2f}±{h['sd']:<6.2f} {h['p_vs_zero']:8.4f} "
                  f"{h['n_p_perm_lt_05']:d}/{e['n_hc']:<7d}")


if __name__ == "__main__":
    main()
