#!/usr/bin/env python3
"""
individual_color_label_permutation.py — 개인 수준 색 라벨 순열 (2026-07-22)

동기
----
canonical `phase2_SRM_across_between/rerun_loo_consistent.py` 는 색 라벨 순열 통제를
이미 수행했으나, 검정 대상이 **CVD 3명 평균(sub-10 포함) vs HC LOO 평균**의
집단 격차였다 (`disparity_diff` p = V1 .327 / V2 .986 / V3 .977 / hV4 .933).

논문의 헤드라인은 그 값이 아니라 **개인별 Crawford--Howell 검정**이다
(protan V1 p = .007, deutan V2 p = .040). 그리고 sub-10 은 near-normal 로
논문에서 제외된 참가자인데 위 순열에는 포함되어 있어 효과가 희석된다.

따라서 여기서는 **논문이 실제로 보고하는 통계량(Crawford--Howell t)** 에 대해,
sub-10 을 제외하고, 개인별로 색 라벨 순열 귀무분포를 만든다.

귀무 구성
--------
canonical 과 동일하게 **모든 피험자(HC 7 + CVD)의 색 라벨을 각각 독립적으로 섞은 뒤**
HC-only SRM 을 재학습한다. 이는 "어떤 색-특이적 구조도 존재하지 않는" 세계이며,
관측 t 가 이 귀무를 넘으면 개인의 disparity 상승이 실제 색 구조에 의존함을 뜻한다.

절차 (canonical LOO-consistent 와 동일)
--------------------------------------
  HC 7명으로 SRM 학습 → HC 는 w_ 로 정렬, CVD 는 SVD 투영
  fold i: ref_i = mean(나머지 6 HC 정렬 패턴)
          hc_disp[i]  = procrustes_disparity(HC_i, ref_i)
          cvd_disp[i] = procrustes_disparity(CVD_j, ref_i)
  CVD score = mean_i cvd_disp[i]
  t = (score - mean(hc_disp)) / (sd(hc_disp) * sqrt((n+1)/n))     (Crawford & Howell 1998)

출력
----
analysis/validation/results/individual_color_label_permutation.json

실행
----
conda run -n srm python analysis/validation/scripts/individual_color_label_permutation.py [--n-perm 1000]
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/individual_color_label_permutation.py  (development repository, commit 53c81c2)
# Role      : Individual-level colour-label permutation of disparity (both estimators).
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
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from brainiak.funcalign.srm import SRM
from scipy.linalg import orthogonal_procrustes
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "analysis/phase1_procrustes_decoding/results/full_dataset_C010"
OUT = ROOT / "analysis/validation/results/individual_color_label_permutation.json"

HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}      # sub-10 제외 (near-normal, 논문 제외)
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
K_SRM = {"V1": 4, "V2": 4, "V3": 3, "V4": 3}
MIN_VOX = 20
SEED = 42

# 논문 보고값 (results_v4.tex:66) — 재현 확인용
PUBLISHED = {("sub-09", "V1"): 0.007, ("sub-08", "V2"): 0.040}
PUBLISHED_LOSO = {("sub-09", "V1"): 0.045, ("sub-08", "V2"): 0.116}
ESTIMATORS = {"common_space": None, "loso_symmetric": None}   # 채워짐


def load(subject, roi):
    p = DATA_DIR / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return None if a.shape[2] < MIN_VOX else a.mean(axis=0)      # (8, V)


def disparity(X, Y):
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    Xn, Yn = Xc / np.linalg.norm(Xc, "fro"), Yc / np.linalg.norm(Yc, "fro")
    R, _ = orthogonal_procrustes(Xn, Yn)
    return float(np.linalg.norm(Xn @ R - Yn, "fro"))


def crawford_howell(score, controls):
    n = len(controls)
    m, s = float(np.mean(controls)), float(np.std(controls, ddof=1))
    if s == 0:
        return float("inf"), 0.0
    t = (score - m) / (s * np.sqrt((n + 1) / n))
    return float(t), float(1 - t_dist.cdf(t, n - 1))            # one-tailed, upper


def run_loso(hc_pats, cvd_pats, k):
    """대칭 LOSO: 6 HC 로 학습, held-out HC 와 CVD 를 **동일하게** SVD 투영.

    common-space 절차는 HC 가 in-sample, CVD 가 out-of-sample 이라 비대칭적이다.
    색 라벨을 섞으면 HC 의 공유 구조만 사라지고 그 비대칭은 남으므로, 순열 귀무가
    색 구조가 아니라 비대칭에 지배된다(귀무 t 가 관측보다 커짐). LOSO 는 양쪽을
    같은 방식으로 배치하므로 순열 귀무가 유효하다.
    """
    n = len(hc_pats)
    hc_disp = np.empty(n)
    cvd_disp = {name: np.empty(n) for name in cvd_pats}
    for i in range(n):
        tr = [hc_pats[j].T for j in range(n) if j != i]
        srm = SRM(n_iter=10, features=k)
        srm.fit(tr)
        ref = np.mean([(srm.w_[t].T @ tr[t]).T for t in range(len(tr))], axis=0)
        s_pinv = np.linalg.pinv(srm.s_)

        def project(pat):
            u, _, vt = np.linalg.svd(pat.T @ s_pinv, full_matrices=False)
            return pat @ (u @ vt)

        hc_disp[i] = disparity(project(hc_pats[i]), ref)
        for name, pat in cvd_pats.items():
            cvd_disp[name][i] = disparity(project(pat), ref)
    return hc_disp, {name: float(v.mean()) for name, v in cvd_disp.items()}


def run_fold(hc_pats, cvd_pats, k):
    """HC-only SRM → LOO-consistent disparities. 반환: (hc_disp[n], {subj: score})"""
    betas = [p.T for p in hc_pats]                               # (V, 8)
    srm = SRM(n_iter=10, features=k)
    srm.fit(betas)
    hc_al = [(srm.w_[i].T @ betas[i]).T for i in range(len(betas))]   # (8, k)

    s_pinv = np.linalg.pinv(srm.s_)
    cvd_al = {}
    for name, pat in cvd_pats.items():
        u, _, vt = np.linalg.svd(pat.T @ s_pinv, full_matrices=False)
        cvd_al[name] = pat @ (u @ vt)

    n = len(hc_al)
    hc_disp = np.empty(n)
    cvd_disp = {name: np.empty(n) for name in cvd_al}
    for i in range(n):
        ref = np.mean([hc_al[j] for j in range(n) if j != i], axis=0)
        hc_disp[i] = disparity(hc_al[i], ref)
        for name in cvd_al:
            cvd_disp[name][i] = disparity(cvd_al[name], ref)
    return hc_disp, {name: float(v.mean()) for name, v in cvd_disp.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "analysis/validation/scripts/individual_color_label_permutation.py",
        "n_perm": args.n_perm,
        "hc": HC, "cvd": list(CVD), "excluded": ["sub-10 (near-normal)"],
        "k_srm": K_SRM, "seed": SEED,
        "statistic": "Crawford & Howell (1998) modified t, one-tailed upper",
        "null": ("색 라벨을 피험자별로 독립 순열 → HC-only SRM 재학습 → 동일 LOO 절차. "
                 "canonical rerun_loo_consistent.py 의 귀무 구성과 동일하되, "
                 "통계량이 집단 격차가 아니라 개인 Crawford-Howell t 이고 sub-10 을 제외함."),
    }, "results": {}}

    for roi in ROIS:
        lab = ROI_LABEL[roi]
        hc_pats = [load(s, roi) for s in HC]
        keep = [i for i, p in enumerate(hc_pats) if p is not None]
        hc_pats = [hc_pats[i] for i in keep]
        hc_used = [HC[i] for i in keep]
        cvd_pats = {s: load(s, roi) for s in CVD}
        cvd_pats = {s: p for s, p in cvd_pats.items() if p is not None}
        k = K_SRM[roi]

        entry = {"k": k, "n_hc": len(hc_pats), "hc_used": hc_used, "estimators": {}}
        for est_name, fn, pub in (("common_space", run_fold, PUBLISHED),
                                  ("loso_symmetric", run_loso, PUBLISHED_LOSO)):
            hc_obs, cvd_obs = fn(hc_pats, cvd_pats, k)
            obs = {}
            for name, score in cvd_obs.items():
                t, p = crawford_howell(score, hc_obs)
                obs[name] = {"disparity": round(score, 4), "t": round(t, 3), "p_ch": round(p, 4)}

            t0 = time.time()
            null_t = {name: [] for name in cvd_obs}
            for _ in range(args.n_perm):
                hc_sh = [q[rng.permutation(8)] for q in hc_pats]
                cvd_sh = {n_: q[rng.permutation(8)] for n_, q in cvd_pats.items()}
                hc_d, cvd_s = fn(hc_sh, cvd_sh, k)
                for name, score in cvd_s.items():
                    null_t[name].append(crawford_howell(score, hc_d)[0])
            elapsed = time.time() - t0

            sub = {}
            for name in cvd_obs:
                nt = np.array([v for v in null_t[name] if np.isfinite(v)])
                p_perm = float((1 + (nt >= obs[name]["t"]).sum()) / (1 + len(nt)))
                sub[name] = {
                    "cvd_type": CVD[name], **obs[name],
                    "p_published": pub.get((name, roi)),
                    "null_t": {"mean": round(float(nt.mean()), 3),
                               "sd": round(float(nt.std(ddof=1)), 3),
                               "pct95": round(float(np.percentile(nt, 95)), 3),
                               "n": int(nt.size)},
                    "p_permutation": round(p_perm, 4),
                }
            entry["estimators"][est_name] = {
                "hc_disparity": {"mean": round(float(hc_obs.mean()), 4),
                                 "sd": round(float(hc_obs.std(ddof=1)), 4)},
                "elapsed_s": round(elapsed, 1), "subjects": sub}
            print(f"[{lab}/{est_name}] {elapsed:.0f}s")
        out["results"][lab] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSAVED {OUT}\n")

    for est in ("common_space", "loso_symmetric"):
        print(f"\n=== {est} ===")
        print(f"{'ROI':5s} {'subject':9s} {'type':7s} {'disp':>7s} {'t':>7s} "
              f"{'p_CH':>7s} {'p_pub':>7s} {'null t':>8s} {'null p95':>9s} {'p_perm':>8s}")
        for lab, e in out["results"].items():
            for name, r in e["estimators"][est]["subjects"].items():
                pub = f"{r['p_published']:.3f}" if r["p_published"] is not None else "·"
                print(f"{lab:5s} {name:9s} {r['cvd_type']:7s} {r['disparity']:7.3f} {r['t']:7.2f} "
                      f"{r['p_ch']:7.4f} {pub:>7s} {r['null_t']['mean']:8.2f} "
                      f"{r['null_t']['pct95']:9.2f} {r['p_permutation']:8.4f}")


if __name__ == "__main__":
    main()
