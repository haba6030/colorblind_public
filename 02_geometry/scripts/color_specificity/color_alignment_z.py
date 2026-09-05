#!/usr/bin/env python3
"""
color_alignment_z.py — 자기-귀무 정규화 색-정렬 강도 z (2026-08-05)

동기
----
`individual_color_label_permutation.py` (2026-07-22) 는 논문 헤드라인
(protan V1 p=.007, deutan V2 p=.040) 이 색 라벨 순열을 통과하지 못함을 보였다
(p_perm .22–.98). 그러나 그 귀무에는 두 결함이 있다.

  (1) **귀무가 색 구조와 SRM 공유공간을 동시에 파괴한다.** 모든 피험자의 라벨을
      섞고 SRM 을 재학습하므로, 귀무 세계에는 정렬 대상인 공유 색 공간 자체가 없다.
  (2) **색과 무관한 개인차가 상쇄되지 않는다.** 실제로 귀무 t 평균이 0 이 아니라
      +0.9~+3.9 였다 — 라벨을 파괴해도 CVD 가 HC 보다 멀다.

개선 통계량
----------
참조 공간을 **고정한 채** 대상 피험자의 색 라벨만 섞는다. 각 피험자가 자기 자신의
순열 분포로 정규화되므로 색-무관 개인차(SNR, 패턴 크기, 정렬 난이도)가 상쇄된다.

    z(s) = [ 관측 정렬도(s) - mean(순열 정렬도(s)) ] / sd(순열 정렬도(s))
           (disparity 처럼 작을수록 좋은 지표는 부호를 뒤집어 계산)

  z > 0  = 참 색 라벨일 때가 섞었을 때보다 뚜렷하게 잘 정렬된다
         = 이 피험자의 패턴이 HC 참조와 **색 특이적으로** 정렬된다.

가설: CVD 는 색 특이적 정렬이 약하다 → z_CVD < z_HC (Crawford-Howell one-tailed lower).

전제 검정 (prerequisite) — 이게 먼저다
--------------------------------------
HC 자신이 z > 0 이어야 한다. HC 에서 색 특이적 정렬이 검출되지 않으면 CVD 가 잃을
색 구조도 없으므로 CVD 검정은 무의미하다. HC z 의 일표본 t 와 피험자별 순열 p 를
함께 보고한다.

세 지표 (전제 실패 시 원인 판별용)
----------------------------------
전제가 깨졌을 때 "데이터에 색 구조가 없다" 와 "통계량이 색 구조를 못 본다" 를
구분해야 하므로, 같은 fold·같은 순열 위에서 세 지표를 나란히 계산한다.

  A. `procrustes_srm`  — **논문 지표**. SRM 투영 후 orthogonal Procrustes disparity.
       ⚠ 정렬 과정에서 자유 직교회전을 적합하므로, 색 라벨 순열의 상당 부분을
         회전으로 흡수할 수 있다. 즉 구성상 라벨에 둔감할 소지가 있다.
  B. `rdm_srm`         — 동일 SRM 공간에서의 RDM 상관. RDM 은 직교변환에 불변이므로
       회전이 흡수하지 못하고, 색 라벨 순열에는 직접 민감하다. → A 의 대조군.
  C. `rdm_voxel`       — SRM 없이 native voxel 공간 RDM 상관. B 가 실패할 때 그것이
       SRM(K=3~4) 차원 축소 탓인지 데이터 탓인지 가른다.

  A 가 z≈0 인데 B/C 가 z>0 이면 → 색 구조는 있고 **논문 지표가 그것을 못 본다**.
  A·B·C 모두 z≈0 이면 → 해당 공간에서 색 특이적 대응 자체가 검출되지 않는다.

RDM 정의: 패턴 행렬을 중심화 + Frobenius 정규화(척도 제거) 후 8 색 간 유클리드
거리. 직교변환 불변이므로 A 와 같은 척도 자유도를 갖되 라벨에는 민감하다.
참조 RDM = 학습 6 HC 의 개별 RDM 평균 (패턴 평균의 RDM 이 아님 — 평균화 붕괴 회피).

설계 (대칭 LOSO — `individual_color_label_permutation.py:run_loso` 와 동일 배치)
------------------------------------------------------------------------------
  fold i (held-out HC_i):
      SRM 을 나머지 6 HC 로 학습 (참 라벨) → ref_i (패턴) / refRDM_i (RDM 평균)
      대상 = {held-out HC_i, sub-08, sub-09} — 모두 **동일하게** SVD 투영
      관측/순열 정렬도를 A·B·C 각각에 대해 계산 (SRM·참조는 재학습하지 않음)

  HC_s  : 자신이 held-out 인 fold 하나의 z
  CVD_s : 7 fold z 의 평균 (논문의 CVD score 정의와 동일)

알려진 비대칭 (보수적 방향)
--------------------------
CVD 는 7 fold 평균 z, HC 는 단일 fold z 이므로 HC z 의 분산이 더 크다 → Crawford-
Howell 분모를 키워 **CVD 유의성을 낮춘다**. 유의가 나오면 이 비대칭 탓은 아니다.
fold 별 z 산포와 최불리 fold z 를 함께 보고한다.

출력
----
analysis/validation/results/color_alignment_z.json

실행
----
conda run -n srm python analysis/validation/scripts/color_alignment_z.py [--n-perm 2000]
(BrainIAK MPI 이슈 시: conda run -n srm mpirun -np 1 python ...)
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/color_alignment_z.py  (development repository, commit 53c81c2)
# Role      : Alignment z-scores across metrics.
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
from scipy.spatial.distance import pdist
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "analysis/phase1_procrustes_decoding/results/full_dataset_C010"
OUT = ROOT / "analysis/validation/results/color_alignment_z.json"

HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}      # sub-10 제외 (near-normal, 논문 제외)
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
K_SRM = {"V1": 4, "V2": 4, "V3": 3, "V4": 3}
MIN_VOX = 20
N_COLOR = 8
SEED = 42

METRICS = ["procrustes_srm", "rdm_srm", "rdm_voxel"]
METRIC_NOTE = {
    "procrustes_srm": "논문 지표 (SRM 투영 + orthogonal Procrustes disparity; 낮을수록 정렬)",
    "rdm_srm": "SRM 공간 RDM 상관 (직교불변·라벨민감; 논문 지표의 대조군)",
    "rdm_voxel": "native voxel 공간 RDM 상관 (SRM 미사용)",
}


def load(subject, roi):
    p = DATA_DIR / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return None if a.shape[2] < MIN_VOX else a.mean(axis=0)      # (8, V)


def normalize(X):
    Xc = X - X.mean(0)
    return Xc / np.linalg.norm(Xc, "fro")


def disparity(X, Y):
    Xn, Yn = normalize(X), normalize(Y)
    R, _ = orthogonal_procrustes(Xn, Yn)
    return float(np.linalg.norm(Xn @ R - Yn, "fro"))


def rdm(X):
    """중심화 + Frobenius 정규화 후 8 색 간 유클리드 거리 (28,). 직교변환 불변."""
    return pdist(normalize(X), metric="euclidean")


def rdm_corr(X, ref_rdm):
    return float(np.corrcoef(rdm(X), ref_rdm)[0, 1])


def crawford_howell(score, controls, tail="lower"):
    n = len(controls)
    m, s = float(np.mean(controls)), float(np.std(controls, ddof=1))
    if s == 0 or not np.isfinite(s):
        return float("nan"), float("nan")
    t = (score - m) / (s * np.sqrt((n + 1) / n))
    p = t_dist.cdf(t, n - 1) if tail == "lower" else 1 - t_dist.cdf(t, n - 1)
    return float(t), float(p)


def self_null_z(observed, null_vals, higher_is_better):
    """자기-귀무 정규화. 반환 z 는 항상 '클수록 색 특이적 정렬이 강함'."""
    nv = np.asarray(null_vals, dtype=float)
    nv = nv[np.isfinite(nv)]
    m, sd = float(nv.mean()), float(nv.std(ddof=1))
    if sd == 0 or not np.isfinite(sd) or not np.isfinite(observed):
        return float("nan"), m, sd, float("nan")
    z = (observed - m) / sd if higher_is_better else (m - observed) / sd
    hits = (nv >= observed).sum() if higher_is_better else (nv <= observed).sum()
    p_perm = float((1 + hits) / (1 + len(nv)))
    return float(z), m, sd, p_perm


def fold_metrics(pat, s_pinv, ref_pat, ref_rdm_srm, ref_rdm_vox, perms):
    """한 fold 에서 대상 피험자의 A·B·C 지표 z 를 계산 (참조 고정, 대상 라벨만 순열)."""
    def project(p_):
        u, _, vt = np.linalg.svd(p_.T @ s_pinv, full_matrices=False)
        return p_ @ (u @ vt)

    proj = project(pat)
    obs = {"procrustes_srm": disparity(proj, ref_pat),
           "rdm_srm": rdm_corr(proj, ref_rdm_srm),
           "rdm_voxel": rdm_corr(pat, ref_rdm_vox)}
    null = {k: [] for k in METRICS}
    for pi in perms:
        sh = pat[pi]
        psh = project(sh)
        null["procrustes_srm"].append(disparity(psh, ref_pat))
        null["rdm_srm"].append(rdm_corr(psh, ref_rdm_srm))
        null["rdm_voxel"].append(rdm_corr(sh, ref_rdm_vox))

    out = {}
    for k in METRICS:
        higher = k != "procrustes_srm"
        z, m, sd, pp = self_null_z(obs[k], null[k], higher)
        out[k] = {"z": z, "observed": obs[k], "null_mean": m, "null_sd": sd, "p_perm": pp}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    perms = [rng.permutation(N_COLOR) for _ in range(args.n_perm)]

    out = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "analysis/validation/scripts/color_alignment_z.py",
        "n_perm": args.n_perm, "seed": SEED,
        "hc": HC, "cvd": list(CVD), "excluded": ["sub-10 (near-normal)"],
        "k_srm": K_SRM, "min_voxels": MIN_VOX,
        "statistic": ("z = (관측 - 순열평균)/순열sd, 부호는 '클수록 색 특이적 정렬 강함'. "
                      "대상 피험자의 색 라벨만 순열하고 SRM 공유공간·참조는 고정. "
                      "CVD z 를 HC z 분포에 Crawford & Howell (1998) one-tailed LOWER."),
        "design": ("대칭 LOSO: 6 HC 로 SRM 학습, held-out HC 와 CVD 를 동일 SVD 투영. "
                   "HC z = 자신이 held-out 인 단일 fold; CVD z = 7 fold 평균 (보수적)."),
        "prerequisite": "HC z > 0 (색 특이적 정렬이 HC 에서 먼저 검출되어야 함).",
        "metrics": METRIC_NOTE,
    }, "results": {}}

    for roi in ROIS:
        lab = ROI_LABEL[roi]
        t0 = time.time()
        hc_all = [(s, load(s, roi)) for s in HC]
        hc_used = [s for s, p in hc_all if p is not None]
        hc_pats = [p for _, p in hc_all if p is not None]
        cvd_pats = {s: p for s, p in ((s, load(s, roi)) for s in CVD) if p is not None}
        k, n = K_SRM[roi], len(hc_pats)

        hc_fold = {s: None for s in hc_used}
        cvd_fold = {s: [] for s in cvd_pats}

        for i in range(n):
            tr_idx = [j for j in range(n) if j != i]
            tr = [hc_pats[j].T for j in tr_idx]                  # (V, 8)
            srm = SRM(n_iter=10, features=k)
            srm.fit(tr)
            aligned = [(srm.w_[t].T @ tr[t]).T for t in range(len(tr))]   # (8, k)
            ref_pat = np.mean(aligned, axis=0)
            ref_rdm_srm = np.mean([rdm(a) for a in aligned], axis=0)
            ref_rdm_vox = np.mean([rdm(hc_pats[j]) for j in tr_idx], axis=0)
            s_pinv = np.linalg.pinv(srm.s_)

            hc_fold[hc_used[i]] = fold_metrics(hc_pats[i], s_pinv, ref_pat,
                                               ref_rdm_srm, ref_rdm_vox, perms)
            for s, pat in cvd_pats.items():
                cvd_fold[s].append(fold_metrics(pat, s_pinv, ref_pat,
                                                ref_rdm_srm, ref_rdm_vox, perms))

        entry = {"k": k, "n_hc": n, "hc_used": hc_used, "metrics": {}}
        for met in METRICS:
            hc_vals = np.array([hc_fold[s][met]["z"] for s in hc_used], dtype=float)
            hc_pp = [hc_fold[s][met]["p_perm"] for s in hc_used]
            t_hc = float(hc_vals.mean() / (hc_vals.std(ddof=1) / np.sqrt(len(hc_vals))))
            p_hc = float(1 - t_dist.cdf(t_hc, len(hc_vals) - 1))

            cvd_out = {}
            for s in cvd_pats:
                zs = np.array([f[met]["z"] for f in cvd_fold[s]], dtype=float)
                t_ch, p_ch = crawford_howell(float(zs.mean()), hc_vals, tail="lower")
                worst = float(zs.max())
                t_w, p_w = crawford_howell(worst, hc_vals, tail="lower")
                cvd_out[s] = {
                    "cvd_type": CVD[s],
                    "z_mean": round(float(zs.mean()), 3),
                    "z_sd_across_folds": round(float(zs.std(ddof=1)), 3),
                    "z_min": round(float(zs.min()), 3), "z_max": round(worst, 3),
                    "p_perm_mean": round(float(np.mean([f[met]["p_perm"]
                                                        for f in cvd_fold[s]])), 4),
                    "t_crawford_howell": round(t_ch, 3),
                    "p_one_tailed_lower": round(p_ch, 4),
                    "sensitivity_worst_fold": {"z": round(worst, 3), "t": round(t_w, 3),
                                               "p": round(p_w, 4)},
                }
            entry["metrics"][met] = {
                "note": METRIC_NOTE[met],
                "hc_z": {
                    "mean": round(float(hc_vals.mean()), 3),
                    "sd": round(float(hc_vals.std(ddof=1)), 3),
                    "min": round(float(hc_vals.min()), 3),
                    "max": round(float(hc_vals.max()), 3),
                    "t_vs_zero": round(t_hc, 3), "p_vs_zero": round(p_hc, 4),
                    "n_p_perm_lt_05": int(sum(p < 0.05 for p in hc_pp)),
                    "per_subject": {s: {kk: round(vv, 4) for kk, vv in
                                        hc_fold[s][met].items()} for s in hc_used},
                },
                "cvd": cvd_out,
            }
        entry["elapsed_s"] = round(time.time() - t0, 1)
        out["results"][lab] = entry
        print(f"[{lab}] {entry['elapsed_s']:.0f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSAVED {OUT}\n")

    for met in METRICS:
        print(f"\n=== {met} — {METRIC_NOTE[met]} ===")
        print(f"{'ROI':5s} {'HC z':>13s} {'p(HC z>0)':>10s} {'HC perm sig':>12s} | "
              f"{'subject':9s} {'type':7s} {'z':>6s} {'z range':>14s} {'t':>6s} {'p_low':>7s}")
        for lab, e in out["results"].items():
            m = e["metrics"][met]
            h = m["hc_z"]
            head = (f"{lab:5s} {h['mean']:6.2f}±{h['sd']:<6.2f} {h['p_vs_zero']:10.4f} "
                    f"{h['n_p_perm_lt_05']:d}/{e['n_hc']:<10d} | ")
            for s, r in m["cvd"].items():
                print(head + f"{s:9s} {r['cvd_type']:7s} {r['z_mean']:6.2f} "
                      f"[{r['z_min']:5.2f},{r['z_max']:5.2f}] "
                      f"{r['t_crawford_howell']:6.2f} {r['p_one_tailed_lower']:7.4f}")
                head = " " * len(head)


if __name__ == "__main__":
    main()
