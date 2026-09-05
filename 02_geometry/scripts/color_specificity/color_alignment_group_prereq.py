#!/usr/bin/env python3
"""
color_alignment_group_prereq.py — 집단 수준 전제 검정 (2026-08-05)

`color_alignment_z.py` 에서 개인별 자기-귀무 z 가 HC 에서 0 을 넘지 못했다
(세 지표 × 4 ROI 전부, p_vs0 = .14~.92). 이 실패의 원인은 둘 중 하나다.

  (i)  개인 수준 z 검정이 n=7 로 검정력이 부족하다.
  (ii) 피험자 간 **색 특이적** 대응 자체가 검출 한계 아래다.

둘을 가르기 위해, 정렬 절차(SRM·SVD 투영)를 전혀 쓰지 않는 native voxel 공간에서
HC 21 쌍 **전체**를 한 통계량으로 묶어 집단 라벨 순열을 돌린다. 정렬 단계가 없으므로
"적합된 직교변환이 라벨 순열을 흡수한다"는 교란이 원천적으로 없다.

통계량
------
  r_true = mean over 21 HC pairs of  corr( RDM(A), RDM(B) )
  귀무   : 매 반복마다 **피험자별로 독립적인 색 라벨 순열**을 적용하고 동일 계산
  p      = (1 + #{r_null >= r_true}) / (1 + n_perm)

RDM = 패턴 행렬 중심화 + Frobenius 정규화 후 8 색 간 유클리드 거리 (직교변환 불변).

부수 검정
--------
  · 참조 대비 CVD: 각 CVD 와 7 HC 사이 평균 r 을, HC 각자와 나머지 6 HC 사이 평균 r
    분포(n=7)에 Crawford-Howell one-tailed lower. 색 특이 대응이 CVD 에서 약한가.
  · 8 색 decodability 는 이미 논문에서 확인됨(LORO chance 초과) — 즉 **개인 내** 색
    정보는 존재한다. 여기서 묻는 것은 오직 **피험자 간 색 대응**이다.

출력: analysis/validation/results/color_alignment_group_prereq.json
실행: conda run -n srm python analysis/validation/scripts/color_alignment_group_prereq.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/color_alignment_group_prereq.py  (development repository, commit 53c81c2)
# Role      : Group-level prerequisite for the alignment test.
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
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "analysis/phase1_procrustes_decoding/results/full_dataset_C010"
OUT = ROOT / "analysis/validation/results/color_alignment_group_prereq.json"

HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
MIN_VOX = 20
N_COLOR = 8
SEED = 42


def load(subject, roi):
    p = DATA_DIR / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return None if a.shape[2] < MIN_VOX else a.mean(axis=0)


def rdm(X):
    Xc = X - X.mean(0)
    return pdist(Xc / np.linalg.norm(Xc, "fro"), metric="euclidean")


def mean_pair_r(rdms):
    idx = list(combinations(range(len(rdms)), 2))
    return float(np.mean([np.corrcoef(rdms[i], rdms[j])[0, 1] for i, j in idx])), len(idx)


def crawford_howell(score, controls):
    n = len(controls)
    m, s = float(np.mean(controls)), float(np.std(controls, ddof=1))
    if s == 0:
        return float("nan"), float("nan")
    t = (score - m) / (s * np.sqrt((n + 1) / n))
    return float(t), float(t_dist.cdf(t, n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)

    out = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "analysis/validation/scripts/color_alignment_group_prereq.py",
        "n_perm": args.n_perm, "seed": SEED, "min_voxels": MIN_VOX,
        "space": "native voxel (정렬 절차 없음 — 적합된 직교변환의 라벨 흡수 교란 없음)",
        "statistic": "mean pairwise RDM correlation over all HC pairs; 피험자별 독립 색 라벨 순열 귀무",
        "note": "개인 내 색 decodability 는 논문에서 이미 확인됨. 여기서 묻는 것은 피험자 간 색 대응.",
    }, "results": {}}

    for roi in ROIS:
        lab = ROI_LABEL[roi]
        hc_pats = [(s, load(s, roi)) for s in HC]
        hc_used = [s for s, p in hc_pats if p is not None]
        pats = [p for _, p in hc_pats if p is not None]
        cvd_pats = {s: p for s, p in ((s, load(s, roi)) for s in CVD) if p is not None}

        r_true, n_pairs = mean_pair_r([rdm(p) for p in pats])
        null = np.empty(args.n_perm)
        for b in range(args.n_perm):
            sh = [p[rng.permutation(N_COLOR)] for p in pats]
            null[b], _ = mean_pair_r([rdm(p) for p in sh])
        p_perm = float((1 + (null >= r_true).sum()) / (1 + args.n_perm))
        z_group = float((r_true - null.mean()) / null.std(ddof=1))

        # CVD-vs-HC: 각자와 "나머지 HC" 사이 평균 r
        hc_rdms = [rdm(p) for p in pats]
        hc_to_rest = []
        for i in range(len(hc_rdms)):
            rest = [hc_rdms[j] for j in range(len(hc_rdms)) if j != i]
            hc_to_rest.append(float(np.mean([np.corrcoef(hc_rdms[i], r)[0, 1] for r in rest])))
        cvd_out = {}
        for s, pat in cvd_pats.items():
            rr = rdm(pat)
            score = float(np.mean([np.corrcoef(rr, h)[0, 1] for h in hc_rdms]))
            t, p = crawford_howell(score, np.array(hc_to_rest))
            cvd_out[s] = {"cvd_type": CVD[s], "mean_r_to_hc": round(score, 4),
                          "t_crawford_howell": round(t, 3),
                          "p_one_tailed_lower": round(p, 4)}

        out["results"][lab] = {
            "n_hc": len(pats), "hc_used": hc_used, "n_pairs": n_pairs,
            "r_true": round(r_true, 4),
            "null": {"mean": round(float(null.mean()), 4),
                     "sd": round(float(null.std(ddof=1)), 4),
                     "pct95": round(float(np.percentile(null, 95)), 4)},
            "z_group": round(z_group, 3), "p_permutation": round(p_perm, 4),
            "hc_to_rest_r": {s: round(v, 4) for s, v in zip(hc_used, hc_to_rest)},
            "cvd": cvd_out,
        }
        print(f"[{lab}] r_true={r_true:+.4f}  null={null.mean():+.4f}±{null.std(ddof=1):.4f}  "
              f"z={z_group:+.2f}  p={p_perm:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSAVED {OUT}\n")

    print(f"{'ROI':5s} {'pairs':>5s} {'r_true':>8s} {'null r':>8s} {'z':>6s} {'p_perm':>8s} | "
          f"{'subject':9s} {'type':7s} {'r to HC':>8s} {'t':>6s} {'p_low':>7s}")
    for lab, e in out["results"].items():
        head = (f"{lab:5s} {e['n_pairs']:5d} {e['r_true']:+8.4f} {e['null']['mean']:+8.4f} "
                f"{e['z_group']:+6.2f} {e['p_permutation']:8.4f} | ")
        for s, r in e["cvd"].items():
            print(head + f"{s:9s} {r['cvd_type']:7s} {r['mean_r_to_hc']:+8.4f} "
                  f"{r['t_crawford_howell']:6.2f} {r['p_one_tailed_lower']:7.4f}")
            head = " " * len(head)


if __name__ == "__main__":
    main()
