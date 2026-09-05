#!/usr/bin/env python3
"""
frozen_disparity_label_check.py — 동결 투영에서 *disparity* 는 색 라벨에 민감한가 (2026-08-05)

동기
----
`color_correspondence_loro.py` 는 투영 동결이 재적합의 라벨 흡수를 없앤다는 것을
**RDM 상관** 지표로 보였다. 그러나 논문 추정량은 **Procrustes disparity** 이고, 병행
분석(다른 터미널)의 "동결 LORO" 행도 disparity 기반이다.

disparity 는 평가 시점에 `orthogonal_procrustes` 로 **직교회전을 한 번 더 적합**한다.
투영 P 를 동결해도 이 회전이 남아 있으므로, 라벨 순열을 이 회전이 흡수할 수 있다.
그렇다면 "동결 disparity" 의 유의성은 색 대응이 아니라 **배치 형상(configuration shape)**
의 차이를 뜻하게 된다. 이 스크립트가 그것을 직접 판정한다.

설계
----
`color_correspondence_loro.py` 와 **동일한 LORO 동결 루프** (7 subject fold × 6 run fold,
SRM·참조는 학습 6 HC × 학습 5 run, 투영 P 는 대상자의 학습 5 run 에서 적합·동결,
평가는 held-out run). 같은 fold 위에서 두 지표를 나란히 계산한다:

  rdm        : RDM 상관 (직교불변 → 회전이 흡수 못 함)          [양성 대조]
  disparity  : Procrustes disparity (평가 시 회전 재적합)        [논문 추정량]

귀무는 둘 다 **대상자의 held-out run 색 라벨만** 순열 (P·SRM·참조 고정).
z 는 항상 '클수록 색 특이적 정렬이 강함' 부호로 통일.

판정
----
  rdm z > 0 인데 disparity z ≈ 0  →  동결해도 **disparity 는 라벨에 둔감**.
      그 경우 동결 LORO disparity 의 유의성은 색 대응이 아니라 형상 차이의 증거다.
  둘 다 z > 0  →  동결이 disparity 의 색 특이성까지 회복시킨다.

출력: analysis/validation/results/frozen_disparity_label_check.json
실행: conda run -n srm python analysis/validation/scripts/frozen_disparity_label_check.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/frozen_disparity_label_check.py  (development repository, commit 53c81c2)
# Role      : Label-permutation check of the frozen disparity.
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
OUT = ROOT / "analysis/validation/results/frozen_disparity_label_check.json"

HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
K_SRM = {"V1": 4, "V2": 4, "V3": 3, "V4": 3}
MIN_VOX = 20
N_COLOR = 8
SEED = 42
METRICS = ["rdm", "disparity"]


def load(subject, roi):
    p = DATA_DIR / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)
    return None if (a.shape[2] < MIN_VOX or a.shape[0] < 6) else a


def normalize(X):
    Xc = X - X.mean(0)
    n = np.linalg.norm(Xc, "fro")
    return Xc / n if n > 0 else Xc


def rdm(X):
    return pdist(normalize(X), metric="euclidean")


def disparity(X, Y):
    Xn, Yn = normalize(X), normalize(Y)
    R, _ = orthogonal_procrustes(Xn, Yn)
    return float(np.linalg.norm(Xn @ R - Yn, "fro"))


def fit_projection(pat, s_pinv):
    u, _, vt = np.linalg.svd(pat.T @ s_pinv, full_matrices=False)
    return u @ vt


def crawford_howell(score, controls):
    n = len(controls)
    m, s = float(np.mean(controls)), float(np.std(controls, ddof=1))
    if s == 0 or not np.isfinite(s):
        return float("nan"), float("nan")
    t = (score - m) / (s * np.sqrt((n + 1) / n))
    return float(t), float(t_dist.cdf(t, n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(SEED)
    perms = np.array([rng.permutation(N_COLOR) for _ in range(args.n_perm)])

    out = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "analysis/validation/scripts/frozen_disparity_label_check.py",
        "n_perm": args.n_perm, "seed": SEED, "k_srm": K_SRM,
        "design": ("color_correspondence_loro.py 와 동일한 LORO 동결 루프. 같은 fold 위에서 "
                   "rdm(직교불변) 과 disparity(평가 시 회전 재적합) 를 나란히 계산."),
        "question": "투영을 동결해도 Procrustes disparity 가 색 라벨에 민감한가.",
        "z_sign": "클수록 색 특이적 정렬이 강함 (disparity 는 낮을수록 좋으므로 부호 반전)",
    }, "results": {}}

    for roi in ROIS:
        lab = ROI_LABEL[roi]
        raw = [(s, load(s, roi)) for s in HC]
        hc_used = [s for s, v in raw if v is not None]
        hc = [v for _, v in raw if v is not None]
        cvd = {s: v for s, v in ((s, load(s, roi)) for s in CVD) if v is not None}
        k, n, n_run = K_SRM[roi], len(hc), hc[0].shape[0]

        acc = {m: {nm: {"obs": [], "null": []} for nm in hc_used + list(cvd)}
               for m in METRICS}

        for i in range(n):
            tr_subj = [j for j in range(n) if j != i]
            for r in range(n_run):
                tr_run = [q for q in range(n_run) if q != r]
                srm = SRM(n_iter=10, features=k)
                srm.fit([hc[j][tr_run].mean(0).T for j in tr_subj])
                s_pinv = np.linalg.pinv(srm.s_)
                aligned = [(srm.w_[t].T @ hc[j][r].T).T for t, j in enumerate(tr_subj)]
                ref_pat = np.mean(aligned, axis=0)
                ref_rdm = np.mean([rdm(a) for a in aligned], axis=0)

                for name, arr in [(hc_used[i], hc[i])] + list(cvd.items()):
                    P = fit_projection(arr[tr_run].mean(0), s_pinv)   # 동결
                    Y = arr[r] @ P                                     # (8, k)
                    # rdm: 직교불변 → 행 재배열이 곧 라벨 순열
                    D = rdm(Y)
                    acc["rdm"][name]["obs"].append(float(np.corrcoef(D, ref_rdm)[0, 1]))
                    Yn = normalize(Y)
                    dm = np.linalg.norm(Yn[:, None, :] - Yn[None, :, :], axis=-1)
                    iu = np.triu_indices(N_COLOR, 1)
                    acc["rdm"][name]["null"].append(np.array(
                        [np.corrcoef(dm[np.ix_(p, p)][iu], ref_rdm)[0, 1] for p in perms]))
                    # disparity: 평가 시 회전 재적합 (이게 흡수하는지가 쟁점)
                    acc["disparity"][name]["obs"].append(disparity(Y, ref_pat))
                    acc["disparity"][name]["null"].append(np.array(
                        [disparity(Y[p], ref_pat) for p in perms]))

        entry = {"k": k, "n_hc": n, "hc_used": hc_used, "metrics": {}}
        for met in METRICS:
            higher = (met == "rdm")

            def z_of(name, n_folds):
                a = acc[met][name]
                o = float(np.mean(a["obs"]))
                st = np.stack(a["null"])
                nl = (np.mean(st.reshape(n_folds, -1, args.n_perm), axis=(0, 1))
                      if n_folds > 1 else np.mean(st, axis=0))
                m, sd = float(nl.mean()), float(nl.std(ddof=1))
                z = ((o - m) / sd) if higher else ((m - o) / sd)
                hits = (nl >= o).sum() if higher else (nl <= o).sum()
                return {"z": round(float(z), 3), "observed": round(o, 4),
                        "null_mean": round(m, 4), "null_sd": round(sd, 4),
                        "p_perm": round(float((1 + hits) / (1 + len(nl))), 4)}

            hc_rec = {s: z_of(s, 1) for s in hc_used}
            vals = np.array([hc_rec[s]["z"] for s in hc_used], float)
            t_hc = float(vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals))))
            p_hc = float(1 - t_dist.cdf(t_hc, len(vals) - 1))
            cvd_rec = {}
            for s in cvd:
                rec = z_of(s, n)
                t_ch, p_ch = crawford_howell(rec["z"], vals)
                cvd_rec[s] = {"cvd_type": CVD[s], **rec,
                              "t_crawford_howell": round(t_ch, 3),
                              "p_one_tailed_lower": round(p_ch, 4)}
            entry["metrics"][met] = {
                "hc_z": {"mean": round(float(vals.mean()), 3),
                         "sd": round(float(vals.std(ddof=1)), 3),
                         "t_vs_zero": round(t_hc, 3), "p_vs_zero": round(p_hc, 4),
                         "observed_mean": round(float(np.mean(
                             [hc_rec[s]["observed"] for s in hc_used])), 4),
                         "null_mean_mean": round(float(np.mean(
                             [hc_rec[s]["null_mean"] for s in hc_used])), 4),
                         "n_p_perm_lt_05": int(sum(hc_rec[s]["p_perm"] < 0.05
                                                    for s in hc_used)),
                         "per_subject": hc_rec},
                "cvd": cvd_rec}
        out["results"][lab] = entry
        print(f"[{lab}] done", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSAVED {OUT}\n")

    for met in METRICS:
        print(f"\n=== {met} (동결 투영) ===")
        print(f"{'ROI':5s} {'HC obs':>9s} {'HC null':>9s} {'HC z':>14s} {'p(z>0)':>8s} | "
              f"{'subject':9s} {'z':>6s} {'p_perm':>7s} {'CH p_low':>9s}")
        for lab, e in out["results"].items():
            h = e["metrics"][met]["hc_z"]
            head = (f"{lab:5s} {h['observed_mean']:+9.4f} {h['null_mean_mean']:+9.4f} "
                    f"{h['mean']:+6.2f}±{h['sd']:<6.2f} {h['p_vs_zero']:8.4f} | ")
            for s, r in e["metrics"][met]["cvd"].items():
                print(head + f"{s:9s} {r['z']:6.2f} {r['p_perm']:7.4f} "
                      f"{r['p_one_tailed_lower']:9.4f}")
                head = " " * len(head)


if __name__ == "__main__":
    main()
