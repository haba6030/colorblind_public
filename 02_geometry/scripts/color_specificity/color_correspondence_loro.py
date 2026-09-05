#!/usr/bin/env python3
"""
color_correspondence_loro.py — 동결 투영, LORO 판(SNR 손실 최소화) (2026-08-05)

배경
----
`color_correspondence_heldout.py` 는 run 을 반으로 갈라(A=1--3, B=4--6) 투영을 A 에서
적합·동결하고 B 에서 평가했다. 결과: 재적합(논문 절차)에서는 라벨 순열 귀무 r 이
+.29~+.39 로 관측값과 거의 같았고(= 색 대응 측정 불가), 동결에서는 귀무가 ~0 으로
떨어지면서 **V1 에서만** 진짜 대응이 유의했다(HC z=+0.97, p=.014).

그러나 반쪽 분할은 적합·평가 양쪽을 3 run 으로 깎는다. V1 만 살아남은 것이 SNR 손실
때문일 수 있다. 여기서는 같은 논리를 유지하되 데이터를 훨씬 덜 버린다.

설계 (leave-one-run-out 동결)
-----------------------------
  피험자 fold i (held-out HC_i) × run fold r (held-out run):
      학습 run = r 을 뺀 5 run
      SRM 을 학습 HC 6명의 **학습-run 평균 패턴** 으로 적합 → w_j 동결
      참조 : ref_rdm^r = mean_j rdm( (w_j^T X_j^{(r)})^T )      ← held-out run 만 사용
      대상 : P_t = SVD( X_t^{학습run T} · pinv(S) ) 로 5 run 에서 적합 → **동결**
             평가 = rdm_corr( X_t^{(r)} · P_t , ref_rdm^r )
      귀무 : 대상의 held-out run 라벨만 순열. P_t 동결이므로 흡수 불가.
             (X[p] · P = (X · P)[p] 이므로 8x8 거리행렬을 한 번 만들고 재색인 — 순열 비용 ~0)
  6 run fold 의 관측·귀무를 각각 평균한 뒤 z 계산 → 피험자당 z 하나.

  HC z  : 자신이 held-out 인 fold
  CVD z : 7 피험자 fold 평균 (논문의 CVD score 정의와 동일)

비교 대상
--------
같은 루프에서 `refit_projection`(P_t 를 held-out run 에서 재적합 = 논문 절차)도 계산해
동일 SNR 조건에서 동결 vs 재적합을 직접 대비한다.

출력: analysis/validation/results/color_correspondence_loro.json
실행: conda run -n srm python analysis/validation/scripts/color_correspondence_loro.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/color_correspondence_loro.py  (development repository, commit 53c81c2)
# Role      : LORO-frozen colour-correspondence permutation.
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
from scipy.spatial.distance import pdist, squareform
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "analysis/phase1_procrustes_decoding/results/full_dataset_C010"
OUT = ROOT / "analysis/validation/results/color_correspondence_loro.json"

HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
K_SRM = {"V1": 4, "V2": 4, "V3": 3, "V4": 3}
MIN_VOX = 20
N_COLOR = 8
SEED = 42
MODES = ["frozen_projection", "refit_projection"]
IU = np.triu_indices(N_COLOR, k=1)


def load(subject, roi):
    p = DATA_DIR / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)                                   # (6 runs, 8 colors, V)
    return None if (a.shape[2] < MIN_VOX or a.shape[0] < 6) else a


def normalize(X):
    Xc = X - X.mean(0)
    n = np.linalg.norm(Xc, "fro")
    return Xc / n if n > 0 else Xc


def rdm(X):
    return pdist(normalize(X), metric="euclidean")


def dist_matrix(X):
    return squareform(pdist(normalize(X), metric="euclidean"))


def fit_projection(pat, s_pinv):
    u, _, vt = np.linalg.svd(pat.T @ s_pinv, full_matrices=False)
    return u @ vt                                    # (V, k)


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


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
        "script": "analysis/validation/scripts/color_correspondence_loro.py",
        "n_perm": args.n_perm, "seed": SEED, "k_srm": K_SRM, "min_voxels": MIN_VOX,
        "design": ("leave-one-run-out 동결: 투영·SRM 을 5 run 에서 적합하고 held-out 1 run 으로만 평가. "
                   "6 run fold 의 관측·귀무를 평균한 뒤 z. 반쪽 분할 대비 SNR 손실 최소."),
        "modes": {"frozen_projection": "P_t 를 5 학습 run 에서 적합·동결 (라벨 순열 흡수 불가)",
                  "refit_projection": "P_t 를 held-out run 에서 재적합 (논문 절차; 흡수 가능)"},
        "metric": "SRM 공간 RDM 상관 (중심화+Frobenius 정규화 후 유클리드 RDM)",
    }, "results": {}}

    for roi in ROIS:
        lab = ROI_LABEL[roi]
        raw = [(s, load(s, roi)) for s in HC]
        hc_used = [s for s, v in raw if v is not None]
        hc = [v for _, v in raw if v is not None]
        cvd = {s: v for s, v in ((s, load(s, roi)) for s in CVD) if v is not None}
        k, n, n_run = K_SRM[roi], len(hc), hc[0].shape[0]

        acc = {m: {} for m in MODES}          # subject -> {'obs': [...], 'null': [...]} per run fold
        for name in hc_used + list(cvd):
            for m in MODES:
                acc[m][name] = {"obs": [], "null": []}

        for i in range(n):
            tr_subj = [j for j in range(n) if j != i]
            for r in range(n_run):
                tr_run = [q for q in range(n_run) if q != r]
                srm = SRM(n_iter=10, features=k)
                srm.fit([hc[j][tr_run].mean(0).T for j in tr_subj])
                s_pinv = np.linalg.pinv(srm.s_)
                ref = np.mean([rdm((srm.w_[t].T @ hc[j][r].T).T)
                               for t, j in enumerate(tr_subj)], axis=0)

                targets = [(hc_used[i], hc[i])] + list(cvd.items())
                for name, arr in targets:
                    X_tr, X_te = arr[tr_run].mean(0), arr[r]
                    for mode in MODES:
                        if mode == "frozen_projection":
                            P = fit_projection(X_tr, s_pinv)
                            D = dist_matrix(X_te @ P)
                            o = corr(D[IU], ref)
                            nl = np.array([corr(D[np.ix_(p, p)][IU], ref) for p in perms])
                        else:
                            o = corr(rdm(X_te @ fit_projection(X_te, s_pinv)), ref)
                            nl = np.array([corr(rdm(X_te[p] @ fit_projection(X_te[p], s_pinv)),
                                                ref) for p in perms])
                        acc[mode][name]["obs"].append(o)
                        acc[mode][name]["null"].append(nl)

        entry = {"k": k, "n_hc": n, "hc_used": hc_used, "n_run": n_run, "modes": {}}
        for mode in MODES:
            def z_of(name, n_folds):
                a = acc[mode][name]
                o = float(np.mean(a["obs"]))
                nl = np.mean(np.stack(a["null"]).reshape(n_folds, -1, args.n_perm), axis=(0, 1)) \
                    if n_folds > 1 else np.mean(np.stack(a["null"]), axis=0)
                m, sd = float(nl.mean()), float(nl.std(ddof=1))
                z = (o - m) / sd if sd > 0 else float("nan")
                p = float((1 + (nl >= o).sum()) / (1 + len(nl)))
                return {"z": round(z, 3), "observed": round(o, 4),
                        "null_mean": round(m, 4), "null_sd": round(sd, 4), "p_perm": round(p, 4)}

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

            entry["modes"][mode] = {
                "hc_z": {"mean": round(float(vals.mean()), 3),
                         "sd": round(float(vals.std(ddof=1)), 3),
                         "t_vs_zero": round(t_hc, 3), "p_vs_zero": round(p_hc, 4),
                         "n_p_perm_lt_05": int(sum(hc_rec[s]["p_perm"] < 0.05 for s in hc_used)),
                         "observed_mean": round(float(np.mean(
                             [hc_rec[s]["observed"] for s in hc_used])), 4),
                         "null_mean_mean": round(float(np.mean(
                             [hc_rec[s]["null_mean"] for s in hc_used])), 4),
                         "per_subject": hc_rec},
                "cvd": cvd_rec}
        out["results"][lab] = entry
        print(f"[{lab}] done")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nSAVED {OUT}\n")

    for mode in MODES:
        print(f"\n=== {mode} ===")
        print(f"{'ROI':5s} {'HC obs r':>9s} {'HC null r':>10s} {'HC z':>14s} {'p(z>0)':>8s} | "
              f"{'subject':9s} {'type':7s} {'z':>6s} {'t':>6s} {'p_low':>7s}")
        for lab, e in out["results"].items():
            h = e["modes"][mode]["hc_z"]
            head = (f"{lab:5s} {h['observed_mean']:+9.4f} {h['null_mean_mean']:+10.4f} "
                    f"{h['mean']:+6.2f}±{h['sd']:<6.2f} {h['p_vs_zero']:8.4f} | ")
            for s, r in e["modes"][mode]["cvd"].items():
                print(head + f"{s:9s} {r['cvd_type']:7s} {r['z']:6.2f} "
                      f"{r['t_crawford_howell']:6.2f} {r['p_one_tailed_lower']:7.4f}")
                head = " " * len(head)


if __name__ == "__main__":
    main()
