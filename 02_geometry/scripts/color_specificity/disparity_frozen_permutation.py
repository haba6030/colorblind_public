#!/usr/bin/env python3
"""
disparity_frozen_permutation.py — 동결 투영 색 라벨 순열, disparity 판 (2026-08-05)

배경
----
논문의 기하 결과(Procrustes disparity)에 붙은 색 라벨 순열은 `rerun_loo_consistent.py`
에서 **재적합 투영**으로 계산되었다. `color_correspondence_loro.py` 가 보인 대로 재적합
SVD 투영은 라벨 순열을 흡수한다(귀무가 관측값 쪽으로 끌려 올라간다). 그 결과 현행
순열 p 값은 보수적으로 치우쳐 있다.

  현행(재적합 투영) 순열 p:
    V2 (sub-08)  cvd_score_disp .033  cvd_pairwise .035  disparity_diff .986  hc_loo .894
    V1 (sub-09)  cvd_score_disp .427  cvd_pairwise .077  disparity_diff .327  hc_loo .070

`color_correspondence_loro.py` 는 동결 투영 기계를 갖고 있으나 계산하는 통계량이 RDM
상관이다. 이 스크립트는 **같은 동결 설계에 논문의 통계량(Procrustes disparity)** 을 얹는다.

설계 (leave-one-run-out 동결) — color_correspondence_loro.py 와 동일
--------------------------------------------------------------------
  피험자 fold i (held-out HC_i) × run fold r (held-out run):
      학습 run = r 을 뺀 5 run
      SRM 을 학습 HC 6명의 학습-run 평균 패턴으로 적합 → w_j 동결
      참조 ref^r = mean_j (w_j^T X_j^{(r)})^T                (8, k)  ← held-out run 만
      대상 P_t = SVD( X_t^{학습run T} · pinv(S) ) 로 5 run 에서 적합 → **동결**
      관측 = procrustes_disparity( X_t^{(r)} · P_t , ref^r )
      귀무 = 대상의 held-out run 색 라벨만 순열. P_t 가 동결이라 흡수 불가.
             X[p] · P = (X · P)[p] 이므로 투영을 한 번만 계산하고 행을 재색인한다.

  disparity 는 거리이므로 색 특이성의 방향은 **관측 < 귀무** 다.
      p_perm = (1 + #{null <= obs}) / (1 + n_perm)

보고 항목
---------
  1. p_perm      — disparity 가 색 라벨에 특이적인가 (관측이 순열 귀무보다 작은가)
  2. z           — (obs - null_mean) / null_sd. 음수일수록 색 특이적
  3. Crawford–Howell one-tailed **upper** — CVD disparity 가 HC 분포보다 높은가
                   (논문의 주 검정과 같은 방향)
  비교를 위해 `refit_projection`(논문 절차) 도 같은 루프에서 계산한다.

실행 (서버; BrainIAK 필요 → bare python 금지)
    mpirun -np 1 python analysis/validation/scripts/disparity_frozen_permutation.py

출력: analysis/validation/results/disparity_frozen_permutation.json
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/disparity_frozen_permutation.py  (development repository, commit 53c81c2)
# Role      : Colour-correspondence permutation with frozen vs re-estimated projection (S18, tab:frozen_control, tab:color_specificity).
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
from scipy.stats import t as t_dist

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIRS = [
    ROOT / "analysis/phase1_procrustes_decoding/results/full_dataset_C010",
    ROOT / "analysis/phase1_procrustes_decoding/results/visualization/full_dataset_C010_with_residuals",
]
OUT = ROOT / "analysis/validation/results/disparity_frozen_permutation.json"

HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}
ROIS = ["V1", "V2", "V3", "V4"]
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
K_SRM = {"V1": 4, "V2": 4, "V3": 3, "V4": 3}
MIN_VOX = 20
N_COLOR = 8
SEED = 42
MODES = ["frozen_projection", "refit_projection"]


def resolve_data_dir(cli):
    if cli:
        return Path(cli)
    for d in DEFAULT_DATA_DIRS:
        if d.exists():
            return d
    raise FileNotFoundError(f"no amplitudes directory among {DEFAULT_DATA_DIRS}")


def load(data_dir, subject, roi):
    p = data_dir / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)                                   # (6 runs, 8 colors, V)
    return None if (a.shape[2] < MIN_VOX or a.shape[0] < 6) else a


def normalize(X):
    """열 중심화 + Frobenius 정규화 — rerun_loo_consistent.py 와 동일."""
    Xc = X - X.mean(axis=0)
    n = np.linalg.norm(Xc, "fro")
    return Xc / n if n > 0 else Xc


def disparity(X, Y):
    """Procrustes disparity. rerun_loo_consistent.compute_procrustes_disparity 와 동일."""
    Xn, Yn = normalize(X), normalize(Y)
    R, _ = orthogonal_procrustes(Xn, Yn)
    return float(np.linalg.norm(Xn @ R - Yn, "fro"))


def fit_projection(pat, s_pinv):
    u, _, vt = np.linalg.svd(pat.T @ s_pinv, full_matrices=False)
    return u @ vt                                    # (V, k)


def crawford_howell_upper(score, controls):
    """단측 상단 — CVD 가 HC 분포보다 높은가. 논문 주 검정과 같은 방향."""
    n = len(controls)
    m, s = float(np.mean(controls)), float(np.std(controls, ddof=1))
    if s == 0 or not np.isfinite(s):
        return float("nan"), float("nan")
    t = (score - m) / (s * np.sqrt((n + 1) / n))
    return float(t), float(1.0 - t_dist.cdf(t, n - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data_dir = resolve_data_dir(args.data_dir)
    out_path = Path(args.out) if args.out else OUT
    rng = np.random.default_rng(SEED)
    perms = np.array([rng.permutation(N_COLOR) for _ in range(args.n_perm)])

    out = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "analysis/validation/scripts/disparity_frozen_permutation.py",
        "data_dir": str(data_dir),
        "n_perm": args.n_perm, "seed": SEED, "k_srm": K_SRM, "min_voxels": MIN_VOX,
        "statistic": "Procrustes disparity (열 중심화 + Frobenius 정규화 후 직교 회전 잔차)",
        "design": ("leave-one-run-out 동결: SRM 과 대상 투영을 5 학습 run 에서 적합하고 "
                   "held-out 1 run 으로만 평가. 6 run fold 를 평균한 뒤 z."),
        "modes": {"frozen_projection": "P_t 를 5 학습 run 에서 적합·동결 (라벨 순열 흡수 불가)",
                  "refit_projection": "P_t 를 held-out run 에서 재적합 (논문 절차; 흡수 가능)"},
        "direction": "disparity 는 거리이므로 색 특이성은 관측 < 귀무. p_perm = (1+#{null<=obs})/(1+n_perm)",
    }, "results": {}}

    for roi in ROIS:
        raw = [(s, load(data_dir, s, roi)) for s in HC]
        hc_used = [s for s, v in raw if v is not None]
        hc = [v for _, v in raw if v is not None]
        cvd = {s: v for s, v in ((s, load(data_dir, s, roi)) for s in CVD) if v is not None}
        if len(hc) < 3 or not cvd:
            out["results"][ROI_LABEL[roi]] = {"skipped": "insufficient subjects"}
            continue
        k, n, n_run = K_SRM[roi], len(hc), hc[0].shape[0]

        acc = {m: {name: {"obs": [], "null": []} for name in hc_used + list(cvd)}
               for m in MODES}

        for i in range(n):
            tr_subj = [j for j in range(n) if j != i]
            for r in range(n_run):
                tr_run = [q for q in range(n_run) if q != r]
                srm = SRM(n_iter=10, features=k)
                srm.fit([hc[j][tr_run].mean(0).T for j in tr_subj])
                s_pinv = np.linalg.pinv(srm.s_)
                ref = np.mean([(srm.w_[t].T @ hc[j][r].T).T
                               for t, j in enumerate(tr_subj)], axis=0)      # (8, k)

                targets = [(hc_used[i], hc[i])] + list(cvd.items())
                for name, arr in targets:
                    X_tr, X_te = arr[tr_run].mean(0), arr[r]
                    for mode in MODES:
                        if mode == "frozen_projection":
                            Z = X_te @ fit_projection(X_tr, s_pinv)          # (8, k)
                            o = disparity(Z, ref)
                            nl = np.array([disparity(Z[p], ref) for p in perms])
                        else:
                            o = disparity(X_te @ fit_projection(X_te, s_pinv), ref)
                            nl = np.array([disparity(X_te[p] @ fit_projection(X_te[p], s_pinv),
                                                     ref) for p in perms])
                        acc[mode][name]["obs"].append(o)
                        acc[mode][name]["null"].append(nl)

        entry = {"k": k, "n_hc": n, "hc_used": hc_used, "n_run": n_run, "modes": {}}
        for mode in MODES:
            def stat_of(name, n_folds):
                a = acc[mode][name]
                o = float(np.mean(a["obs"]))
                stack = np.stack(a["null"])
                nl = (np.mean(stack.reshape(n_folds, -1, args.n_perm), axis=(0, 1))
                      if n_folds > 1 else np.mean(stack, axis=0))
                m, sd = float(nl.mean()), float(nl.std(ddof=1))
                return {"observed": round(o, 4),
                        "null_mean": round(m, 4), "null_sd": round(sd, 4),
                        "z": round((o - m) / sd, 3) if sd > 0 else float("nan"),
                        "p_perm": round(float((1 + (nl <= o).sum()) / (1 + len(nl))), 4)}

            hc_rec = {s: stat_of(s, 1) for s in hc_used}
            hc_obs = [hc_rec[s]["observed"] for s in hc_used]
            cvd_rec = {}
            for s in cvd:
                rec = stat_of(s, n)                       # CVD 는 7 피험자 fold 평균
                t, p = crawford_howell_upper(rec["observed"], hc_obs)
                rec.update({"cvd_type": CVD[s],
                            "t_crawford_howell": round(t, 3),
                            "p_one_tailed_upper": round(p, 4)})
                cvd_rec[s] = rec

            entry["modes"][mode] = {
                "hc": {"per_subject": hc_rec,
                       "observed_mean": round(float(np.mean(hc_obs)), 4),
                       "observed_sd": round(float(np.std(hc_obs, ddof=1)), 4),
                       "n_p_perm_lt_05": int(sum(hc_rec[s]["p_perm"] < .05 for s in hc_used))},
                "cvd": cvd_rec,
            }
        out["results"][ROI_LABEL[roi]] = entry
        print(f"[{ROI_LABEL[roi]}] done")
        for mode in MODES:
            for s, rec in out["results"][ROI_LABEL[roi]]["modes"][mode]["cvd"].items():
                print(f"   {mode:18s} {s}  obs={rec['observed']:.3f} "
                      f"null={rec['null_mean']:.3f} p_perm={rec['p_perm']:.3f} "
                      f"C-H p={rec['p_one_tailed_upper']:.4f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
