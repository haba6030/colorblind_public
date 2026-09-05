#!/usr/bin/env python3
"""
within_hc_reliability.py — RDM 신뢰도 패널 (reviewer response 2026-07-22)

동기
----
exp2 후속 분석에서 hV4 voxel-RDM의 HC self-consistency가 ~0 (-0.036)으로 나와
"기하 표적이 불안정하다"는 우려가 제기되었다. 그러나 그 수치는
  (a) voxel 공간   — 논문 Methods의 RDM은 SRM 정렬 공간,
  (b) hV4          — 논문의 기하 결손 ROI는 V2(deutan)/V1(protan)
이라 논문 주장에 직접 대응하지 않는다. 또한 production 적합 손실 L_RDM 은
per-subject voxel PCA(K=6) 공간을 쓴다(s10b_v6_pca_rdm.py).

따라서 실제로 쓰이는 세 공간 각각에서 세 종류의 신뢰도를 계산한다.

계산 항목
--------
1. within_subject_split_half
   피험자 내 3-run vs 3-run (C(6,3)/2 = 10 splits) RDM 상관.
   = **단일 피험자 RDM의 잡음 상한**. 어떤 모형도 이 값을 넘어 개인 RDM을
   설명할 수 없다. Spearman-Brown 으로 6-run 상당으로 보정한 값도 보고.

2. between_hc_loo
   HC i 의 RDM vs 나머지 6 HC 평균 RDM (leave-one-out).
   = 피험자들이 RDM **모양**에 대해 얼마나 동의하는가.

3. hc_mean_split_half
   HC 7명을 3 vs 4 로 나눈 모든 분할(35개)에서 두 평균 RDM 간 상관.
   = ΔRDM 의 **기준항**(HC 평균 RDM)이 얼마나 안정적인가.
   L_RDM 은 ΔRDM = RDM_CVD − mean(RDM_HC) 를 쓰므로 1번과 3번이 함께
   L_RDM 의 신호 대 잡음을 결정한다.

공간
----
- voxel : 원 복셀 패턴 (exp2 후속 분석과 동일 — 비교용)
- pca6  : per-subject voxel PCA top-6 점수 (production L_RDM 과 동일)
- srm   : HC 학습 SRM 공간, K = {V1:4, V2:4, V3:3, hV4:3}
          (SRM 은 피험자 간 공동 학습이 필요하므로 2번에만 적용)

출력
----
analysis/validation/results/within_hc_reliability.json          (canonical)
analysis/phase6_behavioral_analysis/exp2_neural/results/  (사본)

실행
----
conda run -n srm python analysis/validation/scripts/within_hc_reliability.py
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/within_hc_reliability.py  (development repository, commit 53c81c2)
# Role      : Within-control RDM reliability in SRM, PCA and voxel spaces.
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


import json
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "analysis/phase1_procrustes_decoding/results/full_dataset_C010"

HC = [f"sub-0{i}" for i in range(1, 8)]
ROIS = ["V1", "V2", "V3", "V4"]          # V4 = hV4 on disk
ROI_LABEL = {"V1": "V1", "V2": "V2", "V3": "V3", "V4": "hV4"}
K_SRM = {"V1": 4, "V2": 4, "V3": 3, "V4": 3}
K_PCA = 6                                 # s10b_v6_pca_rdm.py 와 동일
MIN_VOX = 20                              # exp2 후속 분석과 동일 (sub-07 hV4 = 16 → 제외)
N_PERM = 200                              # SRM 색 라벨 순열 통제 반복 수

OUT_CANON = ROOT / "analysis/validation/results/within_hc_reliability.json"
OUT_COPY = (ROOT / "analysis/phase6_behavioral_analysis/exp2_neural"
                   "/results/within_hc_reliability.json")


# ---------------------------------------------------------------- RDM builders
def rdm_voxel(pattern):
    """pattern (8, V) → 28-vector correlation-distance RDM."""
    return pdist(pattern, metric="correlation")


def rdm_pca(pattern, k=K_PCA):
    """per-subject voxel PCA top-k → (8, k) scores → 28-vector RDM."""
    x = pattern - pattern.mean(axis=0, keepdims=True)
    k_eff = min(k, min(x.shape) - 1)
    if k_eff < 2:
        return None
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    return pdist(x @ vt[:k_eff].T, metric="correlation")


def agree(a, b):
    if a is None or b is None:
        return None, None
    if np.std(a) == 0 or np.std(b) == 0:
        return None, None
    return float(pearsonr(a, b)[0]), float(spearmanr(a, b)[0])


def spearman_brown(r, factor=2.0):
    """split-half(3 run) → full(6 run) 보정."""
    if r is None:
        return None
    return float(factor * r / (1 + (factor - 1) * r)) if (1 + (factor - 1) * r) != 0 else None


def summarize(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    return {"mean": round(float(np.mean(vals)), 4),
            "sd": round(float(np.std(vals, ddof=1)), 4) if len(vals) > 1 else None,
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
            "n": len(vals)}


# ------------------------------------------------------------------ data access
def load(subject, roi):
    p = DATA_DIR / subject / roi / "amplitudes_procrustes.npy"
    if not p.exists():
        return None
    a = np.load(p)                        # (6, 8, V)
    if a.shape[2] < MIN_VOX:
        return None
    return a


def main():
    out = {
        "meta": {
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "script": "analysis/validation/scripts/within_hc_reliability.py",
            "data": str(DATA_DIR.relative_to(ROOT)),
            "hc_subjects": HC,
            "rois": {r: ROI_LABEL[r] for r in ROIS},
            "k_srm": K_SRM,
            "k_pca": K_PCA,
            "min_voxels": MIN_VOX,
            "note": ("모든 상관은 8x8 RDM 의 28-원소 상삼각 벡터에 대해 계산. "
                     "pca6 = production L_RDM 공간(s10b_v6_pca_rdm.py), "
                     "srm = Methods 기술 공간, voxel = exp2 후속 분석 비교용."),
        },
        "within_subject_split_half": {},
        "between_hc_loo": {},
        "hc_mean_split_half": {},
        "excluded": {},
    }

    # ---------------------------------------------------- 1. within-subject split-half
    splits = [s for s in combinations(range(6), 3) if 0 in s]   # 10 splits
    for roi in ROIS:
        lab = ROI_LABEL[roi]
        out["within_subject_split_half"][lab] = {}
        for subj in HC + ["sub-08", "sub-09"]:
            amp = load(subj, roi)
            if amp is None:
                out["excluded"].setdefault(lab, []).append(subj)
                continue
            res = {}
            for space, fn in (("voxel", rdm_voxel), ("pca6", rdm_pca)):
                rp, rs = [], []
                for s in splits:
                    other = [i for i in range(6) if i not in s]
                    a = fn(amp[list(s)].mean(axis=0))
                    b = fn(amp[other].mean(axis=0))
                    p, q = agree(a, b)
                    rp.append(p)
                    rs.append(q)
                sp, ss = summarize(rp), summarize(rs)
                res[space] = {
                    "pearson": sp,
                    "spearman": ss,
                    "pearson_spearman_brown_6run": spearman_brown(sp["mean"]) if sp else None,
                }
            out["within_subject_split_half"][lab][subj] = res

    # ---------------------------------------------------- 2. between-HC LOO agreement
    from brainiak.funcalign.srm import SRM

    for roi in ROIS:
        lab = ROI_LABEL[roi]
        subs, mats = [], []
        for s in HC:
            a = load(s, roi)
            if a is None:
                continue
            subs.append(s)
            mats.append(a.mean(axis=0))            # (8, V)
        entry = {"subjects": subs, "n": len(subs)}

        for space, fn in (("voxel", rdm_voxel), ("pca6", rdm_pca)):
            rdms = [fn(m) for m in mats]
            rp, rs = [], []
            for i in range(len(rdms)):
                rest = np.mean([rdms[j] for j in range(len(rdms)) if j != i], axis=0)
                p, q = agree(rdms[i], rest)
                rp.append(p)
                rs.append(q)
            entry[space] = {"pearson": summarize(rp), "spearman": summarize(rs),
                            "per_subject_pearson": [None if v is None else round(v, 4) for v in rp]}

        # SRM: 대칭 LOSO — 6명으로 학습, 남긴 1명은 SVD 로 투영 (canonical 절차)
        #
        # + 색 라벨 순열 통제 (2026-07-22):
        #   SRM 투영은 held-out 피험자를 공유 공간에 '맞추도록' 회전시키므로,
        #   높은 일치도가 정렬 절차의 산물일 수 있다. held-out 피험자의 8개 색 라벨을
        #   섞은 뒤 동일 절차로 투영하여 귀무분포를 만든다. 귀무가 0 근처면 정렬이
        #   무에서 일치를 만들어내는 것이 아님을 뜻한다.
        k = K_SRM[roi]
        rng = np.random.default_rng(42)
        rp, rs, null_all, pvals = [], [], [], []
        for i in range(len(mats)):
            tr = [mats[j].T for j in range(len(mats)) if j != i]      # (V, 8)
            srm = SRM(n_iter=10, features=k)
            srm.fit(tr)
            tr_aligned = [(srm.w_[t].T @ tr[t]).T for t in range(len(tr))]   # (8, k)
            ref = np.mean([rdm_voxel(a) for a in tr_aligned], axis=0)
            s_pinv = np.linalg.pinv(srm.s_)

            def project(pattern):                                    # (8, V) → (8, k)
                u, _, vt = np.linalg.svd(pattern.T @ s_pinv, full_matrices=False)
                return pattern @ (u @ vt)

            p, q = agree(rdm_voxel(project(mats[i])), ref)
            rp.append(p)
            rs.append(q)

            null = []
            for _ in range(N_PERM):
                perm = rng.permutation(8)
                pv, _ = agree(rdm_voxel(project(mats[i][perm])), ref)
                if pv is not None:
                    null.append(pv)
            null_all.extend(null)
            if p is not None and null:
                pvals.append((1 + sum(v >= p for v in null)) / (1 + len(null)))

        entry["srm"] = {
            "k": k,
            "pearson": summarize(rp), "spearman": summarize(rs),
            "per_subject_pearson": [None if v is None else round(v, 4) for v in rp],
            "label_permutation_null": {
                "n_perm_per_subject": N_PERM,
                "pearson": summarize(null_all),
                "pct95": round(float(np.percentile(null_all, 95)), 4) if null_all else None,
                "per_subject_p": [round(v, 4) for v in pvals],
                "p_combined_median": round(float(np.median(pvals)), 4) if pvals else None,
            },
        }
        out["between_hc_loo"][lab] = entry

    # ---------------------------------------------------- 3. HC-mean reference stability
    for roi in ROIS:
        lab = ROI_LABEL[roi]
        mats = [load(s, roi) for s in HC]
        mats = [m.mean(axis=0) for m in mats if m is not None]
        n = len(mats)
        entry = {"n": n}
        for space, fn in (("voxel", rdm_voxel), ("pca6", rdm_pca)):
            rdms = [fn(m) for m in mats]
            rp, rs = [], []
            for half in combinations(range(n), n // 2):
                other = [i for i in range(n) if i not in half]
                a = np.mean([rdms[i] for i in half], axis=0)
                b = np.mean([rdms[i] for i in other], axis=0)
                p, q = agree(a, b)
                rp.append(p)
                rs.append(q)
            entry[space] = {"pearson": summarize(rp), "spearman": summarize(rs)}
        out["hc_mean_split_half"][lab] = entry

    for path in (OUT_CANON, OUT_COPY):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(out)
        payload["meta"] = dict(out["meta"], canonical_path=str(OUT_CANON.relative_to(ROOT)))
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        print(f"SAVED {path}")

    # ---------------------------------------------------------------- console 요약
    print("\n=== 1. within-subject split-half (단일 피험자 RDM 잡음 상한, Pearson) ===")
    print(f"{'ROI':5s} {'space':6s} " + " ".join(f"{s[-2:]:>6s}" for s in HC + ["sub-08", "sub-09"]))
    for lab, d in out["within_subject_split_half"].items():
        for space in ("voxel", "pca6"):
            row = []
            for s in HC + ["sub-08", "sub-09"]:
                v = d.get(s, {}).get(space, {}).get("pearson")
                row.append(f"{v['mean']:+6.3f}" if v else "     ·")
            print(f"{lab:5s} {space:6s} " + " ".join(row))

    print("\n=== 2. between-HC LOO agreement (모양 일치도, Pearson mean) ===")
    print(f"{'ROI':5s} {'voxel':>8s} {'pca6':>8s} {'srm':>8s}")
    for lab, d in out["between_hc_loo"].items():
        vals = []
        for space in ("voxel", "pca6", "srm"):
            v = d[space]["pearson"]
            vals.append(f"{v['mean']:+8.3f}" if v else "       ·")
        print(f"{lab:5s} " + " ".join(vals))

    print("\n=== 2b. SRM 색 라벨 순열 통제 (정렬이 일치도를 만들어내는가) ===")
    print(f"{'ROI':5s} {'true':>8s} {'null mean':>10s} {'null p95':>9s} {'median p':>9s}")
    for lab, d in out["between_hc_loo"].items():
        s_ = d["srm"]; n_ = s_["label_permutation_null"]
        print(f"{lab:5s} {s_['pearson']['mean']:+8.3f} {n_['pearson']['mean']:+10.3f} "
              f"{n_['pct95']:+9.3f} {n_['p_combined_median']:9.3f}")

    print("\n=== 3. HC-mean reference split-half stability (Pearson mean) ===")
    print(f"{'ROI':5s} {'voxel':>8s} {'pca6':>8s}")
    for lab, d in out["hc_mean_split_half"].items():
        vals = []
        for space in ("voxel", "pca6"):
            v = d[space]["pearson"]
            vals.append(f"{v['mean']:+8.3f}" if v else "       ·")
        print(f"{lab:5s} " + " ".join(vals))

    if out["excluded"]:
        print("\n제외:", out["excluded"])


if __name__ == "__main__":
    main()
