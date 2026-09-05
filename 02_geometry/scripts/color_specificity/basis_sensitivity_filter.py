#!/usr/bin/env python3
"""
basis_sensitivity_filter.py — RDM 기저(PCA vs SRM)를 바꿨을 때
적합 파라미터와 **실제 출력 색**이 얼마나 달라지는가 (reviewer response 2026-07-22)

배경
----
production L_RDM 은 per-subject voxel PCA(K=6) 공간을 쓴다(s10b_v6_pca_rdm.py).
그러나 within_hc_reliability.py 결과, PCA 공간에서는 HC 평균 RDM(= ΔRDM 의 기준항)이
재현되지 않는다(HC-mean split-half: V1 +0.15, V2 -0.04). SRM 공간에서는
HC 간 일치도가 V1 +0.58 / V2 +0.59 로 정상이다.

따라서 동일 combo 를 SRM-RDM 기저(s10b_v6_srm_rdm_results_*.json)로 재적합했을 때
argmin 이 어떻게 이동하는지, 그리고 그 이동이 **필터 출력 색**에서 몇 도/ΔE 인지 계산한다.

출력
----
analysis/validation/results/basis_sensitivity_filter.json
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/validation/scripts/basis_sensitivity_filter.py  (development repository, commit 53c81c2)
# Role      : Sensitivity of the filter to the RDM basis.
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
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from skimage.color import deltaE_ciede2000

ROOT = Path(__file__).resolve().parents[3]
P2 = ROOT / "analysis/phase5_filter_optimization"
sys.path.insert(0, str(P2 / "scripts"))

from forward_models.two_component import pre_image_2comp   # noqa: E402

RES = P2 / "results/s10_inclusion"
OUT = ROOT / "analysis/validation/results/basis_sensitivity_filter.json"

HUES = np.arange(0, 360, 45.0)          # 8 displayed hues
COLOR_NAMES = ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]
L_STAR, CHROMA = 75.0, 40.0

# production combo (논문 보고값을 재현하는 조합) — s10b 결과 대조로 확인함
SUBJECTS = {
    "sub-08": {"cvd_type": "deutan", "combo": "γOY|RDMV2|noLOCO", "published": (6.0, -42.0)},
    "sub-09": {"cvd_type": "protan", "combo": "γGB|RDMV1|noLOCO", "published": (2.0, 24.0)},
}


def argmin_stats(path, combo):
    rows = [r for r in json.load(open(path))["storage"][combo]["2comp"]
            if r.get("beta_s") is not None]
    bs = [r["beta_s"] for r in rows]
    bc = [r["beta_c"] for r in rows]
    mode, cnt = Counter([(r["beta_s"], r["beta_c"]) for r in rows]).most_common(1)[0]
    return {
        "median": [float(np.median(bs)), float(np.median(bc))],
        "iqr": [float(np.subtract(*np.percentile(bs, [75, 25]))),
                float(np.subtract(*np.percentile(bc, [75, 25])))],
        "mode": [float(mode[0]), float(mode[1])],
        "mode_share": round(cnt / len(rows), 3),
        "boundary_rate": round(float(np.mean([r["boundary"] for r in rows])), 3),
        "n": len(rows),
    }


def lab(hue_deg):
    h = np.deg2rad(np.asarray(hue_deg, float))
    return np.stack([np.full_like(h, L_STAR), CHROMA * np.cos(h), CHROMA * np.sin(h)], axis=-1)


def filter_output(cvd_type, bs, bc):
    """각 표시 색에 대한 필터 출력(= pre-image) 색상각. 잔차는 별도 반환."""
    sol = [pre_image_2comp(t, cvd_type, bs, bc) for t in HUES]
    return np.array([s[0] for s in sol]), np.array([s[1] for s in sol])


def circ_diff(a, b):
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0


def main():
    out = {"meta": {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "script": "analysis/validation/scripts/basis_sensitivity_filter.py",
        "question": ("production L_RDM 의 PCA 기저를 SRM 기저로 교체하면 "
                     "적합 파라미터와 필터 출력 색이 얼마나 달라지는가"),
        "stimulus": {"L_star": L_STAR, "chroma": CHROMA, "hues_deg": HUES.tolist(),
                     "color_names": COLOR_NAMES},
        "reliability_context": ("HC-mean RDM split-half stability — "
                                "PCA: V1 +0.146 V2 -0.041 | SRM LOO agreement: V1 +0.578 V2 +0.590 "
                                "(analysis/validation/results/within_hc_reliability.json)"),
    }, "subjects": {}}

    for subj, cfg in SUBJECTS.items():
        combo, ctype = cfg["combo"], cfg["cvd_type"]
        pca = argmin_stats(RES / f"s10b_v6_pca_rdm_results_{subj}.json", combo)
        srm = argmin_stats(RES / f"s10b_v6_srm_rdm_results_{subj}.json", combo)

        bs_p, bc_p = pca["median"]
        bs_s, bc_s = srm["median"]

        out_p, res_p = filter_output(ctype, bs_p, bc_p)
        out_s, res_s = filter_output(ctype, bs_s, bc_s)
        d_hue = circ_diff(out_s, out_p)

        de = deltaE_ciede2000(lab(out_p), lab(out_s))

        # 참고: 필터가 무보정(항등)과 얼마나 다른가 — 두 기저 각각
        de_p_vs_id = deltaE_ciede2000(lab(HUES), lab(out_p))
        de_s_vs_id = deltaE_ciede2000(lab(HUES), lab(out_s))

        out["subjects"][subj] = {
            "cvd_type": ctype,
            "combo": combo,
            "published_pca_argmin": list(cfg["published"]),
            "pca_basis": pca,
            "srm_basis": srm,
            "param_shift_deg": {
                "beta_s": round(bs_s - bs_p, 2),
                "beta_c": round(bc_s - bc_p, 2),
                "euclidean": round(float(np.hypot(bs_s - bs_p, bc_s - bc_p)), 2),
            },
            "filter_output_hue_deg": {
                "pca": [round(float(v), 2) for v in out_p],
                "srm": [round(float(v), 2) for v in out_s],
                "difference": [round(float(v), 2) for v in d_hue],
                "mean_abs_difference": round(float(np.mean(np.abs(d_hue))), 2),
                "max_abs_difference": round(float(np.max(np.abs(d_hue))), 2),
            },
            "delta_e_ciede2000_between_bases": {
                "per_color": {c: round(float(v), 2) for c, v in zip(COLOR_NAMES, de)},
                "mean": round(float(np.mean(de)), 2),
                "max": round(float(np.max(de)), 2),
            },
            "delta_e_vs_no_filter": {
                "pca_mean": round(float(np.mean(de_p_vs_id)), 2),
                "srm_mean": round(float(np.mean(de_s_vs_id)), 2),
            },
            "pre_image_max_abs_residual_deg": {
                "pca": round(float(np.max(np.abs(res_p))), 4),
                "srm": round(float(np.max(np.abs(res_s))), 4),
            },
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"SAVED {OUT}\n")

    for subj, r in out["subjects"].items():
        print(f"=== {subj} ({r['cvd_type']}) — {r['combo']} ===")
        print(f"  PCA argmin  median={r['pca_basis']['median']} "
              f"mode={r['pca_basis']['mode']} share={r['pca_basis']['mode_share']}")
        print(f"  SRM argmin  median={r['srm_basis']['median']} "
              f"mode={r['srm_basis']['mode']} share={r['srm_basis']['mode_share']}")
        print(f"  파라미터 이동: Δβs={r['param_shift_deg']['beta_s']:+.0f}° "
              f"Δβc={r['param_shift_deg']['beta_c']:+.0f}°")
        f = r["filter_output_hue_deg"]
        print(f"  필터 출력 색상각 차이: mean |Δ| = {f['mean_abs_difference']}°, "
              f"max = {f['max_abs_difference']}°")
        d = r["delta_e_ciede2000_between_bases"]
        print(f"  ΔE00 (두 기저 간): mean = {d['mean']}, max = {d['max']}")
        print("   " + "  ".join(f"{c[:3]}:{v:5.1f}" for c, v in d["per_color"].items()))
        print(f"  참고 ΔE00 vs 무보정: PCA {r['delta_e_vs_no_filter']['pca_mean']} / "
              f"SRM {r['delta_e_vs_no_filter']['srm_mean']}\n")


if __name__ == "__main__":
    main()
