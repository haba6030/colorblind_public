"""_shift_gain_ch.py — 순환이동 이득의 선택편향 정합 검정 (2026-08-24).

RESULTS_GEOMETRY_VALIDITY_2026-08-05.md §4 의 유보 1("8개 이동 중 최솟값을 취하므로
이득이 하향 편향된다. 45°가 특별함을 보이려면 귀무분포가 필요하다")에 대한 답이다.

핵심: **통제군도 똑같이 8개 중 최솟값을 취한다.** 따라서 선택편향이 두 집단에 동일하게
걸리고, "최적이동 이득" 이라는 같은 통계량을 통제군 분포에 대고 Crawford-Howell 단측
상단으로 검정하면 편향이 상쇄된다. 무작위 대응 귀무분포를 따로 만들 필요가 없다.

입력은 동결 산출물 `analysis/validation/results/cyclic_shift_disparity{,_motreg}.json`
(ROI x subject -> 8칸 이동별 disparity, 0번이 항등).
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 02_geometry
# Manuscript: Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison)
# Original  : analysis/future_phase1_sensitivity/scripts/_shift_gain_ch.py  (development repository, commit 53c81c2)
# Role      : Cyclic hue-shift gain with Crawford-Howell test against the control gain distribution (S18).
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
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "analysis/validation/results"
OUT = Path(__file__).resolve().parent.parent / "results"
HC = [f"sub-0{i}" for i in range(1, 8)]
CVD = {"sub-08": "deutan", "sub-09": "protan"}
ARMS = {"with_residuals": "cyclic_shift_disparity.json",
        "motreg": "cyclic_shift_disparity_motreg.json"}


def gain(vals):
    v = np.asarray(vals, float)
    return float((v[0] - v.min()) / v[0]), int(v.argmin())


def main():
    out = {"note": "best-shift gain, Crawford-Howell one-tailed upper vs controls; "
                   "selection bias is matched because controls take the same min over 8 shifts",
           "arms": {}}
    for arm, fname in ARMS.items():
        d = json.load(open(SRC / fname))
        arm_out = {}
        print(f"===== {arm}")
        for roi, R in d.items():
            g = {s: gain(v) for s, v in R.items()}
            hc = [g[s][0] for s in HC if s in g]
            m, sd, n = float(np.mean(hc)), float(np.std(hc, ddof=1)), len(hc)
            cell = {"hc_n": n, "hc_gain_mean": m, "hc_gain_sd": sd,
                    "hc_best_shift_idx": [g[s][1] for s in HC if s in g], "cvd": {}}
            for s, lab in CVD.items():
                if s not in g:
                    continue
                gg, idx = g[s]
                t = (gg - m) / (sd * np.sqrt((n + 1) / n))
                cell["cvd"][lab] = {"gain": gg, "best_shift_idx": idx,
                                    "best_shift_deg": idx * 45,
                                    "t": float(t), "p_one_tailed_upper": float(1 - stats.t.cdf(t, n - 1)),
                                    "d_cc": float((gg - m) / sd)}
            arm_out[roi] = cell
            c = cell["cvd"]
            print(f"  {roi:4s} HC {m*100:5.1f}% +-{sd*100:4.1f} (n={n})  "
                  f"deutan {c['deutan']['best_shift_deg']:3d}deg {c['deutan']['gain']*100:5.1f}% p={c['deutan']['p_one_tailed_upper']:.4f}  "
                  f"protan {c['protan']['best_shift_deg']:3d}deg {c['protan']['gain']*100:5.1f}% p={c['protan']['p_one_tailed_upper']:.4f}")
        out["arms"][arm] = arm_out

    out["caveat_hV4_motreg"] = ("hV4 motreg 의 통제군 이득 SD 가 0.4% 로 붕괴해 t 가 31 까지 "
                                "부풀었다. 그 칸은 인용하지 않는다.")
    json.dump(out, open(OUT / "shift_gain_ch.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nwrote {OUT / 'shift_gain_ch.json'}")


if __name__ == "__main__":
    main()
