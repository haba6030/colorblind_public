"""_staircase_pairs_table.py — 쌍별 두 스테어케이스 표 (2026-08-25).

원고 조치: sub-09 개인화 orange-yellow 트랙 불일치를 본문 각주로 공개하던 안을 철회하고,
**부록 표 + 표 아래 설명**으로 바꾼다 (`MANUSCRIPT_EDITS_CONSOLIDATED` §4.6b).

표는 두 CVD 참가자 x 세 조건 x 8쌍의 두 트랙 값을 그대로 싣는다. 독자가 특이 셀을
직접 보고 판단할 수 있게 하는 것이 목적이며, 값을 조정하지 않는다.

입력 `results/exp2_behavior/a2_staircase_diagnosis.json`.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 03_psychophysics
# Manuscript: Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs)
# Original  : analysis/phase6_behavioral_analysis/scripts/_staircase_pairs_table.py  (development repository, commit 53c81c2)
# Role      : Renders tab:staircase_pairs (LaTeX) from the audit.
# Inputs    : paths inside this file follow the development-repository layout. The
#             neuroimaging amplitude arrays and trial-level behavioural files are not
#             distributed (REPRODUCE.md); the outputs this script produced are in
#             ../results/ of this stage and are what the stage notebook verifies.
# ============================================================================
import sys as _sys, pathlib as _pl  # public-release import shim (shared modules)
_PUBLIC_ROOT = _pl.Path(__file__).resolve().parents[2]
for _p in ("common", "01_decoding/scripts", "04_distortion_model/scripts"):
    _sys.path.insert(0, str(_PUBLIC_ROOT / _p))
del _sys, _pl, _p

import json
from pathlib import Path
import statistics as st

ROOT = Path(__file__).resolve().parents[1]
D = json.load(open(ROOT / "results/exp2_behavior/a2_staircase_diagnosis.json"))
C = D["conditions"]
PAIRS = ["red-orange", "orange-yellow", "yellow-green", "green-blue",
         "blue-purple", "yellow-purple", "red-cyan", "cyan-magenta"]
COLS = [("sub-08", "session1"), ("sub-08", "deployed"), ("sub-08", "individualized"),
        ("sub-09", "session1"), ("sub-09", "deployed"), ("sub-09", "individualized")]
FLAG = 0.10


def cell(sub, cond, pair):
    p = C[f"{sub}/{cond}"]["pairs"][pair]
    s = f"{p['sc_hi']:.3f}/{p['sc_lo']:.3f}"
    return (r"\textbf{" + s + "}") if p["spread"] > FLAG else s


def main():
    print(r"% --- \S S15 pair-level staircase table (auto-generated) ---")
    print(r"\begin{table}[h]\centering\small")
    print(r"\caption{Both staircase estimates for every color pair, given as "
          r"high-start/low-start in units of the rendered separation. Bold marks the "
          r"cells whose two estimates differ by more than " f"{FLAG:.2f}" r". Thresholds "
          r"reported elsewhere are the mean of the two.}")
    print(r"\label{tab:staircase_pairs}")
    print(r"\begin{tabular}{l" + "c" * 6 + "}")
    print(r"\toprule")
    print(r" & \multicolumn{3}{c}{Deutan} & \multicolumn{3}{c}{Protan} \\")
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}")
    print(r"Pair & Session 1 & Deployed & Individualized & Session 1 & Deployed & Individualized \\")
    print(r"\midrule")
    for pair in PAIRS:
        print(f"{pair} & " + " & ".join(cell(s, c, pair) for s, c in COLS) + r" \\")
    print(r"\bottomrule\end{tabular}\end{table}")

    allsp = [C[k]["pairs"][p]["spread"] for k in C for p in C[k]["pairs"]]
    flagged = sorted(((C[k]["pairs"][p]["spread"], k, p) for k in C for p in C[k]["pairs"]
                      if C[k]["pairs"][p]["spread"] > FLAG), reverse=True)
    hi_thr = sum(1 for k in C if max(C[k]["pairs"], key=lambda p: C[k]["pairs"][p]["threshold"])
                 == "orange-yellow")
    print(f"\n% n = {len(allsp)} pairs across all participants and conditions")
    print(f"% median spread {st.median(allsp):.4f}, max {max(allsp):.4f}, "
          f"second largest {sorted(allsp)[-2]:.4f}")
    print(f"% cells above {FLAG}: " + "; ".join(f"{k} {p} {s:.3f}" for s, k, p in flagged))
    print(f"% orange-yellow is the largest-threshold pair in {hi_thr}/{len(C)} blocks")


if __name__ == "__main__":
    main()
