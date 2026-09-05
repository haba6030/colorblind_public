"""정본 rerun_loo_consistent.py 를 수정하지 않고 DATA_DIR 만 주입해 실행한다.

CLAUDE.md: rerun_loo_consistent.py 는 정본이며 원복 금지. 따라서 import 후
모듈 속성만 덮어쓴다. CVD_SUBJECTS 의 sub-10 도 여기서 제외한다
(전 분석 제외 대상이고 hmc_v2 산출물에도 존재하지 않는다).

사용: python loso_arm_wrapper.py <arm>
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 08_sensitivity
# Manuscript: Supplementary S2 (tab:motion_arms, tab:interp_arms, session-2 endpoints), S13 sign stability, S18 (tab:color_specificity, head-motion column)
# Original  : analysis/future_phase1_sensitivity/scripts/_loso_arm_wrapper.py  (development repository, commit 53c81c2)
# Role      : Drives rerun_loo_consistent.py on an arm (DATA_DIR injection, sub-10 dropped) -> loso_two_arm_summary.json
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

import sys, json, importlib.util
from pathlib import Path

REPO = Path("/Users/jinilkim/LocalProj/colorBlind_analysis")
SRC = REPO / "analysis/phase2_SRM_across_between/rerun_loo_consistent.py"
BASE = REPO / "analysis/phase1_procrustes_decoding/results/visualization"
OUT = Path(__file__).resolve().parent / "loso_arms"

arm = sys.argv[1]
data_dir = BASE / f"full_dataset_C010_{arm}"
assert data_dir.is_dir(), data_dir

spec = importlib.util.spec_from_file_location("loo_canon", SRC)
mod = importlib.util.module_from_spec(spec)
sys.modules["loo_canon"] = mod
spec.loader.exec_module(mod)

mod.DATA_DIR = data_dir
mod.CVD_SUBJECTS = ["sub-08", "sub-09"]
mod.OUTPUT_DIR = OUT / arm

print(f"[wrapper] arm={arm}")
print(f"[wrapper] DATA_DIR={mod.DATA_DIR}")
print(f"[wrapper] CVD_SUBJECTS={mod.CVD_SUBJECTS}")
print(f"[wrapper] K={mod.K_VALUES}  perm group={mod.N_PERM_GROUP} color={mod.N_PERM_COLOR}")
mod.main()
