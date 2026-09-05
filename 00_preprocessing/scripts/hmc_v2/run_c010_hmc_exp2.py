#!/usr/bin/env python3
"""
run_c010_hmc_exp2.py — HMC 재산출용 exp2 C010 실행기 (Candidate A, 2026-08-05)

`run_c010_hmc.py`의 exp2 대응본. 동일 원칙: 원본 스크립트
`analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp2_C010_conditions.py`
를 고치지 않고 import 한 뒤 경로 상수만 재바인딩한다.

원본의 `OUTPUT_DIR`은 `main()` 안에서 mask variant 에 따라 설정되므로 `main()`을 우회하고
`run_subject_roi`를 직접 호출한다. `MASK_VARIANT`도 함께 설정해야 한다.

  FMRIPREP_DIR : fmriprep_out_method3_2nd  →  fmriprep_out_method3_2nd_hmc
  OUTPUT_DIR   : full_dataset_C010_exp2[_matched]  →  ..._hmc

ROI 마스크(`ROI_MASKS_DIR`)는 exp1과 마찬가지로 바꾸지 않는다 — 두 arm 이 동일 voxel 집합을
공유해야 차이가 HMC 단독으로 귀인된다.

실행
    python run_c010_hmc_exp2.py --subject 08 --roi V1 --mask-variant native
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/hmc_reanalysis/run_c010_hmc_exp2.py  (development repository, commit 53c81c2)
# Role      : Session-2 amplitude estimation on the head-motion-corrected arm.
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
import importlib.util
import sys
from pathlib import Path

SRC = Path("/scratch/connectome/haba6030/colorBlind/analysis"
           "/phase6_behavioral_analysis/exp2_neural/scripts/exp2_C010_conditions.py")
DEFAULT_FMRIPREP = Path("/storage/connectome/haba6030/fmriprep_out_method3_2nd_hmc")
DERIV = Path("/scratch/connectome/haba6030/colorBlind/derivatives")


def load_src(path):
    if not path.exists():
        raise FileNotFoundError(f"exp2 pipeline not found: {path}")
    spec = importlib.util.spec_from_file_location("exp2_c010_src", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp2_c010_src"] = mod
    spec.loader.exec_module(mod)          # main() 은 __main__ 가드 뒤라 실행되지 않는다
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--roi", required=True, choices=["V1", "V2", "V3", "V4"])
    ap.add_argument("--mask-variant", default="native", choices=["native", "matched"])
    ap.add_argument("--fmriprep-dir", default=str(DEFAULT_FMRIPREP))
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()

    mod = load_src(Path(args.src))

    out = Path(args.output_dir) if args.output_dir else DERIV / (
        "full_dataset_C010_exp2_hmc" if args.mask_variant == "native"
        else "full_dataset_C010_exp2_matched_hmc")

    before = mod.FMRIPREP_DIR
    mod.FMRIPREP_DIR = Path(args.fmriprep_dir)
    mod.MASK_VARIANT = args.mask_variant
    mod.OUTPUT_DIR = out

    print("=" * 80)
    print("exp2 C010 re-run — Candidate A (method3 + HMC)")
    print(f"  source        : {args.src}")
    print(f"  FMRIPREP_DIR  : {before}\n               -> {mod.FMRIPREP_DIR}")
    print(f"  OUTPUT_DIR    : {mod.OUTPUT_DIR}")
    print(f"  mask variant  : {mod.MASK_VARIANT}")
    print(f"  ROI_MASKS_DIR : {mod.ROI_MASKS_DIR}   (unchanged)")
    print(f"  target        : sub-{args.subject} {args.roi}")
    print("=" * 80)

    if not mod.FMRIPREP_DIR.exists():
        raise FileNotFoundError(f"BOLD tree missing: {mod.FMRIPREP_DIR}")

    mod.run_subject_roi(args.subject, args.roi)


if __name__ == "__main__":
    main()
