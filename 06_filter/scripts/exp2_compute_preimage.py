"""exp2_compute_preimage.py — closure A13 pre-image for exp2 deployment.

Solves θ_display s.t. θ_display + forward_2comp(θ_display) ≡ θ_canonical (mod 360),
per-hue, for the §0 override-confirmed per-subject filters:
  sub-08 deutan: (β_s=6, β_c=-42)   — S08-srm-robust (γOY|RDMV2|noLOCO), s17 LOO median
  sub-09 protan: (β_s=2, β_c=+24)   — γGB|RDMV1|noLOCO, s10b_v6_pca_rdm

Output (flat): results/exp2_preimage/sub-{ID}_2component_preimage.json
  fields per project Output Convention (CLAUDE.md §7) + filters_exp2.py interface
  (δθ_apply[i] = rotation applied to RENDERED color_i by index, direction=preimage).

Verifies 8/8 exact (residual < 1e-3°). 8/8 exact는 closure invariant; 미달 시 reject.
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 06_filter
# Manuscript: Results section 6; Figure 6; Methods 'Stimulus-space filter and its evaluation'
# Original  : analysis/phase5_filter_optimization/scripts/exp2_compute_preimage.py  (development repository, commit 53c81c2)
# Role      : Numerical pre-image of the fitted two-component transform -> sub-*_2component_preimage.json
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
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent))
from two_comp import forward_2comp, HUE_CANON, THETA_CONF


def residual(theta_d, theta_target, beta_s, beta_c, family):
    pred = theta_d + forward_2comp(beta_s, beta_c, family, hues=np.array([theta_d]))[0]
    delta = (pred - theta_target + 180.0) % 360.0 - 180.0
    return delta


def solve_preimage(theta_target, beta_s, beta_c, family, search=60.0):
    """Find θ_d such that θ_d + forward(θ_d) ≡ θ_target (mod 360)."""
    lo, hi = theta_target - search, theta_target + search
    f_lo = residual(lo, theta_target, beta_s, beta_c, family)
    f_hi = residual(hi, theta_target, beta_s, beta_c, family)
    if f_lo * f_hi > 0:
        grid = np.linspace(lo, hi, 720)
        for i in range(len(grid) - 1):
            a, b = grid[i], grid[i + 1]
            fa = residual(a, theta_target, beta_s, beta_c, family)
            fb = residual(b, theta_target, beta_s, beta_c, family)
            if fa * fb <= 0:
                return brentq(residual, a, b, args=(theta_target, beta_s, beta_c, family), xtol=1e-9)
        raise RuntimeError(f"No sign change for target={theta_target}, ({beta_s},{beta_c},{family})")
    return brentq(residual, lo, hi, args=(theta_target, beta_s, beta_c, family), xtol=1e-9)


COLOR_NAMES = ['color_1', 'color_2', 'color_3', 'color_4',
               'color_5', 'color_6', 'color_7', 'color_8']
HUE_NAMES = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta']


def build_preimage_record(subject, family, beta_s, beta_c, source):
    preimage = np.array([solve_preimage(t, beta_s, beta_c, family) for t in HUE_CANON])
    perceived = preimage + forward_2comp(beta_s, beta_c, family, hues=preimage)
    residuals = ((perceived - HUE_CANON + 180.0) % 360.0 - 180.0)
    delta_apply = ((preimage - HUE_CANON + 180.0) % 360.0 - 180.0)
    max_res = float(np.max(np.abs(residuals)))
    if max_res > 1e-3:
        raise RuntimeError(
            f"Pre-image NOT exact for {subject}: max residual {max_res:.6f}° > 1e-3°. "
            f"Closure invariant violated — reject."
        )
    return {
        "subject": subject,
        "cvd_type": family,
        "model": "2component_closure_A13",
        "source_note": source,
        "phase_a_params": {"beta_s": float(beta_s), "beta_c": float(beta_c)},
        "theta_conf_deg": THETA_CONF[family],
        "stimulus_angles_cielab": HUE_CANON.tolist(),
        "color_names": COLOR_NAMES,
        "hue_names": HUE_NAMES,
        "preimage_angles_cielab": preimage.tolist(),
        "delta_theta_apply_deg": delta_apply.tolist(),
        "perceived_angles_check": perceived.tolist(),
        "residuals_deg": residuals.tolist(),
        "max_residual_deg": max_res,
        "exact_status": "8/8 exact (residual < 1e-3°)",
        "interface_note": (
            "delta_theta_apply_deg[i] = rotation in CIElab a*-b* plane to apply to "
            "RENDERED color_{i+1} by INDEX. direction=preimage (correction). "
            "Subject perceives canonical hue after CVD distortion."
        ),
        "forward_provenance": (
            "scripts/two_comp.py:forward_2comp (closure A13 raw nominal-θ). "
            "DO NOT use forward_models/two_component.py frozen variant."
        ),
    }


def main():
    out_dir = Path(__file__).parent.parent / "results" / "exp2_preimage"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("sub-08", "deutan", 6.0, -42.0,
         "S08-srm-robust (γOY|RDMV2|noLOCO), 2-comp, s17 LOO median, IQR(bs=5,bc=4). "
         "§0 override confirmed 2026-05-30."),
        ("sub-09", "protan", 2.0, 24.0,
         "γGB|RDMV1|noLOCO, 2-comp, s10b_v6_pca_rdm. §0 override confirmed 2026-05-30."),
    ]

    records = []
    for sub, fam, bs, bc, src in cases:
        rec = build_preimage_record(sub, fam, bs, bc, src)
        out_fp = out_dir / f"{sub}_2component_preimage.json"
        out_fp.write_text(json.dumps(rec, indent=2))
        records.append(rec)
        print(f"\n=== {sub} ({fam}, β_s={bs}, β_c={bc}) ===")
        print(f"  max residual: {rec['max_residual_deg']:.2e}° ({rec['exact_status']})")
        print(f"  preimage_angles_cielab + delta_theta_apply_deg:")
        for i, (name, h, pre, da) in enumerate(
            zip(HUE_NAMES, HUE_CANON, rec['preimage_angles_cielab'], rec['delta_theta_apply_deg'])
        ):
            print(f"    color_{i+1:d} ({name:8s} @{h:5.1f}°)  preimage={pre:8.3f}°  δθ_apply={da:+7.3f}°")
        print(f"  saved: {out_fp.relative_to(Path.cwd()) if str(out_fp).startswith(str(Path.cwd())) else out_fp}")

    config = {
        "generated": "2026-05-30",
        "purpose": "exp2 Optimal-condition pre-image filters",
        "override_note": "§0 override confirmed by user for tentative adoption; S7 results pending re-validation",
        "subjects": [r["subject"] for r in records],
        "model": "2component_closure_A13",
        "forward_source": "scripts/two_comp.py:forward_2comp",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"\n  wrote: {out_dir / 'config.json'}")


if __name__ == "__main__":
    main()
