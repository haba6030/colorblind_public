#!/usr/bin/env python3
"""exp1 unified QC figures for all 10 subjects — SAME format as exp2 fixed QC,
so exp1 (no-filter) and exp2 (Window/Optimal) can be compared on one yardstick.

Per subject:
  roi_overlay_fixed.png : V1/V2/V3/hV4 on mean-BOLD at each ROI's COM, with
                          voxel count + ROI∩BOLD coverage %.
  tsnr_fixed.png        : tSNR map (run-1) + within-ROI tSNR histograms+medians.

exp1 differs from exp2 only in: fmriprep dir (method3_header_mi) and N_RUNS=6.
Mask = analysis-actual native variant: atlas∩GM(masknone,subjFalse) ∩ 6-run coverage.

Usage:  python exp1_qc_figures.py 01 02 ... 10     (default = all 10)
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/exp1_qc_figures.py  (development repository, commit 53c81c2)
# Role      : Session-1 preprocessing QC figures.
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

import sys
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from scipy import ndimage
from nilearn import plotting

FMRIPREP = Path("/storage/connectome/haba6030/fmriprep_out_method3_header_mi")
ROIDIR   = Path("/scratch/connectome/haba6030/colorBlind/analysis/roi_masks/method3_header_mi")
OUTBASE  = Path("/scratch/connectome/haba6030/colorBlind/analysis/phase6_behavioral_analysis/exp2_neural/preproc_qc_exp1")
N_RUNS = 6
ROIS = ["V1", "V2", "V3", "hV4"]


def bold_path(sub, r):
    return FMRIPREP / f"sub-{sub}" / "func" / \
        f"sub-{sub}_task-rsvp_run-{r}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"


def bmask_path(sub, r):
    return FMRIPREP / f"sub-{sub}" / "func" / \
        f"sub-{sub}_task-rsvp_run-{r}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"


def atlas_path(sub, roi):
    f = f"{roi}_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjFalse.nii.gz"
    return ROIDIR / f"sub-{sub}" / "roi_pipeline" / f


def analysis_mask(sub, roi):
    aimg = nib.load(str(atlas_path(sub, roi)))
    atlas = aimg.get_fdata() > 0
    cov = None
    for r in range(1, N_RUNS + 1):
        bm = nib.load(str(bmask_path(sub, r))).get_fdata() > 0
        cov = bm if cov is None else (cov & bm)
    return aimg.affine, (atlas & cov), atlas.sum()


def mean_bold_img(sub, r=1):
    img = nib.load(str(bold_path(sub, r)))
    return nib.Nifti1Image(img.get_fdata().mean(-1), img.affine)


def tsnr_arr(sub, r=1):
    img = nib.load(str(bold_path(sub, r)))
    d = img.get_fdata()
    mu, sd = d.mean(-1), d.std(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(sd > 0, mu / sd, 0.0)
    return img.affine, t


def com_world(maskbool, affine):
    return nib.affines.apply_affine(affine, np.array(ndimage.center_of_mass(maskbool)))


def fig_overlay(sub, outdir):
    mb = mean_bold_img(sub)
    mbd = mb.get_fdata()
    tmp_pngs, titles = [], []
    for roi in ROIS:
        affine, m, n_atlas = analysis_mask(sub, roi)
        nvox = int(m.sum())
        pct = 100.0 * int((m & (mbd > 0)).sum()) / max(nvox, 1)
        mimg = nib.Nifti1Image(m.astype(np.int16), affine)
        cc = com_world(m, affine) if nvox > 0 else np.array([0.0, 0.0, 0.0])
        if not np.all(np.isfinite(cc)):              # empty/degenerate ROI -> brain centre
            cc = np.array([0.0, 0.0, 0.0])
        tmp = outdir / f"_tmp_{roi}.png"
        disp = plotting.plot_roi(mimg, bg_img=mb, display_mode="ortho",
                                 cut_coords=cc, dim=-0.2, cmap="autumn", alpha=0.75,
                                 black_bg=False, annotate=True, colorbar=False,
                                 figure=plt.figure(figsize=(10, 2.6)))
        disp.savefig(str(tmp), dpi=150)
        disp.close()
        tmp_pngs.append(tmp)
        titles.append(f"{roi}:  {nvox} vox (atlas∩GM={int(n_atlas)})  |  "
                      f"ROI∩BOLD = {pct:.1f}%   — cut at ROI centre of mass")
    nr = len(ROIS)
    fig, axes = plt.subplots(nr, 1, figsize=(10, 2.75 * nr))
    for ax, tmp, ttl in zip(axes, tmp_pngs, titles):
        ax.imshow(mpimg.imread(str(tmp)))
        ax.axis("off")
        ax.set_title(ttl, fontsize=10.5, pad=4)
    fig.suptitle(f"sub-{sub} exp1 (no-filter) — ROI on mean-BOLD  (atlas∩GM∩exp1-coverage)",
                 fontsize=12, y=0.997)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.01, hspace=0.18)
    out = outdir / "roi_overlay_fixed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    for tmp in tmp_pngs:
        try:
            tmp.unlink()
        except OSError:
            pass
    print(f"  saved {out}")


def fig_tsnr(sub, outdir):
    affine, t = tsnr_arr(sub)
    mb = mean_bold_img(sub)
    timg = nib.Nifti1Image(t, affine)
    fig = plt.figure(figsize=(13, 8.5))
    ax_map = fig.add_subplot(2, 1, 1)
    plotting.plot_stat_map(timg, bg_img=mb, axes=ax_map, display_mode="z",
                           cut_coords=7, vmax=60, cmap="viridis", colorbar=True,
                           black_bg=False, annotate=True,
                           title=f"sub-{sub} exp1 tSNR (run-1, mean/SD over time)")
    ax_h = fig.add_subplot(2, 1, 2)
    summ = []
    for roi in ROIS:
        _, m, _ = analysis_mask(sub, roi)
        v = t[m]; v = v[np.isfinite(v) & (v > 0)]
        med = float(np.median(v)) if v.size else float("nan")
        summ.append((roi, med))
        ax_h.hist(v, bins=40, range=(0, 80), alpha=0.5,
                  label=f"{roi}: median={med:.1f} (n={int(m.sum())})")
    ax_h.set_xlabel("tSNR"); ax_h.set_ylabel("voxel count")
    ax_h.set_xlim(0, 80); ax_h.legend(fontsize=9)
    ax_h.set_title("Within-ROI tSNR distribution (analysis mask)", fontsize=10.5)
    fig.tight_layout()
    out = outdir / "tsnr_fixed.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}  | " + "  ".join(f"{r}={m:.1f}" for r, m in summ))


def main():
    subs = sys.argv[1:] or [f"{i:02d}" for i in range(1, 11)]
    for sub in subs:
        outdir = OUTBASE / f"sub-{sub}"
        outdir.mkdir(parents=True, exist_ok=True)
        print(f"=== sub-{sub} ===")
        try:
            fig_overlay(sub, outdir)
            fig_tsnr(sub, outdir)
        except Exception as e:
            print(f"  ERROR sub-{sub}: {e}")


if __name__ == "__main__":
    main()
