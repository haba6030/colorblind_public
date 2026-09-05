# Paper Figures — Source of Truth (2026-09-02, rev. 2)

**Seven main figures + three supplementary figures.** The supplementary count was two until 2026-09-02, when `fig:landscape` was found to be compiling as main **Figure 8**: the `\renewcommand{\thefigure}` that turns a supplementary float into `S`-numbering was missing from that float alone. The reset now sits in `fig:landscape`, the first supplementary figure, and the three renumbered to S1--S3. Captions live in the LaTeX (`Methods/methods_v2.tex`, `Results/results_v4.tex`, `Supplementary/supplementary.tex`), **not** in `FIGURE_CAPTIONS.md` (that file is 2026-05-11, carries superseded filter parameters, and is kept for history only).

Figure numbers are assigned by LaTeX float order via semantic `\ref` labels. **Main** filenames are deliberately **not** renamed to match (e.g. `fig2_loro_loco` = Figure 4): numbers can shift again with float order, and renaming would break generating scripts, notes, and past commit references.

**The three supplementary figures are the exception, renamed 2026-09-02 on author instruction.** Their old names encoded *section* numbers that the S2--S21 -> S1--S20 pull had invalidated twice over (`figS18_landscape` sat in what is now §S17, `figS16_adjacc_saturation` in what is now §S18), so the names were pointing at the wrong section rather than merely at a stale number. They now carry their rendered **figure** number, and the generating scripts were renamed with them.

The map below is checked against `main.aux` (2026-09-02 build).

| Fig | `\label`          | File (PDF/PNG)             | Generating script                             | Type       | Data |
|----:|-------------------|----------------------------|-----------------------------------------------|------------|------|
| 1   | `fig:paradigm`    | `fig1_paradigm_v3`         | `scripts/generate_fig1_v3.py`                 | schematic  | — (mpl; replaces AI-generated `fig1_generated_v2`) |
| 2   | `fig:forward`     | `fig_forward_encoder`      | `scripts/generate_forward_encoder_fig.py` (moved out of `../archive/` 2026-09-02) | schematic | — (illustrative; no suptitle/stage subtitles since 2026-09-02 — the caption carries them) |
| 3   | `fig:pipeline`    | `fig3_workflow`            | PowerPoint composite of `fig3_assets/box*`; box 2 rebuilt by `scripts/generate_box2_loss_gates.py` and inserted by `scripts/patch_fig3_box2.py` | schematic | — (illustrative; box 2 shows the procedure only since 2026-09-03 — the selected combinations moved to Results) |
| 4   | `fig:loco`        | `fig2_loro_loco`           | `scripts/generate_fig2.py`                    | data chart | LORO/LOCO results (adjacent accuracy, chance 91/360) |
| 5   | `fig:geometry`    | `fig3_geometry_r6`         | `scripts/generate_fig3_geometry_r6.py`        | data chart | Procrustes disparity (single panel, no asterisks — §5.1 2026-09-02) |
| 6   | `fig:filter`      | `fig7_filter`              | `scripts/phase2/generate_fig7_filter.py`      | rendering  | analytical pre-image (no in-panel text since 2026-09-02: subject/β labels and per-hue δθ moved to caption/text) |
| 7   | `fig:filter_eval` | `fig8_filter_eval`         | `scripts/generate_fig8.py`                    | data chart | exp2 neural evaluation (N=2) |
| S1  | `fig:landscape`         | `figS1_landscape`         | `scripts/phase2/generate_figS1_landscape.py`  | data chart | real (7-HC pool + 300 resamples); moved from main Fig 6 into Supp §S17 (2026-09-02) |
| S2  | `fig:adjacc_saturation` | `figS2_adjacc_saturation` | `scripts/generate_figS2_adjacc_saturation.py` | data chart | run-count saturation; lives in Supp §S18 |
| S3  | `fig:forward_tuning`    | `figS3_forward_tuning`    | `scripts/generate_figS3_forward_tuning.py`    | data chart | forward-tuning ρ (companion to Fig 7); lives in Supp §S18 |

## Directory hygiene (2026-09-02, MANUSCRIPT_EDITS_CONSOLIDATED.md §5.4)

- The ten files above (PDF + PNG pairs) are the only images the manuscript loads; `\graphicspath{{Figures/}}` + PDF-first resolution.
- Moved to `old/`: `fig3_workflow_composited.pdf` (md5-identical duplicate of `fig3_workflow.pdf`), `fig3_geometry.pdf` (pre-R6 two-panel version), `fig5_generated_v.png` + `fig5_notes.md` (no `fig5` figure exists in the manuscript), `fig3_workflow.png` (unused raster copy; LaTeX resolves the PDF).
- `submission_assets/fig3_workflow.tif` — journal-submission-only TIFF, not read by LaTeX.
- `.DS_Store` is gitignored.

## Figure 3 is the one PowerPoint composite

Its slide (`fig3_assets/Presentation1_arial_fontpatched.pptx`) carries a full-slide background PICTURE that renders an OLDER version of the whole figure, with live shapes on top, so editing a box means covering that region rather than only deleting shapes. The 2026-09-03 box-2 rebuild is scripted end to end:

```bash
conda activate srm
cd docs/PAPER/Figures/scripts
export MATPLOTLIBRC="$PWD/inrc"
python generate_box2_loss_gates.py          # the new box-2 asset
python patch_fig3_box2.py --export          # patch the .pptx, export a PDF
cd ..
pdfcrop --bbox "6.0 73.0 956.4 417.8" \
        fig3_assets/Presentation1_box2_2026-09-03.pdf fig3_workflow.pdf
```

`patch_fig3_box2.py` also rewrites the font stack inside the embedded wheel SVGs, which declare matplotlib's default list with DejaVu Sans ahead of Arial. Without that step LibreOffice substitutes DejaVu for the wheel labels and the export stops being Arial-only; check with `pdffonts fig3_workflow.pdf`.

**The .pptx files are not tracked by git.** They are the only source able to regenerate this figure, so keep them.

## Canonical Phase-2 parameters (PIPELINE_2_CLOSURE.md 2026-06-01)

- sub-08 deutan: **(β_s, β_c) = (+6°, −42°)**, loss γ_OY + RDM_V2
- sub-09 protan: **(β_s, β_c) = (+2°, +24°)**, loss γ_all + RDM_V1

Figs 6 & 7 use these via the canonical A13 forward `two_comp.py:forward_2comp` (raw CIELab nominal-θ). Sign check: sub-08 c4 green δθ = −36.3° (NOT the frozen +19° variant).

## Regenerating Fig 6 and Supp Fig S1

```bash
conda activate srm
cd docs/PAPER/Figures/scripts/phase2
python generate_figS1_landscape.py   # Supp Fig S1; imports viz_closure_ground_plot (real data)
python generate_fig7_filter.py       # main Fig 6; imports two_comp + stim_lab_render
```

These scripts import canonical modules from `analysis/phase5_filter_optimization/scripts/` (added to `sys.path` at runtime). They are reproducible in-repo but **not** standalone — they depend on that package and, for Supp Fig S1, on the local C010 amplitudes at `analysis/phase1_procrustes_decoding/results/full_dataset_C010/`.

## Superseded (archived 2026-06-05)

Moved to `../archive/figures_superseded_2026-06-05/`: `fig4_twocomp` (old landscape, stale params, never wired), `fig5_preimage` + `fig5a/b_preimage` (old 4-column filter, stale params (38,−14)/(6,−22)). The old `scripts/generate_fig4.py` (landscape) and `generate_fig5.py` (4-col) are superseded by `scripts/phase2/` and retained for history only. `scripts/phase2/generate_fig5_pipeline.py` produced the old `fig5_pipeline` schematic, replaced by the `fig3_workflow` composite.

## Known open items

1. ~~**Fig metric mismatch**: `generate_fig2.py` plots MAE (°) but the manuscript reports adjacent accuracy.~~ **CLOSED 2026-09-01** — the current `generate_fig2.py` (2026-08-17) plots panels B·C as `adjacent_acc` with the chance level `91/360 = 0.25` marked; script and caption agree.
