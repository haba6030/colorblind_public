# colorblind_public

Code and result files for the manuscript

> **Preserved categorical identification with impaired hue interpolation in color vision deficiency motivates a cortically derived correction filter**
> Jinil Kim, Albert Minkue Cho, Jungwoo Seo, Jiook Cha (Seoul National University)

Two adults with red-green colour vision deficiency (one deutan, one protan) and seven colour-normal controls viewed eight uniform colour discs during fMRI and completed hue-discrimination and identification tasks. Categorical hue identity remained decodable from the CVD participants' visual cortex, whereas continuous hue interpolation at hV4 fell to chance and the hue geometry departed from the control reference. A two-parameter cortical distortion model was fitted to each participant's psychophysical and neural data and inverted into an individualized stimulus-space filter, which both participants then evaluated in a second session against a deployed accessibility filter.

This repository is organised by the stages of the manuscript. Each stage folder holds the analysis scripts, the committed result files those scripts produced, and one executed Jupyter notebook that compares the committed results with every number printed in the manuscript for that stage. `REPORT.md` tallies those checks; `MANIFEST.md` lists the printed values; `MAP.md` links each value to the file and script behind it.

## Quick start

```bash
git clone https://github.com/haba6030/colorblind_public
cd colorblind_public
conda env create -f environment.yml && conda activate colorblind
python run_notebooks.py        # re-executes the nine notebooks and rewrites REPORT.md
```

The notebooks need only numpy and scipy. They read the committed result files; they do not recompute the analyses from imaging data, because the imaging and behavioural data are not distributed (see below).

## Layout

| Folder | Stage | Manuscript |
|---|---|---|
| [`00_preprocessing/`](00_preprocessing/) | Preprocessing, registration and response estimation | Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage) |
| [`01_decoding/`](01_decoding/) | Hue classification (LORO) and hue interpolation (LOCO) | Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness) |
| [`02_geometry/`](02_geometry/) | Representational geometry: SRM, Procrustes disparity and its validity checks | Results section 2; Figure 5; Methods 'Shared Response Model', 'Representational geometry'; Supplementary S4 (dimensionality), S8 (LOO disparity), S9 (alignment-independent checks), S10 (activation), S18 (validity of the geometric comparison) |
| [`03_psychophysics/`](03_psychophysics/) | Session-1 hue-discrimination thresholds (JND) and identification (8AFC) | Results section 3; Methods 'Psychophysical tasks'; Supplementary S1 (tab:jnd_baseline, tab:staircase_pairs) |
| [`04_distortion_model/`](04_distortion_model/) | Cortical distortion model: inverse fitting and parameter selection | Results sections 4-5; Methods 'Cortical distortion model', 'Inverse fitting', 'Parameter selection'; Supplementary S11 (retinal-family model), S13 (stability), Figure S1, tab:modelfits, tab:fit_stability |
| [`05_identifiability/`](05_identifiability/) | Identifiability and recovery of the fitted parameters | Methods 'Identifiability and recovery'; Supplementary S12 (tab:identifiability); S13 control leave-one-out magnitude anchor |
| [`06_filter/`](06_filter/) | Per-participant stimulus-space filter (pre-image of the fitted distortion) | Results section 6; Figure 6; Methods 'Stimulus-space filter and its evaluation' |
| [`07_filter_evaluation/`](07_filter_evaluation/) | Second-session evaluation of the filters (psychophysics and fMRI) | Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3 |
| [`08_sensitivity/`](08_sensitivity/) | Preprocessing sensitivity: the head-motion-corrected pipeline | Supplementary S2 (tab:motion_arms, tab:interp_arms, session-2 endpoints), S13 sign stability, S18 (tab:color_specificity, head-motion column) |

Shared modules are in `common/` (hue-channel basis model, canonical LOCO readouts, colour definitions, check helpers). `figures/` holds the ten manuscript figures with their generators, and `paper/` the LaTeX source.

## Figures

| Manuscript | File | Generator |
|---|---|---|
| Figure 1 | `figures/fig1_paradigm_v3.pdf` | `figures/scripts/generate_fig1_v3.py` |
| Figure 2 | `figures/fig_forward_encoder.pdf` | `figures/scripts/generate_forward_encoder_fig.py` |
| Figure 3 | `figures/fig3_workflow.pdf` | `figures/scripts/generate_box2_loss_gates.py + patch_fig3_box2.py (PowerPoint composite)` |
| Figure 4 | `figures/fig2_loro_loco.pdf` | `01_decoding/scripts/figure_generators/generate_fig2.py` |
| Figure 5 | `figures/fig3_geometry_r6.pdf` | `02_geometry/scripts/figure_generators/generate_fig3_geometry_r6.py` |
| Figure 6 | `figures/fig7_filter.pdf` | `06_filter/scripts/figure_generators/generate_fig7_filter.py` |
| Figure 7 | `figures/fig8_filter_eval.pdf` | `07_filter_evaluation/scripts/figure_generators/generate_fig8.py` |
| Figure S1 | `figures/figS1_landscape.pdf` | `04_distortion_model/scripts/figure_generators/generate_figS1_landscape.py` |
| Figure S2 | `figures/figS2_adjacc_saturation.pdf` | `07_filter_evaluation/scripts/figure_generators/generate_figS2_adjacc_saturation.py` |
| Figure S3 | `figures/figS3_forward_tuning.pdf` | `07_filter_evaluation/scripts/figure_generators/generate_figS3_forward_tuning.py` |

## Participants

Controls sub-01 to sub-07 (n = 7). CVD participants sub-08 (deutan) and sub-09 (protan) (n = 2). A third CVD participant (sub-10) was scanned in session 1 but did not complete the second-session filter evaluation and is excluded from every analysis in the manuscript. Some committed result files still carry sub-10 entries from the original runs; the notebooks read only the sub-08 and sub-09 entries and say so where it matters.

## Data

The neuroimaging and behavioural data cannot be shared openly. The protocol approved by the Seoul National University Institutional Review Board (SNUIRB No. 2510/002-023) and the written consent obtained under it place all study data in an encrypted store to which the principal investigator controls access. Requests for access should be directed to the principal investigator (Jiook Cha, connectome@snu.ac.kr); the conditions are stated in the Data and Code Availability section of the manuscript (`paper/main.tex`).

What is distributed: analysis code, per-participant summary statistics (decoding accuracies, disparities, fitted parameters, thresholds), permutation null distributions, figures, and the manuscript source. What is not: raw or preprocessed images, voxel-level amplitude arrays, trial-level behavioural files.

## Reproduction status

See `REPORT.md`. Checks that could not be performed from committed files (for example, values computed on the server from field maps) are listed there as pointers rather than dropped. Analyses that require BrainIAK with MPI or SLURM (shared response model, N = 300 resample fits, permutation tests) are verified against their committed outputs, not re-run; `REPRODUCE.md` explains each case.

## Citation and licence

Code is released under the MIT licence (`LICENSE`); result files and figures under CC BY 4.0 (`LICENSE-RESULTS`). Please cite the manuscript and, for the code, `CITATION.cff` (a Zenodo DOI will be added at release).

Built from the development repository at commit 53c81c2 on 2026-09-06.
