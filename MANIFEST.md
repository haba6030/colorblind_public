# Manifest of reported values

Every number printed in `paper/Results/results_v4.tex`, the numeric parts of `paper/Methods/methods_v2.tex`, and `paper/Supplementary/supplementary.tex`, with the check id used by the stage notebooks. `kind` = check (numeric comparison), table (a table verified cell by cell inside the notebook), pointer (no committed artifact; listed, not verified).


## 00_preprocessing — Preprocessing, registration and response estimation

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 00.01 | S2 ¶1 | mean FD, nine analysed participants (mm) | `0.318` | check |
| 00.02 | S2 ¶1 | SD of FD, nine participants | `0.044` | check |
| 00.03 | S2 ¶1 | mean FD, controls | `0.313` | check |
| 00.04 | S2 ¶1 | SD of FD, controls | `0.042` | check |
| 00.05 | S2 ¶1 | mean FD, CVD | `0.338` | check |
| 00.06 | S2 ¶1 | SD of FD, CVD | `0.046` | check |
| 00.07 | S2 ¶1 | percentage of volumes with FD > 0.5 mm | `16.2` | check |
| 00.08 | S2 ¶1 | smallest tSNR reduction over ROIs (%) | `1.7` | check |
| 00.09 | S2 ¶1 | largest tSNR reduction over ROIs (%) | `2.7` | check |
| 00.10 | S3 | mean ROI coverage (%) | `83.5` | check |
| 00.11 | S3 | SD of ROI coverage (%) | `22.7` | check |
| 00.12 | S3 | ROI voxels with reliable stimulus-evoked responses (%) | `99.5` | check |
| 00.13 | S3 | lowest coverage in the sample (%) | `30.8` | check |
| 00.14 | S3 | number of analysed participants | `9` | check |
| 00.15 | S2 'Susceptibility distortion' | field-map displacement within ROIs 0.01-0.76 voxels (0.02-1.52 mm); within-ROI variation 0.05-0.21 voxels (0.38 in one participant) | `no committed artifact (server field-map computation)` | pointer |
| 00.16 | S2 'Registration' | run-to-run displacement of the MI solution 0.9-4.2 mm; defacing moved it by 1.9 mm (deutan) and 9.4 mm (protan) | `no committed artifact (server registration logs)` | pointer |
| 00.17 | S2 'Registration' | BBR snapped the slab about 10 mm off on visual inspection; MI within about 1 mm | `visual criterion; bbr_vs_mi_displacement.json gives centroid displacement for three participants (descriptive)` | pointer |

## 01_decoding — Hue classification (LORO) and hue interpolation (LOCO)

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 01.01 | tab:alignment | LORO classification, Procrustes space: 4 ROIs x (controls, deutan, protan) | `12 cells, see 01.T1.*` | table |
| 01.02 | Results §1 ¶1 | lowest CVD classification cell (hV4) | `0.375` | check |
| 01.03 | Results §1 ¶1 | lowest cell as a multiple of chance | `3.0` | check |
| 01.04 | Results §1 ¶1 | every CVD cell above chance 0.125 | `0.125` | check |
| 01.05 | S16 | smallest |d_cc| over the eight classification contrasts | `0.25` | check |
| 01.06 | S16 | largest |d_cc| | `1.58` | check |
| 01.07 | S16 | smallest two-tailed p | `0.189` | check |
| 01.08 | S16 | deutan V3 d_cc | `-1.58` | check |
| 01.09 | S16 | hV4 d_cc, both participants | `-1.08` | check |
| 01.10 | S16 | hV4 two-tailed p, both participants | `0.352` | check |
| 01.11 | Results §1 ¶2 | HC-to-CVD mean accuracy over eight cells | `0.432` | check |
| 01.12 | Results §1 ¶2 | number of CVD cells | `8` | check |
| 01.13 | Results §1 ¶2 | t(7) against chance 0.125 | `6.51` | check |
| 01.14 | Results §1 ¶2 | p against chance (< 0.001) | `0.001` | check |
| 01.15 | Results §1 ¶2 | HC-to-HC mean over 28 cells | `0.526` | check |
| 01.16 | S5 | Mann-Whitney U | `163.5` | check |
| 01.17 | Results §1 ¶2 | Mann-Whitney p | `0.052` | check |
| 01.18 | S5 | rank-biserial r | `0.46` | check |
| 01.19 | S5 | lowest single HC-to-CVD cell | `0.271` | check |
| 01.20 | Results §1 ¶3 | hV4 control adjacent accuracy | `0.456` | check |
| 01.21 | Results §1 ¶3 | hV4 SEM (n = 7) | `0.039` | check |
| 01.22 | Results §1 ¶3 | hV4 permutation p | `0.011` | check |
| 01.23 | Results §1 ¶3 | V1 control adjacent accuracy | `0.393` | check |
| 01.24 | Results §1 ¶3 | V1 permutation p | `0.164` | check |
| 01.25 | Results §1 ¶3 | V2 control adjacent accuracy | `0.357` | check |
| 01.26 | Results §1 ¶3 | V2 permutation p | `0.424` | check |
| 01.27 | Results §1 ¶3 | V3 control adjacent accuracy | `0.339` | check |
| 01.28 | Results §1 ¶3 | V3 permutation p | `0.586` | check |
| 01.29 | Results §1 ¶3 | permutation null lies near 0.35 (all four ROIs within 0.34-0.36) | `(0.34, 0.36)` | check |
| 01.30 | Results §1 ¶3 | all four ROIs exceed analytic chance 0.25 | `0.25` | check |
| 01.31 | Results §1 ¶3 | number of permutations | `1000` | check |
| 01.32 | Results §1 ¶4 | deutan hV4 adjacent accuracy | `0.25` | check |
| 01.33 | Results §1 ¶4 | deutan Crawford-Howell t | `-1.89` | check |
| 01.34 | Results §1 ¶4 | deutan p (one-tailed) | `0.054` | check |
| 01.35 | Results §1 ¶4 | deutan d_cc | `-2.02` | check |
| 01.36 | Results §1 ¶4 | protan hV4 adjacent accuracy | `0.13` | check |
| 01.37 | Results §1 ¶4 | protan Crawford-Howell t | `-3.04` | check |
| 01.38 | Results §1 ¶4 | protan p (one-tailed) | `0.012` | check |
| 01.39 | Results §1 ¶4 | protan d_cc | `-3.25` | check |
| 01.40 | Results §1 ¶4 | control mean | `0.46` | check |
| 01.41 | Results §1 ¶4 | all seven controls above the protan participant | `7` | check |
| 01.42 | Results §1 ¶4 | at least five controls above the deutan participant | `5` | check |
| 01.43 | Results §1 ¶5 | deutan blue adjacent accuracy = 0 | `0.0` | check |
| 01.44 | Results §1 ¶5 | deutan purple = 0 | `0.0` | check |
| 01.45 | Results §1 ¶5 | deutan magenta = 0 | `0.0` | check |
| 01.46 | Results §1 ¶5 | protan blue = 0 | `0.0` | check |
| 01.47 | Results §1 ¶5 | protan purple = 0 | `0.0` | check |
| 01.48 | Results §1 ¶5 | protan magenta = 0 | `0.0` | check |
| 01.49 | tab:effect_sizes | blue t | `-1.93` | check |
| 01.50 | tab:effect_sizes | blue p | `0.051` | check |
| 01.51 | tab:effect_sizes | blue d_cc | `-2.06` | check |
| 01.52 | tab:effect_sizes | purple t | `-0.79` | check |
| 01.53 | tab:effect_sizes | purple p | `0.229` | check |
| 01.54 | tab:effect_sizes | purple d_cc | `-0.85` | check |
| 01.55 | tab:effect_sizes | magenta t | `-1.47` | check |
| 01.56 | tab:effect_sizes | magenta p | `0.096` | check |
| 01.57 | tab:effect_sizes | magenta d_cc | `-1.57` | check |
| 01.58 | Results §1 ¶5 | smallest per-hue p over 16 tests (p >= 0.051 at every hue) | `0.051` | check |
| 01.59 | tab:alignment | LOCO interpolation, Procrustes space: 4 ROIs x (controls, deutan, protan) | `12 cells, see 01.T2.*` | table |
| 01.60 | tab:loro_decoders | LORO six decoders x 4 ROIs: control mean, SD, deutan, protan | `96 cells, see 01.T3.*` | table |
| 01.61 | S5 ¶2 | LDA has the highest control mean at every ROI | `True` | check |
| 01.62 | S5 ¶2 | four of the six decoders exceed chance at every ROI (by control mean > 0.125 the count is five, kernel ridge at V3 being 0.158 ± 0.052; by control mean minus one SD it is three: the printed 'four' matches neither criterion) | `4` | check |
| 01.63 | tab:loco_decoders | LOCO six decoders x 4 ROIs: control mean, SD, deutan, protan | `96 cells, see 01.T4.*` | table |
| 01.64 | S5 ¶3 | hue-channel basis exceeds 0.25 at hV4, V1 and V2 (widest margin at hV4) | `True` | check |
| 01.65 | S5 ¶3 | classifiers remain at or below their 0.375 chance level at every ROI | `0.375` | check |
| 01.66 | S5 ¶4 | kernel ridge returns exactly 0.000 in every participant and ROI | `0.0` | check |
| 01.67 | S7 | fixed-reference hue-channel LORO accuracy (pooled over ten participants incl. sub-10, as printed) | `0.545` | check |
| 01.68 | S7 | nested re-estimation raises it to (same pool) | `0.578` | check |
| 01.68b | S7 | the increase persists with sub-10 excluded (nine participants) | `True` | check |
| 01.69 | S17 | lowest CVD classification cell over both spaces | `0.312` | check |
| 01.70 | S17 | Procrustes gives the higher control classification mean at all four ROIs | `4` | check |
| 01.71 | S17 | and the higher control interpolation mean at three of four | `3` | check |
| 01.72 | S17 | Procrustes higher in 26 of 36 participant-by-region classification cells | `26` | check |
| 01.73 | S17 | SRM V1 classification d_cc, deutan | `0.59` | check |
| 01.74 | S17 | SRM V1 classification d_cc, protan | `0.79` | check |
| 01.75 | S17 | Procrustes V1 classification d_cc, both | `-0.25` | check |
| 01.76 | S6 | GCV searches seven penalties from 1e-3 to 1e3 | `(7, 0.001, 1000.0)` | check |

## 02_geometry — Representational geometry: SRM, Procrustes disparity and its validity checks

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 02.01 | tab:disparity_loso | 8 participant-by-ROI cells x (t, p, d_cc) x two estimators | `48 cells, see 02.T1.*` | table |
| 02.02 | Results §2 ¶1 | largest deviation is V1 in the protan participant (primary pipeline) | `V1` | check |
| 02.03 | Results §2 ¶1 | largest deviation is V2 in the deutan participant (primary pipeline) | `V2` | check |
| 02.04 | Results §2 ¶1 | under the symmetric LOSO reference only protan V1 reaches significance | `[('sub-09', 'V1')]` | check |
| 02.05 | Methods 'Shared Response Model' | seed, permutation counts recorded in the run config | `(42, 1000, 10000)` | check |
| 02.06 | S4 | k = 4 at V1 | `4` | check |
| 02.07 | S4 | k = 4 at V2 | `4` | check |
| 02.08 | S4 | k = 3 at V3 | `3` | check |
| 02.09 | S4 | k = 3 at hV4 | `3` | check |
| 02.10 | S4 | cross-participant RDM Spearman r at V1 | `0.6` | check |
| 02.11 | S4 | at V2 | `0.57` | check |
| 02.12 | S4 | at V3 | `0.55` | check |
| 02.13 | S4 | at hV4 | `0.32` | check |
| 02.14 | S4 | candidate range 2..6 over seven folds | `('2', '6', 7)` | check |
| 02.15 | tab:triangulation | Spearman r, 3 distances x (4 ROIs + pooled) | `15 cells, see 02.T2.*` | table |
| 02.16 | S9 ¶1 | crossnobis pooled p < .001 | `0.001` | check |
| 02.17 | S9 ¶1 | PCA pooled p < .001 | `0.001` | check |
| 02.18 | S9 ¶3 | PCA-CCA pooled p < .001 | `0.001` | check |
| 02.19 | S9 ¶2 | crossnobis V1 convergent r p = .005 | `0.005` | check |
| 02.20 | S9 ¶2 | crossnobis V2 convergent r p = .025 | `0.025` | check |
| 02.21 | S9 ¶2 | control-to-control minus control-to-CVD crossnobis RDM similarity at V1 | `0.104` | check |
| 02.22 | S9 ¶2 | the three remaining ROIs give differences below 0.05 | `0.05` | check |
| 02.23 | S9 ¶2 | permutation p = .120 for the V1 difference | `the committed permutation was run with sub-10 included (p = .051); the n = 2 permutation needs the per-participant crossnobis RDMs, which are not stored` | pointer |
| 02.24 | S9 ¶3 | Hedges g under PCA ranges from -0.13 to +0.40 | `(-0.13, 0.4)` | check |
| 02.25 | S9 ¶3 | Hedges g under PCA-CCA ranges from -0.13 to +0.16 | `(-0.13, 0.16)` | check |
| 02.26 | tab:variance_explained | 4 ROIs x (k, controls, CVD, g) | `16 cells, see 02.T3.*` | table |
| 02.27 | S9 ¶4 | variance explained higher in CVD than controls at all four ROIs | `True` | check |
| 02.28 | S9 ¶4 | variance explained vs disparity, pooled Spearman r = -0.214 (common-space disparity, 36 points) | `-0.214` | check |
| 02.29 | S9 ¶4 | p = .211 | `0.211` | check |
| 02.30 | S10 ¶2 | mean |beta|: every single-case p >= 0.182 | `0.182` | check |
| 02.31 | S10 ¶2 | largest mean |beta| deviation is the deutan participant at V3 | `('sub-08', 'V3')` | check |
| 02.32 | S10 ¶2 | its d_cc = +1.61 | `1.61` | check |
| 02.33 | S10 ¶2 | largest modulation-depth deviation is the deutan participant at V2 | `('sub-08', 'V2')` | check |
| 02.34 | S10 ¶2 | its d_cc = +2.16 | `2.16` | check |
| 02.35 | S10 ¶2 | its p = 0.090 | `0.09` | check |
| 02.36 | S10 ¶2 | voxel-level colour selectivity F > 4 in V1-V3 (mean F over the nine participants; one participant's V3 F is 1.46, p = .18, so the statement holds at the group level) | `4.0` | check |
| 02.37 | S10 ¶2 | p < 0.001 for every participant in V1 and V2 | `0.001` | check |
| 02.38 | S10 ¶3 | activation metrics vs disparity: 20 tests, r from -0.29 to +0.51, all p >= 0.130 | `(20, -0.29, 0.51, 0.13)` | check |
| 02.39 | tab:frozen_control | 4 ROIs x (n, detected and mean z under re-estimated and frozen projections) | `20 cells, see 02.T4.*` | table |
| 02.40 | tab:color_specificity | 35 participant-by-region p_perm cells, primary pipeline | `35 cells, see 02.T5.*` | table |
| 02.41 | S18 ¶2 | cells below .05 out of 35 | `16` | check |
| 02.42 | S18 ¶2 | chance expectation 35 x 0.05 | `1.8` | check |
| 02.43 | S18 ¶2 | cells surviving BH over 35 | `7` | check |
| 02.44 | S18 ¶2 | deutan V2 q | `0.018` | check |
| 02.45 | S18 ¶2 | protan V3 q | `0.012` | check |
| 02.46 | S18 ¶2 | the two CVD survivors are deutan V2 and protan V3 | `[('sub-08', 'V2'), ('sub-09', 'V3')]` | check |
| 02.47 | S18 ¶2 | V3: 5 of its 9 cells survive | `5` | check |
| 02.48 | S18 ¶2 | deutan V2 is the lowest V2 value among the nine participants | `sub-08` | check |
| 02.49 | S18 ¶1 | number of permutations | `1000` | check |
| 02.50 | S18 ¶4 | protan V1 split-half pattern reliability | `0.847` | check |
| 02.51 | S18 ¶4 | above the highest control value | `True` | check |
| 02.52 | S18 ¶4 | eight-way classification 0.79 at protan V1 | `readout not stored in a committed artifact (the frozen-permutation driver printed it to stdout)` | pointer |
| 02.53 | S18 ¶4 | protan V1 disparity with identity labels | `1.037` | check |
| 02.54 | S18 ¶4 | after a 45-degree rotation of the labels | `0.788` | check |
| 02.55 | S18 ¶4 | gain (%) | `24.0` | check |
| 02.56 | S18 ¶4 | control gain mean (%) | `3.5` | check |
| 02.57 | S18 ¶4 | control gain SD (%) | `5.9` | check |
| 02.58 | S18 ¶4 | Crawford-Howell t of the protan gain | `3.22` | check |
| 02.59 | S18 ¶4 | p | `0.009` | check |
| 02.60 | S18 ¶4 | d_cc | `3.44` | check |
| 02.61 | S18 ¶4 | control mean disparity at V1 | `0.839` | check |
| 02.62 | S18 ¶4 | control SD | `0.087` | check |
| 02.63 | S18 ¶4 | rotated protan geometry lies below the control mean | `True` | check |
| 02.64 | S18 ¶4 | second-order RSA rho 0.00 -> +0.52 (controls 0.45), z = +5.02, p = .002, p_adj = .032; SRM 225-degree shift +0.50; PCA optimum at 135 degrees p = .048 | `no committed artifact for the second-order RSA cyclic-shift statistics` | pointer |
| 02.65 | S18 ¶4 | deutan: identity mapping optimal at V1, V2 and V3 | `True` | check |

## 03_psychophysics — Session-1 hue-discrimination thresholds (JND) and identification (8AFC)

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 03.01 | tab:jnd_baseline | 8 pairs x (control mean, SD, deutan t/gamma/z, protan t/gamma/z) | `64 cells, see 03.T1.*` | table |
| 03.02 | Results §3 ¶2 | mean |z| over eight pairs, deutan | `2.24` | check |
| 03.03 | Results §3 ¶2 | mean |z|, protan | `0.9` | check |
| 03.04 | Results §3 ¶2 | remaining pairs lie within |z| <= 1.23 in both participants | `1.23` | check |
| 03.05 | Results §3 ¶1 | deutan exceeds z = +4 on both deutan-axis pairs and on yellow-purple | `4.0` | check |
| 03.06 | Results §3 ¶1 | protan exceeds z = +2 on green-blue only | `['green-blue']` | check |
| 03.07 | tab:jnd_baseline caption | staircases collected in the study | `208` | check |
| 03.08 | tab:jnd_baseline caption | staircases censored at the largest presentable separation | `2` | check |
| 03.09 | tab:jnd_baseline caption | both censored staircases are the deutan orange-yellow pair | `[('sub-08_jnd_ses1_no_filter', 'orange-yellow')]` | check |
| 03.10 | tab:staircase_pairs | 8 pairs x 6 columns x (high, low) | `96 cells, see 03.T2.*` | table |
| 03.11 | S1 ¶ after tab:staircase_pairs | number of pair-level cells (13 files x 8 pairs) | `104` | check |
| 03.12 | S1 | median difference between the two staircases | `0.015` | check |
| 03.13 | S1 | cells whose two estimates differ by more than 0.10 | `3` | check |
| 03.14 | S1 | largest difference | `0.515` | check |
| 03.15 | S1 | it is the protan participant's orange-yellow pair under the individualized filter | `('sub-09/individualized', 'orange-yellow')` | check |
| 03.16 | S1 | that cell's z with the mean of the two tracks | `1.33` | check |
| 03.17 | S1 | z restricted to the converged track | `-0.58` | check |
| 03.18 | S1 | protan mean |z| under the individualized filter | `0.93` | check |
| 03.19 | S1 | same with the converged track | `0.84` | check |

## 04_distortion_model — Cortical distortion model: inverse fitting and parameter selection

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 04.01 | Methods grid | beta_s grid: 26 values over [0, 50] at 2 degrees | `(26, 0.0, 50.0, 2.0)` | check |
| 04.02 | Methods grid | beta_c grid: 51 values over [-50, 50] at 2 degrees | `(51, -50.0, 50.0, 2.0)` | check |
| 04.03 | Methods grid | grid cells | `1326` | check |
| 04.04 | Methods model | theta_conf protan = 16 degrees | `16.0` | check |
| 04.05 | Methods model | theta_conf deutan = 150 degrees | `150.0` | check |
| 04.06 | S11 / Methods grid | R+C gain g over [0, 3] in steps of 0.05 (61 values) | `(0.0, 3.0, 0.05, 61)` | check |
| 04.07 | S11 | Delta-lambda anchors deutan 6.0, 6.5, 8.0 nm | `[6.0, 6.5, 8.0]` | check |
| 04.08 | S11 | Delta-lambda anchors protan 1.5, 3.0, 10.0 nm | `[1.5, 3.0, 10.0]` | check |
| 04.09 | Methods selection | N = 300 control 5-train/2-test resamples | `(300, 5)` | check |
| 04.10 | Results §4 ¶2 | deutan combinations passing the boundary-saturation gate | `25` | check |
| 04.11 | Results §4 ¶2 | protan combinations passing | `4` | check |
| 04.12 | Results §4 ¶2 | every gate-passing protan combination with an RDM atom returns (2, +24) | `True` | check |
| 04.13 | Results §4 ¶1 | the LOCO family entered neither winning combination | `False` | check |
| 04.14 | tab:modelfits | deutan selected beta_s | `6.0` | check |
| 04.15 | tab:modelfits | deutan selected beta_c | `-42.0` | check |
| 04.16 | tab:modelfits | deutan selected L_test | `-2.36` | check |
| 04.17 | tab:modelfits | deutan selected IQR | `2.15` | check |
| 04.18 | tab:modelfits | deutan next-ranked beta_s | `38.0` | check |
| 04.19 | tab:modelfits | deutan next-ranked beta_c | `-10.0` | check |
| 04.20 | tab:modelfits | deutan next-ranked L_test | `-1.14` | check |
| 04.21 | tab:modelfits | deutan next-ranked IQR | `0.86` | check |
| 04.22 | tab:modelfits | protan selected beta_s | `2.0` | check |
| 04.23 | tab:modelfits | protan selected beta_c | `24.0` | check |
| 04.24 | tab:modelfits | protan selected L_test | `-1.54` | check |
| 04.25 | tab:modelfits | protan selected IQR | `1.42` | check |
| 04.26 | tab:modelfits | protan next-ranked returns the same estimate | `(2.0, 24.0)` | check |
| 04.27 | tab:modelfits | protan next-ranked L_test | `-1.52` | check |
| 04.28 | tab:modelfits | protan next-ranked IQR | `1.41` | check |
| 04.29 | Results §4 ¶2 | the deutan next-ranked candidate is beta_s-dominant | `True` | check |
| 04.30 | S11 / tab:modelfits | deutan R+C: 100% of resamples saturate at the reported anchor (lowest held-out loss; the three anchors give 0.71-1.00) | `1.0` | check |
| 04.31 | S11 / tab:modelfits | deutan R+C gain at the boundary g = 3.0 | `3.0` | check |
| 04.32 | S11 ¶3 | protan R+C: 41% saturated at the best anchor | `41.0` | check |
| 04.33 | S11 ¶3 | protan R+C gain g = 2.95 | `2.95` | check |
| 04.34 | S11 ¶3 | protan R+C held-out loss -0.86 | `-0.86` | check |
| 04.35 | S11 ¶3 | protan saturation ranges from 0% to 100% across anchors | `(0.0, 1.0)` | check |
| 04.36 | tab:fit_stability | deutan separation d at V1 / V2 / V3 / hV4 | `[2.31, 1.94, 0.86, 2.19]` | check |
| 04.37 | tab:fit_stability | protan separation d at V1 / V2 / V3 / hV4 | `[0.81, -0.23, -0.48, -0.24]` | check |
| 04.38 | Results §4 ¶1 | RDM atom admissible at all four ROIs in deutan and V1 alone in protan (d >= +0.5) | `(4, ['V1'])` | check |
| 04.39 | tab:fit_stability | deutan grid percentile (%) | `4.6` | check |
| 04.40 | tab:fit_stability | protan grid percentile (%) | `8.1` | check |
| 04.41 | Results §4 ¶3 | both within the top 8% of the 1,326-cell grid | `True` | check |
| 04.42 | tab:fit_stability | deutan Delta-L psychophysical atom | `-13.85` | check |
| 04.43 | tab:fit_stability | deutan folds favouring it (of 7) | `5` | check |
| 04.44 | tab:fit_stability | protan Delta-L psychophysical atom | `0.01` | check |
| 04.45 | tab:fit_stability | protan folds favouring it | `3` | check |
| 04.46 | tab:fit_stability | deutan Delta-L RDM atom | `-0.41` | check |
| 04.47 | tab:fit_stability | deutan RDM folds (7 of 7) | `7` | check |
| 04.48 | tab:fit_stability | protan Delta-L RDM atom | `-0.47` | check |
| 04.49 | tab:fit_stability | protan RDM folds (7 of 7) | `7` | check |
| 04.50 | Results §5 ¶1 | protan psychophysical atoms alone | `(26.0, 4.0)` | check |
| 04.51 | Results §5 ¶1 | protan RDM atom alone | `(0.0, 24.0)` | check |
| 04.52 | Results §5 ¶2 | deutan psychophysical atoms alone | `(16.0, -44.0)` | check |
| 04.53 | Results §5 ¶2 | deutan RDM atom alone | `(4.0, -26.0)` | check |
| 04.54 | Results §5 ¶2 | all three deutan argmins on the same side of the confusion axis (beta_c < 0) | `True` | check |
| 04.55 | tab:fit_stability | deutan selected combination, SRM basis | `(8.0, -42.0)` | check |
| 04.56 | tab:fit_stability | protan selected combination, SRM basis | `(32.0, 0.0)` | check |
| 04.57 | tab:fit_stability | deutan boundary-saturation rate, psychophysical alone | `0.23` | check |
| 04.58 | tab:fit_stability | deutan boundary-saturation rate, selected | `0.09` | check |
| 04.59 | Results §5 ¶2 | adding the RDM atom more than halved the deutan saturation rate | `True` | check |
| 04.60 | tab:fit_stability | protan boundary-saturation rate, psychophysical alone -> selected | `(0.0, 0.0)` | check |
| 04.61 | tab:fit_stability | deutan parameter IQR, PCA: psychophysical alone | `(18.0, 6.0)` | check |
| 04.62 | tab:fit_stability | deutan parameter IQR, PCA: selected | `(8.0, 2.0)` | check |
| 04.63 | tab:fit_stability | protan parameter IQR, PCA: psychophysical alone | `(6.0, 4.0)` | check |
| 04.64 | tab:fit_stability | protan parameter IQR, PCA: selected | `(0.0, 0.0)` | check |
| 04.65 | tab:fit_stability | deutan parameter IQR, SRM: psychophysical alone | `(18.0, 6.0)` | check |
| 04.66 | tab:fit_stability | deutan parameter IQR, SRM: selected | `(10.0, 4.0)` | check |
| 04.67 | tab:fit_stability | protan parameter IQR, SRM: psychophysical alone | `(6.0, 4.0)` | check |
| 04.68 | tab:fit_stability | protan parameter IQR, SRM: selected | `(0.0, 2.0)` | check |
| 04.69 | S13 'Resample structure' | deutan beta_c at -42 or -44 in 202 of 300 resamples | `202` | check |
| 04.70 | S13 | deutan beta_c stays within [-48, -36] | `(-48.0, -36.0)` | check |
| 04.71 | S13 | deutan beta_s distributed from 0 to 14 | `(0.0, 14.0)` | check |
| 04.72 | S12 'Basis caveat' | protan SRM basis: modal argmin (32, 0) in 171 of 300 resamples | `171` | check |
| 04.73 | S12 'Basis caveat' | protan SRM basis: beta_c positive in 17% | `17.0` | check |
| 04.74 | S12 'Basis caveat' | and negative in 26% | `26.0` | check |
| 04.75 | S12 'Basis caveat' | deutan sign holds in 300 of 300 resamples under both bases | `(300, 300)` | check |
| 04.76 | Results §4 ¶4 | deutan |beta_c| = 42 exceeds the 26-degree recovery uncertainty; protan 24 lies within it | `(True, True)` | check |
| 04.77 | S13 'Sign stability' | deutan beta_c median, primary | `-42.0` | check |
| 04.78 | S13 | deutan beta_c median, head-motion correction | `-46.0` | check |
| 04.79 | S13 | deutan fraction of negative resamples, primary | `1.0` | check |
| 04.80 | S13 | deutan fraction negative, head-motion correction | `0.947` | check |
| 04.81 | S13 | protan beta_c median, primary | `24.0` | check |
| 04.82 | S13 | protan beta_c median, head-motion correction | `-12.0` | check |
| 04.83 | S13 | protan negative resamples, head-motion correction (%) | `79.3` | check |
| 04.84 | S13 | protan negative resamples, primary (none) | `0.0` | check |
| 04.85 | S13 | deutan combined boundary fraction rises from 0.09 | `0.09` | check |
| 04.86 | S13 | to 0.72 | `0.72` | check |
| 04.87 | Figure S1 | loss landscape reconstruction on the seven-control pool | `figure committed; regeneration requires the amplitude arrays (scripts/viz_closure_ground_plot.py)` | pointer |

## 05_identifiability — Identifiability and recovery of the fitted parameters

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 05.01 | S12 ¶3 | 7 donors x 20 draws = 140 samples per candidate | `(7, 20)` | check |
| 05.02 | tab:identifiability | deutan f_10 | `0.26` | check |
| 05.03 | tab:identifiability | protan f_10 | `0.14` | check |
| 05.04 | tab:identifiability | both fail the f_10 >= 0.5 criterion | `True` | check |
| 05.05 | tab:identifiability | deutan bias on beta_s (+16, median over donors) | `16.0` | check |
| 05.06 | tab:identifiability | deutan bias on beta_c (-4.7, mean over donors; the donor median is -4.0) | `-4.7` | check |
| 05.07 | tab:identifiability | protan bias on beta_s (+11) | `11.0` | check |
| 05.08 | tab:identifiability | protan bias on beta_c (-27) | `-27.0` | check |
| 05.09 | S12 ¶4 | deutan dominant-axis bias 4.7 against 16 on the non-dominant axis | `True` | check |
| 05.10 | tab:identifiability | deutan median |beta_s| (IQR) | `(22.0, 40.0)` | check |
| 05.11 | tab:identifiability | deutan median |beta_c| (IQR) | `(26.0, 10.5)` | check |
| 05.12 | tab:identifiability | deutan f_origin | `0.0` | check |
| 05.13 | tab:identifiability | protan median |beta_s| (IQR) | `(16.0, 17.5)` | check |
| 05.14 | tab:identifiability | protan median |beta_c| (IQR) | `(24.0, 9.0)` | check |
| 05.15 | tab:identifiability | protan f_origin | `0.0` | check |
| 05.16 | S12 ¶4 | effective uncertainty about 20 degrees on beta_s and 25 on beta_c (means of the two medians) | `(19, 25)` | check |
| 05.17 | tab:identifiability | deutan Test 2b rank_dist | `0.875` | check |
| 05.18 | tab:identifiability | protan Test 2b rank_dist | `0.875` | check |
| 05.19 | tab:identifiability | deutan Test 2c real loss | `-2.892` | check |
| 05.20 | tab:identifiability | deutan Test 2c 5% cut | `-3.136` | check |
| 05.21 | tab:identifiability | deutan Test 2c p | `0.167` | check |
| 05.22 | tab:identifiability | protan Test 2c real loss | `-1.681` | check |
| 05.23 | tab:identifiability | protan Test 2c 5% cut | `-3.053` | check |
| 05.24 | tab:identifiability | protan Test 2c p | `0.471` | check |
| 05.25 | S12 ¶1 | N = 1,000 permutations | `1000` | check |
| 05.26 | S12 ¶1 | BH over six checks: none significant | `0` | check |
| 05.27 | S13 ¶1 | deutan ||beta|| | `42.4` | check |
| 05.28 | S13 ¶1 | deutan control range low | `30.5` | check |
| 05.29 | S13 ¶1 | deutan control range high | `58.1` | check |
| 05.30 | S13 ¶1 | deutan control mean | `49.1` | check |
| 05.31 | S13 ¶1 | protan ||beta|| | `24.1` | check |
| 05.32 | S13 ¶1 | protan control range low | `23.4` | check |
| 05.33 | S13 ¶1 | protan control range high | `55.5` | check |
| 05.34 | S13 ¶1 | protan control mean | `35.7` | check |
| 05.35 | S13 ¶1 | both estimates fall inside the control range | `True` | check |

## 06_filter — Per-participant stimulus-space filter (pre-image of the fitted distortion)

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 06.01 | Results §6 | deutan fitted (beta_s, beta_c) | `(6.0, -42.0)` | check |
| 06.02 | Results §6 | deutan theta_conf | `150.0` | check |
| 06.03 | Results §6 | deutan mean per-hue correction |delta theta| (degrees) | `26.3` | check |
| 06.04 | Results §6 | protan fitted (beta_s, beta_c) | `(2.0, 24.0)` | check |
| 06.05 | Results §6 | protan theta_conf | `16.0` | check |
| 06.06 | Results §6 | protan mean |delta theta| (degrees) | `16.2` | check |
| 06.07 | Methods filter | eight hues per participant | `(8, 8)` | check |
| 06.08 | Methods filter | every hue resolves to within 1e-3 degrees (deutan, recomputed) | `0.001` | check |
| 06.09 | Methods filter | every hue resolves to within 1e-3 degrees (protan, recomputed) | `0.001` | check |
| 06.10 | Methods filter | stored per-hue corrections equal pre-image minus stimulus angle | `1e-06` | check |

## 07_filter_evaluation — Second-session evaluation of the filters (psychophysics and fMRI)

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 07.01 | Results §7 | deutan mean |z| at baseline | `2.24` | check |
| 07.02 | Results §7 | deutan mean |z| under the deployed filter (below 0.9) | `0.9` | check |
| 07.03 | Results §7 | deutan mean |z| under the individualized filter (below 0.9) | `0.9` | check |
| 07.04 | Results §7 | deutan: all three elevated pairs within ±1.8 control SD under each filter | `1.8` | check |
| 07.05 | Results §7 | deutan Wilcoxon between the two filters p = 0.84 | `0.84` | check |
| 07.06 | Results §7 | protan mean |z| at baseline | `0.9` | check |
| 07.07 | Results §7 | protan mean |z| under the deployed filter | `1.78` | check |
| 07.08 | Results §7 | protan mean |z| under the individualized filter | `0.93` | check |
| 07.09 | Results §7 | deployed filter left two protan pairs deviant (|z| > 2) | `2` | check |
| 07.10 | Results §7 | individualized filter held every protan pair within ±1.5 | `1.5` | check |
| 07.11 | Results §7 | only the individualized filter kept every previously non-deviant pair inside the control range in both participants | `(True, False)` | check |
| 07.12 | Abstract | the four session-1 elevations (three deutan, one protan) moved inside the control range under the individualized filter | `2.0` | check |
| 07.13 | tab:exp2_8afc | 2 participants x 3 conditions x (accuracy, CI, n) | `24 cells, see 07.T1.*` | table |
| 07.14 | tab:exp2_loro | 4 ROIs x (controls + 2 participants x 3 conditions) | `28 cells, see 07.T2.*` | table |
| 07.15 | Results §7 | lowest cell | `0.5` | check |
| 07.16 | Results §7 | control range low | `0.71` | check |
| 07.17 | Results §7 | control range high | `0.77` | check |
| 07.18 | Results §7 | every cell above chance 0.125 | `0.125` | check |
| 07.19 | tab:exp2_geometry | adjacent accuracy, disparity and RDM similarity by condition with d_cc | `42 cells, see 07.T3.*` | table |
| 07.20 | Results §7 'Interpolation' | deutan: individualized filter is the only condition above chance 0.25 at hV4 | `['Ind']` | check |
| 07.21 | Results §7 'Interpolation' | protan: individualized filter is the lowest of the three, all below chance | `('Ind', True)` | check |
| 07.22 | Results §7 'Geometry' | deutan: both filters raise V2 disparity above baseline and lower RDM similarity | `True` | check |
| 07.23 | Results §7 'Geometry' | protan: both filters reduce V1 disparity; RDM similarity rises under Dep and falls under Ind | `(True, True, True)` | check |
| 07.24 | S15 'Forward-tuning' | deutan hV4 rho under the individualized filter | `0.18` | check |
| 07.25 | S15 | control rho at hV4 | `0.21` | check |
| 07.26 | S15 | deutan hV4 rho under the deployed filter | `-0.39` | check |
| 07.27 | S15 | protan: three conditions all near -0.02 (|rho + 0.02| < 0.01) | `0.01` | check |
| 07.28 | S14 'Run-count adequacy' | hV4 control mean at four runs | `0.45` | check |
| 07.29 | S14 | at six runs | `0.46` | check |
| 07.30 | S14 | deutan at four runs | `0.23` | check |
| 07.31 | S14 | protan at four runs | `0.14` | check |
| 07.32 | S14 | single-case d_cc < -2 in both at four runs | `-2.0` | check |
| 07.33 | S14 | the control-CVD separation holds at every run count down to four (control mean above both CVD cases and above chance) | `True` | check |
| 07.33b | S14 | both CVD participants below chance at four runs | `0.25` | check |
| 07.34 | S14 | protan V1 RDM similarity of the unfiltered baseline, unmatched | `0.25` | check |
| 07.35 | S14 | same after run matching | `0.33` | check |
| 07.36 | S14 | four runs per filter condition | `(4, 4)` | check |

## 08_sensitivity — Preprocessing sensitivity: the head-motion-corrected pipeline

| id | manuscript | quantity | reported | kind |
|---|---|---|---|---|
| 08.01 | tab:motion_arms | 8 cells x (CH t, p; LOSO t, p) | `32 cells, see 08.T1.*` | table |
| 08.02 | S2 'Disparity under the two pipelines' | protan V1 remains the largest deviation under both pipelines | `('V1', 'V1')` | check |
| 08.03 | S2 | protan V1 weakens from p = .007 | `0.007` | check |
| 08.04 | S2 | to p = .077 | `0.077` | check |
| 08.05 | S2 | deutan largest deviation moves from V2 to V1 | `('V2', 'V1')` | check |
| 08.06 | S2 | deutan V2 t reverses in sign (+2.1 -> -1.0) | `(2.1, -1.0)` | check |
| 08.07 | S2 | under LOSO the primary-pipeline protan V1 cell alone reaches significance across both pipelines | `[('with_residuals', 'sub-09', 'V1')]` | check |
| 08.08 | S2 | primary-pipeline table reproduces tab:disparity_loso (deutan V2 .040 / .116; protan V1 .007 / .045) | `(0.04, 0.116, 0.007, 0.045)` | check |
| 08.09 | tab:interp_arms | interpolation and classification cells under both pipelines | `34 cells, see 08.T2.*` | table |
| 08.10 | S2 'Classification and interpolation' | hV4 alone passes the control gate in both pipelines | `True` | check |
| 08.11 | S2 | both participants remain below the control mean in both pipelines | `True` | check |
| 08.12 | S2 | lowest CVD classification cell, primary | `0.375` | check |
| 08.13 | S2 | lowest CVD classification cell, head-motion correction | `0.229` | check |
| 08.14 | S2 | which is 1.8 times chance | `1.8` | check |
| 08.15 | S2 | every single-case classification contrast stays above p = .10 in both pipelines | `0.1` | check |
| 08.16 | S2 | single-case interpolation contrasts reach significance under the primary pipeline alone | `(True, True)` | check |
| 08.17 | tab:interp_arms caption | permutation null mean is 0.35 in both pipelines (all ROIs within 0.34-0.36) | `(0.34, 0.36)` | check |
| 08.18 | tab:color_specificity | 35 cells, head-motion-corrected pipeline | `35 cells, see 08.T3.*` | table |
| 08.19 | S18 ¶3 | cells below .05 under head-motion correction | `15` | check |
| 08.20 | S18 ¶3 | none survive BH correction | `0` | check |
| 08.21 | S18 ¶3 | deutan V2 nominal p .003 | `0.003` | check |
| 08.22 | S18 ¶3 | deutan V2 corrected q .053 | `0.053` | check |
| 08.23 | S18 ¶3 | protan V1 moves to p .010 | `0.01` | check |
| 08.24 | S2 'Session-2 endpoints' | fourteen pre-specified endpoints | `14` | check |
| 08.25 | S2 | control reference hV4 adjacent accuracy, primary arm | `0.456` | check |
| 08.26 | S2 | harmonised arm | `0.445` | check |
| 08.27 | S2 | native-mask directional contrasts reversed between arms | `(8, 10)` | check |
| 08.28 | S2 | run-matched-mask contrasts reversed | `(5, 10)` | check |
| 08.29 | S2 | the contrasts at the pre-specified target regions held in both arms | `(True, True)` | check |
