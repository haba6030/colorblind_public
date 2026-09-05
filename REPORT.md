# Reproduction report

Executed 2026-09-04 17:29 UTC by `run_notebooks.py` (Python 3.9.23).

Legend: **ok** reproduces the printed value to its precision (half a unit of the last printed digit) or satisfies the stated relation; **near** lies within one unit of the last printed digit, which points to rounding of an intermediate value rather than a different result; **mismatch** differs by more; **error** the check raised; **pointer** the manuscript value has no committed artifact and is listed rather than verified.

**Headline: 1046/1058 numeric checks reproduced exactly; 11 within one unit of the last printed digit; 1 mismatch, 0 error; 8 pointer-only.**

| Stage | reproduced | near | mismatch | error | pointer | run |
|---|---|---|---|---|---|---|
| [00_preprocessing](00_preprocessing/00_preprocessing.ipynb) | 14/14 | 0 | 0 | 0 | 3 | completed |
| [01_decoding](01_decoding/01_decoding.ipynb) | 285/289 | 3 | 1 | 0 | 0 | completed |
| [02_geometry](02_geometry/02_geometry.ipynb) | 187/191 | 4 | 0 | 0 | 4 | completed |
| [03_psychophysics](03_psychophysics/03_psychophysics.ipynb) | 175/177 | 2 | 0 | 0 | 0 | completed |
| [04_distortion_model](04_distortion_model/04_distortion_model.ipynb) | 86/86 | 0 | 0 | 0 | 1 | completed |
| [05_identifiability](05_identifiability/05_identifiability.ipynb) | 35/35 | 0 | 0 | 0 | 0 | completed |
| [06_filter](06_filter/06_filter.ipynb) | 10/10 | 0 | 0 | 0 | 0 | completed |
| [07_filter_evaluation](07_filter_evaluation/07_filter_evaluation.ipynb) | 127/128 | 1 | 0 | 0 | 0 | completed |
| [08_sensitivity](08_sensitivity/08_sensitivity.ipynb) | 127/128 | 1 | 0 | 0 | 0 | completed |

## Items that did not reproduce cleanly

| Stage | id | status | description | reported | produced | note |
|---|---|---|---|---|---|---|
| 00_preprocessing | 00.15 | flag | S2 'Susceptibility distortion' | field-map displacement within ROIs 0.01-0.76 voxels (0.02-1.52 mm); within-ROI variation 0.05-0.21 voxels (0.38 in one participant) |  |  | no committed artifact (server field-map computation) |
| 00_preprocessing | 00.16 | flag | S2 'Registration' | run-to-run displacement of the MI solution 0.9-4.2 mm; defacing moved it by 1.9 mm (deutan) and 9.4 mm (protan) |  |  | no committed artifact (server registration logs) |
| 00_preprocessing | 00.17 | flag | S2 'Registration' | BBR snapped the slab about 10 mm off on visual inspection; MI within about 1 mm |  |  | visual criterion; bbr_vs_mi_displacement.json gives centroid displacement for three participants (descriptive) |
| 01_decoding | 01.38 | near | Results §1 ¶4 | protan p (one-tailed) | 0.012 | 0.0114619638613391 |  |
| 01_decoding | 01.T3.V2.MLP.sd | near | tab:loro_decoders V2 MLP control SD | 0.031 | 0.030496877279641663 |  |
| 01_decoding | 01.T3.hV4.Ridge.mean | near | tab:loro_decoders hV4 Ridge control mean | 0.319 | 0.31845238095238093 |  |
| 01_decoding | 01.62 | mismatch | S5 ¶2 | four of the six decoders exceed chance at every ROI (by control mean > 0.125 the count is five, kernel ridge at V3 being 0.158 ± 0.052; by control mean minus one SD it is three: the printed 'four' matches neither criterion) | 4 | 5 |  |
| 02_geometry | 02.T1.deutan.V1.d_loso | near | tab:disparity_loso deutan V1 LOSO d_cc | 0.51 | 0.5163027567475412 |  |
| 02_geometry | 02.T1.protan.V2.d | near | tab:disparity_loso protan V2 common d_cc | 1.06 | 1.0540003518429903 |  |
| 02_geometry | 02.T1.protan.V2.d_loso | near | tab:disparity_loso protan V2 LOSO d_cc | 0.82 | 0.8262669686977658 |  |
| 02_geometry | 02.T1.protan.hV4.d_loso | near | tab:disparity_loso protan hV4 LOSO d_cc | 0.86 | 0.8517031799118711 |  |
| 02_geometry | 02.23 | flag | S9 ¶2 | permutation p = .120 for the V1 difference |  |  | the committed permutation was run with sub-10 included (p = .051); the n = 2 permutation needs the per-participant crossnobis RDMs, which are not stored |
| 02_geometry | 02.38 | flag | S10 ¶3 | activation metrics vs disparity: 20 tests, r from -0.29 to +0.51, all p >= 0.130 | [20, -0.29, 0.51, 0.13] |  | activation_prior_results.json does not store the metric-disparity correlations in a recognised layout |
| 02_geometry | 02.52 | flag | S18 ¶4 | eight-way classification 0.79 at protan V1 |  |  | readout not stored in a committed artifact (the frozen-permutation driver printed it to stdout) |
| 02_geometry | 02.64 | flag | S18 ¶4 | second-order RSA rho 0.00 -> +0.52 (controls 0.45), z = +5.02, p = .002, p_adj = .032; SRM 225-degree shift +0.50; PCA optimum at 135 degrees p = .048 |  |  | no committed artifact for the second-order RSA cyclic-shift statistics |
| 03_psychophysics | 03.T1.yellow-green.sd | near | tab:jnd_baseline yellow-green control SD | 0.044 | 0.04349783382674415 |  |
| 03_psychophysics | 03.T1.cyan-magenta.sd | near | tab:jnd_baseline cyan-magenta control SD | 0.017 | 0.01648411934184506 |  |
| 04_distortion_model | 04.87 | flag | Figure S1 | loss landscape reconstruction on the seven-control pool |  |  | figure committed; regeneration requires the amplitude arrays (scripts/viz_closure_ground_plot.py) |
| 07_filter_evaluation | 07.T3.deutan.hV4.Dep.d | near | tab:exp2_geometry deutan hV4 adjacent Dep d_cc | -1.94 | -1.9349133095839817 |  |
| 08_sensitivity | 08.T2.protan.hV4.primary.p | near | tab:interp_arms protan hV4 primary p | 0.012 | 0.0114619638613391 |  |
