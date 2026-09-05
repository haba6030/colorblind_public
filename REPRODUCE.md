# Reproducing the reported values

## What the notebooks do

Each stage notebook (`<stage>/<stage>.ipynb`) loads the committed result files in `<stage>/results/` and compares the values it derives with the numbers printed in the manuscript. A comparison passes when the produced value equals the printed value to the printed precision (half a unit of the last printed digit) or satisfies the stated relation (above chance, below a threshold, and so on). The notebooks recompute aggregate statistics where the committed files hold the per-participant inputs: Crawford-Howell single-case tests, Mann-Whitney and Spearman statistics, Hedges' g, Benjamini-Hochberg correction, Wilson intervals, resample tallies, and the pre-image residual of the filter. They do not recompute anything from imaging or trial-level behavioural data, which are not distributed.

`run_notebooks.py` executes the notebooks in place and writes `REPORT.md`. `MANIFEST.md` is the list of printed values with their check ids; `MAP.md` links each id to the result file and the producing script.

## Environment

`environment.yml` recreates the environment the analyses were run in locally (Python 3.9, numpy 1.26, scipy 1.13, scikit-learn 1.6, BrainIAK 0.12 through pip; the notebooks themselves need only numpy and scipy). The Methods section of the manuscript names the versions of the original server runs (Python 3.10, numpy 1.24.3, scipy 1.11.3, scikit-learn 1.3.0, BrainIAK 0.11); the committed result files were produced under those. Random seeds are recorded inside the result files (`seed` fields) and in the scripts.

## What cannot be re-run here, and how it is handled

| Item | Why | Handling |
|---|---|---|
| Amplitude estimation, Procrustes alignment, SRM, disparity (`00`, `02`) | needs the BOLD data; SRM needs BrainIAK with `mpirun -np 1` | notebook verifies the committed `loo_consistent_results.json` and the k-selection output |
| Six-decoder comparison, LOCO / LORO readouts (`01`) | needs the amplitude arrays | committed per-fold outputs and the exported per-hue adjacent-accuracy file are verified; the colour-label permutation null arrays are committed |
| Model fitting over N = 300 resamples, identifiability synthesis (`04`, `05`) | SLURM runs of hours | committed `summary` blocks (verbatim) and the per-resample fits of the reported combinations are verified |
| Second-session neural indices (`07`) | needs the session-2 BOLD data and BrainIAK | committed condition-level outputs are verified |
| Field-map displacement, registration run-to-run displacement, second-order RSA cyclic shift | computed on the server; no committed artifact | listed as pointers in the notebooks and in `REPORT.md` |
| Crossnobis permutation p at n = 2 | the committed permutation includes sub-10 | the observed difference is recomputed without sub-10; the permutation p is a pointer |

## Scripts

Scripts are the code that produced the committed results, copied from the development repository with a provenance header (original path, stage, role). Their data paths follow the development layout and they are not runnable without the data; they are included so that every reported quantity can be traced to the code that computed it. Every `.py` file compiles (`BUILD_LOG.md`).

## Re-executing

```bash
python run_notebooks.py          # all nine stages -> REPORT.md
python run_notebooks.py 04 05    # selected stages
```
