#!/usr/bin/env python3
"""Motion QC summary from MCFLIRT .par files (COBIDAS `Quality control reports`).

Background
----------
COBIDAS requires "summaries of subject motion (e.g. mean framewise
displacement)". The preprocessing pipeline
(`run_method3_header_mi{,_2nd}.sbatch`) applies NO realignment: raw BOLD goes
straight to `applywarp`. Motion is estimated separately by
`add_motion_correction{,_2nd}.sbatch`, which writes 6-column MCFLIRT
parameters and discards the realigned 4D. Those `.par` files are the only
motion record, so this script summarizes them.

The `*_desc-confounds_timeseries.tsv` files in the same directory are NOT
usable: their trans/rot columns hold constant header-derived values and
`framewise_displacement` is identically zero (verified 2026-08-03). They were
never regressed, so results are unaffected — but do not compute anything from
them.

FD definition
-------------
Power et al. (2012): FD_t = sum |d trans| + sum |d rot| * r, with rotations
converted to displacement on a sphere of radius r = 50 mm. MCFLIRT `.par`
column order is rot_x rot_y rot_z (rad), trans_x trans_y trans_z (mm)
(see `add_motion_correction.sbatch`, which reads $1..$3 as rotation and
$4..$6 as translation).

Input
-----
<deriv>/sub-XX/func/sub-XX_task-rsvp_run-N_desc-motion.par
  exp1: /storage/connectome/haba6030/fmriprep_out_method3_header_mi
  exp2: /storage/connectome/haba6030/fmriprep_out_method3_2nd

Output
------
results/motion_qc_summary.json  (next to this script's phase folder)

Usage
-----
    # on the server (node1/node2), after add_motion_correction_2nd.sbatch
    python motion_qc_summary.py \
        --exp1 /storage/connectome/haba6030/fmriprep_out_method3_header_mi \
        --exp2 /storage/connectome/haba6030/fmriprep_out_method3_2nd \
        --out  motion_qc_summary.json
"""

# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/motion_qc_summary.py  (development repository, commit 53c81c2)
# Role      : Framewise displacement (Power 2012, r = 50 mm) per run / participant -> motion_qc_summary.json.
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


import argparse
import glob
import json
import os
import re

import numpy as np

ROTATION_RADIUS_MM = 50.0
HC_IDS = [f'sub-0{i}' for i in range(1, 8)]
CVD_IDS = ['sub-08', 'sub-09']
EXCLUDED_IDS = ['sub-10']          # near-normal, excluded from all analyses


def fd_from_par(path):
    """Return (per-volume FD array, n_volumes). None if the file is malformed."""
    a = np.loadtxt(path)
    if a.ndim != 2 or a.shape[1] < 6:
        return None, None
    rot, trans = a[:, :3], a[:, 3:6]
    disp = np.hstack([trans, rot * ROTATION_RADIUS_MM])
    return np.abs(np.diff(disp, axis=0)).sum(axis=1), a.shape[0]


def summarize_session(deriv_dir):
    subjects = {}
    for sdir in sorted(glob.glob(os.path.join(deriv_dir, 'sub-*'))):
        sid = os.path.basename(sdir)
        runs = {}
        for f in sorted(glob.glob(os.path.join(sdir, 'func', '*_desc-motion.par'))):
            m = re.search(r'run-(\d+)', os.path.basename(f))
            fd, nvol = fd_from_par(f)
            if fd is None:
                continue
            runs[f'run-{m.group(1)}' if m else os.path.basename(f)] = {
                'mean_fd_mm': float(fd.mean()),
                'max_fd_mm': float(fd.max()),
                'n_volumes': int(nvol),
                'n_fd_gt_0p5': int((fd > 0.5).sum()),
            }
        if runs:
            per_run = [r['mean_fd_mm'] for r in runs.values()]
            subjects[sid] = {
                'n_runs': len(runs),
                'mean_fd_mm': float(np.mean(per_run)),
                'worst_run_mean_fd_mm': float(np.max(per_run)),
                'max_fd_mm': float(max(r['max_fd_mm'] for r in runs.values())),
                'runs': runs,
            }
    return subjects


def group_stats(subjects, ids):
    vals = [subjects[s]['mean_fd_mm'] for s in ids if s in subjects]
    if not vals:
        return None
    return {'n': len(vals), 'mean_fd_mm': float(np.mean(vals)),
            'sd': float(np.std(vals, ddof=0)),
            'min': float(np.min(vals)), 'max': float(np.max(vals))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp1', default='/storage/connectome/haba6030/fmriprep_out_method3_header_mi')
    p.add_argument('--exp2', default='/storage/connectome/haba6030/fmriprep_out_method3_2nd')
    p.add_argument('--out', default='motion_qc_summary.json')
    args = p.parse_args()

    out = {
        'analysis': 'motion_qc_summary',
        'fd_definition': 'Power et al. 2012; rotations converted at r = 50 mm',
        'par_column_order': 'rot_x rot_y rot_z (rad), trans_x trans_y trans_z (mm)',
        'note_realignment': ('no realignment applied to analysed data; mcflirt run '
                             'only to write motion parameters'),
        'note_confounds_tsv': ('*_desc-confounds_timeseries.tsv in the same folders '
                               'are placeholders (constant trans/rot, FD = 0) — do '
                               'not use'),
        'sessions': {},
    }

    for name, d in (('exp1', args.exp1), ('exp2', args.exp2)):
        subjects = summarize_session(d)
        out['sessions'][name] = {
            'derivatives_dir': d,
            'n_subjects_with_par': len(subjects),
            'groups': {
                'all_analysed': group_stats(subjects, HC_IDS + CVD_IDS),
                'HC': group_stats(subjects, HC_IDS),
                'CVD': group_stats(subjects, CVD_IDS),
                'excluded_sub10': group_stats(subjects, EXCLUDED_IDS),
            },
            'subjects': subjects,
        }

    with open(args.out, 'w') as f:
        json.dump(out, f, indent=2)

    for name in ('exp1', 'exp2'):
        s = out['sessions'][name]
        print(f'\n=== {name} ({s["n_subjects_with_par"]} subjects with .par) ===')
        if not s['n_subjects_with_par']:
            print('  no .par files found')
            continue
        for sid, v in s['subjects'].items():
            print(f'  {sid}: mean FD {v["mean_fd_mm"]:.4f} mm  '
                  f'(worst run {v["worst_run_mean_fd_mm"]:.4f}, '
                  f'peak {v["max_fd_mm"]:.4f}, {v["n_runs"]} runs)')
        for g, st in s['groups'].items():
            if st:
                print(f'  [{g}] n={st["n"]}  {st["mean_fd_mm"]:.4f} +- {st["sd"]:.4f} mm '
                      f'({st["min"]:.4f}-{st["max"]:.4f})')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
