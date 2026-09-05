#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/add_motion_correction_2nd.sbatch  (development repository, commit 53c81c2)
# Role      : Same for session 2.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=add_mcflirt_2nd
#SBATCH --array=1-16%8
#SBATCH --output=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/mcflirt_2nd_%A_%a.out
#SBATCH --error=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/mcflirt_2nd_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --nodelist=node2
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/connectome/haba6030/colorBlind

################################################################################
# Motion parameter estimation for the 2nd MRI dataset (exp2, filter validation)
#
# Counterpart of add_motion_correction.sbatch (1st dataset). Same procedure,
# same output naming, so exp1 and exp2 motion QC are directly comparable.
#
# WHY: the exp2 preprocessing run (run_method3_header_mi_2nd.sbatch) produced
# no motion estimates — fmriprep_out_method3_2nd/ and
# fmriprep_work_method3_2nd_sub-{08,09}/ contain no .par files. exp1 has them
# for all 10 subjects x 6 runs. COBIDAS requires a motion QC summary, so exp2
# needs the same estimates before its results can be reported.
#
# WHAT THIS DOES NOT DO: no realignment is applied to any analysed data. As in
# the 1st dataset, mcflirt is run only to WRITE OUT the 6 motion parameters;
# the realigned 4D is deleted. Preprocessed BOLD files are untouched.
#
# Input:  bids_2nd raw BOLD (before normalization) - 2 subjects x 8 runs
# Output: <method3_2nd>/sub-XX/func/sub-XX_task-rsvp_run-N_desc-motion.par
#
# NOTE ON SLURM FLAGS: the 1st-dataset script used "--qos=shared". Project
# CLAUDE.md forbids --qos/--partition, so it is omitted here. If the scheduler
# rejects the job, add "#SBATCH --qos=shared" back to match the reference run.
################################################################################

set -e

# Subject-run mapping (2 subjects x 8 runs = 16 jobs)
SUBJECTS=(08 09)
RUNS=(1 2 3 4 5 6 7 8)

SUBJECT_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 8 ))
RUN_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 8 ))
SUBJECT_ID=${SUBJECTS[$SUBJECT_IDX]}
RUN=${RUNS[$RUN_IDX]}

echo "================================================================================"
echo "mcflirt (2nd dataset) - sub-${SUBJECT_ID} run-${RUN}"
echo "================================================================================"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}   Node: $(hostname)"
echo "Start: $(date)"
echo ""

# Directories (2nd MRI dataset)
BIDS_DIR="/storage/connectome/haba6030/bids_2nd"
METHOD3_DIR="/storage/connectome/haba6030/fmriprep_out_method3_2nd"
WORK_DIR="/scratch/connectome/haba6030/colorBlind/work/mcflirt_2nd_sub-${SUBJECT_ID}"

mkdir -p ${WORK_DIR}
mkdir -p /scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs

# FSL setup (identical to 1st-dataset script)
if ! command -v mcflirt &> /dev/null; then
    for FSL_PATH in /usr/local/fsl /opt/fsl; do
        if [ -d "${FSL_PATH}/bin" ]; then
            export FSLDIR=${FSL_PATH}
            export PATH="${FSLDIR}/bin:${PATH}"
            if [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ]; then
                source ${FSLDIR}/etc/fslconf/fsl.sh
            fi
            break
        fi
    done
fi

if ! command -v mcflirt &> /dev/null; then
    echo "ERROR: mcflirt not found"
    exit 1
fi
echo "FSL found: $(which mcflirt)"
echo ""

# Input / output
BOLD_ORIG="${BIDS_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_bold.nii.gz"
BOLD_MNI="${METHOD3_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
MOTION_PAR="${METHOD3_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_desc-motion.par"
BOLD_MC="${WORK_DIR}/sub-${SUBJECT_ID}_run-${RUN}_bold_mc.nii.gz"

if [ ! -f "${BOLD_ORIG}" ]; then
    echo "WARNING: raw BOLD not found, skipping: ${BOLD_ORIG}"
    exit 0
fi
if [ ! -f "${BOLD_MNI}" ]; then
    echo "WARNING: preprocessed BOLD not found (run not preprocessed?), skipping: ${BOLD_MNI}"
    exit 0
fi
if [ -f "${MOTION_PAR}" ]; then
    echo "Motion parameters already exist, skipping: ${MOTION_PAR}"
    exit 0
fi

echo "  Input BOLD:  ${BOLD_ORIG}"
echo "  Output .par: ${MOTION_PAR}"
echo ""

# Reference = middle volume (matches the coregistration step of
# run_method3_header_mi_2nd.sbatch and the 1st-dataset motion script)
NVOLS=$(fslinfo ${BOLD_ORIG} | grep "^dim4" | awk '{print $2}')
REF_VOL=$((NVOLS / 2))
echo "  Total volumes: ${NVOLS}"
echo "  Reference volume: ${REF_VOL}"
echo ""

# -plots : write the 6-column .par (rot_x rot_y rot_z [rad], trans_x/y/z [mm])
mcflirt \
    -in ${BOLD_ORIG} \
    -out ${BOLD_MC} \
    -refvol ${REF_VOL} \
    -plots \
    -verbose 1

if [ -f "${BOLD_MC}.par" ]; then
    cp ${BOLD_MC}.par ${MOTION_PAR}
    echo "Motion parameters saved: ${MOTION_PAR}"
    echo ""
    echo "Motion summary:"
    echo "  Format: rotation(rad) cols 1-3 + translation(mm) cols 4-6"
    echo "  Timepoints: $(wc -l < ${MOTION_PAR})"
    MEAN_TRANS=$(awk '{sum+=sqrt($4*$4+$5*$5+$6*$6)} END {print sum/NR}' ${MOTION_PAR})
    MEAN_ROT=$(awk '{sum+=sqrt($1*$1+$2*$2+$3*$3)} END {print sum/NR}' ${MOTION_PAR})
    echo "  Mean translation: ${MEAN_TRANS} mm"
    echo "  Mean rotation:    ${MEAN_ROT} rad"
else
    echo "ERROR: .par not created"
    exit 1
fi

# Keep only the .par; the realigned 4D is not used anywhere
echo ""
echo "Cleaning up intermediate files..."
rm -f ${BOLD_MC} ${BOLD_MC}.mat

echo ""
echo "Done: sub-${SUBJECT_ID} run-${RUN}   End: $(date)"
echo "================================================================================"

exit 0
