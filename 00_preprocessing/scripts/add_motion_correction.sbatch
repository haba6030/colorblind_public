#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/add_motion_correction.sbatch  (development repository, commit 53c81c2)
# Role      : Writes motion parameters for the QC summary.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=add_mcflirt
#SBATCH --array=1-60%20
#SBATCH --output=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/mcflirt_%A_%a.out
#SBATCH --error=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/mcflirt_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --qos=shared
#SBATCH --nodelist=node2
#SBATCH --no-requeue

################################################################################
# Add Motion Correction to Method3 Results
#
# This script adds mcflirt motion correction to existing method3_header_mi
# preprocessed data, generating .par files needed for confounds generation
#
# Input:  method3 BOLD files (already in MNI space)
# Output: Motion parameters (.par files)
################################################################################

set -e

# Subject-run mapping (10 subjects × 6 runs = 60 jobs)
SUBJECTS=(01 02 03 04 05 06 07 08 09 10)
RUNS=(1 2 3 4 5 6)

# Calculate subject and run from array task ID
SUBJECT_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) / 6 ))
RUN_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) % 6 ))

SUBJECT_ID=${SUBJECTS[$SUBJECT_IDX]}
RUN=${RUNS[$RUN_IDX]}

echo "================================================================================"
echo "Adding Motion Correction: sub-${SUBJECT_ID} run-${RUN}"
echo "================================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""

# Directories
BIDS_DIR="/storage/connectome/haba6030/bids_editted"
METHOD3_DIR="/storage/connectome/haba6030/fmriprep_out_method3_header_mi"
WORK_DIR="/scratch/connectome/haba6030/colorBlind/work/mcflirt_sub-${SUBJECT_ID}"

mkdir -p ${WORK_DIR}
mkdir -p /scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs

# FSL setup
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
    echo "❌ ERROR: mcflirt not found"
    exit 1
fi

echo "✅ FSL found: $(which mcflirt)"
echo ""

# Input files
BOLD_ORIG="${BIDS_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_bold.nii.gz"
BOLD_MNI="${METHOD3_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"

if [ ! -f "${BOLD_ORIG}" ]; then
    echo "⚠️  Original BOLD not found, skipping: ${BOLD_ORIG}"
    exit 0
fi

if [ ! -f "${BOLD_MNI}" ]; then
    echo "⚠️  MNI BOLD not found, skipping: ${BOLD_MNI}"
    exit 0
fi

# Output files
MOTION_PAR="${METHOD3_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_desc-motion.par"
BOLD_MC="${WORK_DIR}/sub-${SUBJECT_ID}_run-${RUN}_bold_mc.nii.gz"

# Check if already processed
if [ -f "${MOTION_PAR}" ]; then
    echo "✅ Motion parameters already exist, skipping: ${MOTION_PAR}"
    exit 0
fi

echo "Processing files:"
echo "  Input BOLD:  ${BOLD_ORIG}"
echo "  Output .par: ${MOTION_PAR}"
echo ""

# =========================================================================
# Run mcflirt on ORIGINAL BOLD (before normalization)
# =========================================================================

echo "Running mcflirt on original BOLD..."
echo ""

# Get middle volume as reference
NVOLS=$(fslinfo ${BOLD_ORIG} | grep "^dim4" | awk '{print $2}')
REF_VOL=$((NVOLS / 2))

echo "  Total volumes: ${NVOLS}"
echo "  Reference volume: ${REF_VOL}"
echo ""

# Run mcflirt
# -plots: Generate .par file with motion parameters
# -refvol: Use middle volume as reference
# -mats: Save transformation matrices (optional, for debugging)
mcflirt \
    -in ${BOLD_ORIG} \
    -out ${BOLD_MC} \
    -refvol ${REF_VOL} \
    -plots \
    -verbose 1

# mcflirt creates motion.par in the same directory as output
# Move it to method3 output directory
if [ -f "${BOLD_MC}.par" ]; then
    cp ${BOLD_MC}.par ${MOTION_PAR}
    echo "✅ Motion parameters saved: ${MOTION_PAR}"

    # Show summary
    echo ""
    echo "Motion summary:"
    echo "  Format: rotation(rad) + translation(mm), 6 columns"
    echo "  Timepoints: $(wc -l < ${MOTION_PAR})"

    # Compute mean absolute motion
    MEAN_TRANS=$(awk '{sum+=sqrt($4*$4+$5*$5+$6*$6)} END {print sum/NR}' ${MOTION_PAR})
    MEAN_ROT=$(awk '{sum+=sqrt($1*$1+$2*$2+$3*$3)} END {print sum/NR}' ${MOTION_PAR})

    echo "  Mean translation: ${MEAN_TRANS} mm"
    echo "  Mean rotation: ${MEAN_ROT} rad"
else
    echo "❌ ERROR: Motion parameters not created"
    exit 1
fi

# Cleanup motion-corrected BOLD (we only need .par file)
echo ""
echo "Cleaning up intermediate files..."
rm -f ${BOLD_MC} ${BOLD_MC}.mat

echo ""
echo "================================================================================"
echo "✅ Motion correction complete: sub-${SUBJECT_ID} run-${RUN}"
echo "================================================================================"
echo "End time: $(date)"
echo ""

exit 0
