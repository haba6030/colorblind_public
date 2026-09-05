#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/run_method3_hmc_2nd.sbatch  (development repository, commit 53c81c2)
# Role      : MCFLIRT motion-parameter estimation (session 2).
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=prep_m3hmc2
#SBATCH --array=1-16%8
#SBATCH --output=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/m3hmc2nd_%A_%a.out
#SBATCH --error=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/m3hmc2nd_%A_%a.err
#SBATCH --chdir=/scratch/connectome/haba6030/colorBlind
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --nodelist=node2
#SBATCH --no-requeue
#SBATCH --open-mode=append

################################################################################
# Method 3 + HMC (exp2, filter-evaluation session) — candidate A.
#
# WHAT CHANGES vs the reported pipeline
#   Reported (`run_method3_header_mi_2nd.sbatch`) feeds the RAW BIDS
#   BOLD to applywarp, so the analysed images carry no head-motion realignment.
#   This script realigns first (MCFLIRT) and warps the realigned series.
#   Registration strategy is IDENTICAL, so the two arms differ in HMC alone.
#
# WHY NO RE-COREGISTRATION
#   BOLD->T1w (per run, .lta) and T1w->MNI (affine + warp) are already stored in
#   <reported>/sub-XX/transforms/. MCFLIRT realigns to the run's middle volume,
#   which is the same volume the stored .lta was derived from, so the stored
#   transforms remain exactly valid for the realigned series. bet2, mri_coreg,
#   FLIRT and FNIRT are therefore not repeated.
#
# OUTPUT IS A NEW TREE. The reported derivatives are never touched; they remain
# the sensitivity arm.
#
# Array: 16 tasks = 2 subjects x 8 runs.
################################################################################

set -e

SUBJECTS=(08 09)
N_RUNS=8

TASK=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT_ID=${SUBJECTS[$((TASK / N_RUNS))]}
RUN=$(( (TASK % N_RUNS) + 1 ))

echo "================================================================================"
echo "Method 3 + HMC (exp2) — sub-${SUBJECT_ID} run-${RUN}"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}   Node: ${SLURM_NODELIST}"
echo "Start: $(date)"
echo "================================================================================"

# ---------------------------------------------------------------- directories
BIDS_DIR="/storage/connectome/haba6030/bids_2nd"
SRC_DIR="/storage/connectome/haba6030/fmriprep_out_method3_2nd"     # transforms live here
OUT_DIR="/storage/connectome/haba6030/fmriprep_out_method3_2nd_hmc"           # new tree
M3_WORK="/storage/connectome/haba6030/fmriprep_work_method3_2nd_sub-${SUBJECT_ID}"
WORK_DIR="/scratch/connectome/haba6030/colorBlind/work/m3hmc2nd_sub-${SUBJECT_ID}_run-${RUN}"
MNI_TEMPLATE="/scratch/connectome/haba6030/colorBlind/templates/MNI152NLin2009cAsym_res-2_T1_brain.nii.gz"

mkdir -p ${OUT_DIR}/sub-${SUBJECT_ID}/func
mkdir -p ${WORK_DIR}
mkdir -p /scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs

# ------------------------------------------------------------------ FSL setup
if ! command -v mcflirt &> /dev/null; then
    for FSL_PATH in /usr/local/fsl /opt/fsl; do
        if [ -d "${FSL_PATH}/bin" ]; then
            export FSLDIR=${FSL_PATH}
            export PATH="${FSLDIR}/bin:${PATH}"
            [ -f "${FSLDIR}/etc/fslconf/fsl.sh" ] && source ${FSLDIR}/etc/fslconf/fsl.sh
            break
        fi
    done
fi
command -v mcflirt &> /dev/null || { echo "ERROR: mcflirt not found"; exit 1; }
command -v applywarp &> /dev/null || { echo "ERROR: applywarp not found"; exit 1; }
echo "FSL: $(which mcflirt)"

# ----------------------------------------------------------- FreeSurfer setup
if ! command -v tkregister2 &> /dev/null; then
    for FS_PATH in /usr/local/freesurfer /opt/freesurfer /usr/local/freesurfer/7.4.1; do
        if [ -d "${FS_PATH}/bin" ]; then
            export FREESURFER_HOME=${FS_PATH}
            export PATH="${FREESURFER_HOME}/bin:${PATH}"
            [ -f "${FREESURFER_HOME}/SetUpFreeSurfer.sh" ] && source ${FREESURFER_HOME}/SetUpFreeSurfer.sh
            break
        fi
    done
fi
command -v tkregister2 &> /dev/null || { echo "ERROR: tkregister2 not found"; exit 1; }
export FS_LICENSE=${FREESURFER_HOME}/license.txt
echo "FreeSurfer: $(which tkregister2)"

# ----------------------------------------------------------------- input files
BOLD_RAW="${BIDS_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_bold.nii.gz"
LTA="${SRC_DIR}/sub-${SUBJECT_ID}/transforms/sub-${SUBJECT_ID}_run-${RUN}_bold_to_t1w.lta"
WARP="${SRC_DIR}/sub-${SUBJECT_ID}/transforms/sub-${SUBJECT_ID}_t1w_to_mni_warp.nii.gz"
T1W_BRAIN="${M3_WORK}/sub-${SUBJECT_ID}_T1w_brain.nii.gz"

for f in "${BOLD_RAW}" "${LTA}" "${WARP}" "${T1W_BRAIN}" "${MNI_TEMPLATE}"; do
    [ -f "$f" ] || { echo "ERROR: missing input $f"; exit 1; }
done

# ---------------------------------------------------------------- output files
BOLD_MNI="${OUT_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
MASK_MNI="${OUT_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"
MOTION_PAR="${OUT_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_desc-motion.par"

if [ -f "${BOLD_MNI}" ]; then
    echo "Already present, skipping: ${BOLD_MNI}"
    exit 0
fi

BOLD_MC="${WORK_DIR}/bold_mc.nii.gz"
BOLD_REF="${WORK_DIR}/bold_mc_ref.nii.gz"
BOLD_TO_T1W_MAT="${WORK_DIR}/bold_to_t1w.mat"

# ============================================================ Step 1: MCFLIRT
NVOLS=$(fslinfo ${BOLD_RAW} | grep "^dim4" | awk '{print $2}')
REF_VOL=$((NVOLS / 2))
echo "Step 1: MCFLIRT  (${NVOLS} volumes, reference volume ${REF_VOL})"

mcflirt -in ${BOLD_RAW} -out ${WORK_DIR}/bold_mc -refvol ${REF_VOL} -plots

[ -f "${WORK_DIR}/bold_mc.par" ] && cp ${WORK_DIR}/bold_mc.par ${MOTION_PAR}
[ -f "${BOLD_MC}" ] || { echo "ERROR: MCFLIRT produced no output"; exit 1; }

# ==================================== Step 2: reference frame of realigned run
echo "Step 2: extract reference frame from the realigned series"
fslroi ${BOLD_MC} ${BOLD_REF} ${REF_VOL} 1

# ============================ Step 3: stored LTA -> FSL matrix (no new coreg)
echo "Step 3: convert stored BOLD->T1w LTA to FSL format"
tkregister2 \
    --mov ${BOLD_REF} \
    --targ ${T1W_BRAIN} \
    --reg ${WORK_DIR}/temp_reg.dat \
    --lta ${LTA} \
    --noedit \
    --fslregout ${BOLD_TO_T1W_MAT}

[ -f "${BOLD_TO_T1W_MAT}" ] || { echo "ERROR: tkregister2 produced no matrix"; exit 1; }

# ================================== Step 4: single resampling, realigned -> MNI
echo "Step 4: applywarp (realigned BOLD -> MNI)"
applywarp \
    --in=${BOLD_MC} \
    --ref=${MNI_TEMPLATE} \
    --out=${BOLD_MNI} \
    --warp=${WARP} \
    --premat=${BOLD_TO_T1W_MAT} \
    --interp=trilinear

[ -f "${BOLD_MNI}" ] || { echo "ERROR: applywarp produced no output"; exit 1; }

# ==================================================== Step 5: brain mask in MNI
echo "Step 5: brain mask"
fslmaths ${BOLD_MNI} -Tmean -bin ${MASK_MNI}

# ------------------------------------------------------------------- clean up
rm -rf ${WORK_DIR}

echo "================================================================================"
echo "Done: sub-${SUBJECT_ID} run-${RUN}"
echo "  ${BOLD_MNI}"
echo "End: $(date)"
echo "================================================================================"
exit 0
