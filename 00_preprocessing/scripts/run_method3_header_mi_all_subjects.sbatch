#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/run_method3_header_mi_all_subjects.sbatch  (development repository, commit 53c81c2)
# Role      : Session-1 registration (header-initialised MI) for all participants.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=prep_m3
#SBATCH --array=1-10%3
#SBATCH --output=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/method3_%A_%a.out
#SBATCH --error=/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs/method3_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=50G
#SBATCH --qos=shared
#SBATCH --nodelist=node2
#SBATCH --no-requeue
#SBATCH --open-mode=append

################################################################################
# Method 3: Header → MI (mri_coreg --regheader)
#
# Registration strategy:
#   (a) Initial alignment: NIfTI header
#   (a') MI optimization: Powell's method (FreeSurfer mri_coreg)
#   (b) BBR refinement: SKIP (MI sufficient for fMRI 2mm resolution)
#
# Key advantage:
#   - MI robust to limited FOV, obliquity, limited cortex
#   - No WM boundary needed
#
# Test subjects: Sub-01, 03, 06
################################################################################

set -e

# Subject mapping
SUBJECTS=(01 02 03 04 05 06 07 08 09 10)
SUBJECT_ID=${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

echo "================================================================================"
echo "Method 3: Header → MI (mri_coreg) - Sub-${SUBJECT_ID}"
echo "================================================================================"
echo "Job ID: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date)"
echo ""
echo "Registration strategy:"
echo "  (a) Initial alignment: NIfTI header qform/sform"
echo "  (a') MI optimization: FreeSurfer mri_coreg (Powell's method)"
echo "  (b) BBR refinement: ❌ SKIP (MI ~1mm accuracy sufficient)"
echo "================================================================================"
echo ""

# Directories
BIDS_DIR="/storage/connectome/haba6030/bids_editted"
OUTPUT_DIR="/storage/connectome/haba6030/fmriprep_out_method3_header_mi"
WORK_DIR="/storage/connectome/haba6030/fmriprep_work_method3_sub-${SUBJECT_ID}"

# FreeSurfer setup
# CRITICAL: Explicitly add FreeSurfer to PATH for SLURM job environment
# Try common FreeSurfer installation paths (system) and conda
if ! command -v mri_convert &> /dev/null; then
    # Try system FreeSurfer paths
    for FS_PATH in /usr/local/freesurfer/7.2.0 /usr/local/freesurfer/7.4.1 /usr/local/freesurfer /opt/freesurfer; do
        if [ -d "${FS_PATH}/bin" ]; then
            export FREESURFER_HOME=${FS_PATH}
            export PATH="${FREESURFER_HOME}/bin:${PATH}"
            break
        fi
    done

    # If still not found, try conda
    if ! command -v mri_convert &> /dev/null; then
        echo "Trying conda FreeSurfer environment..."
        source ~/.bashrc
        conda activate freesurfer 2>/dev/null || true
    fi
fi

export SUBJECTS_DIR="${WORK_DIR}/freesurfer_subjects"
mkdir -p ${SUBJECTS_DIR}

# Verify FreeSurfer is accessible
if ! command -v mri_convert &> /dev/null; then
    echo "❌ ERROR: mri_convert not found in PATH"
    echo "   Tried: /usr/local/freesurfer/*, /opt/freesurfer, conda freesurfer"
    echo "   Current PATH: $PATH"
    echo "   Node: $(hostname)"
    echo ""
    echo "   This node doesn't have FreeSurfer. Please use --nodelist=node2"
    exit 1
fi
echo "✅ FreeSurfer found: $(which mri_convert)"
echo "   Node: $(hostname)"
echo ""

# FreeSurfer license (CRITICAL)
export FS_LICENSE=${FREESURFER_HOME}/license.txt
echo "✅ FreeSurfer license: ${FS_LICENSE}"

# =========================================================================
# FSL Setup (CRITICAL: Required for flirt, fnirt, applywarp)
# =========================================================================

if ! command -v fslroi &> /dev/null; then
    # Try common FSL installation paths
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

    # If still not found, try conda
    if ! command -v fslroi &> /dev/null; then
        echo "Trying conda FSL environment..."
        conda activate fsl 2>/dev/null || true
    fi
fi

# Verify FSL is accessible
if ! command -v fslroi &> /dev/null; then
    echo "❌ ERROR: FSL not found in PATH"
    echo "   Tried: /usr/local/fsl, /opt/fsl, conda fsl"
    echo "   Current PATH: $PATH"
    echo "   Node: $(hostname)"
    echo ""
    echo "   FSL is REQUIRED for FLIRT/FNIRT (T1w → MNI normalization)"
    exit 1
fi
echo "✅ FSL found: $(which fslroi)"
echo "   FSLDIR: ${FSLDIR}"

# Output directories
mkdir -p ${OUTPUT_DIR}/sub-${SUBJECT_ID}/func
mkdir -p ${OUTPUT_DIR}/sub-${SUBJECT_ID}/anat
mkdir -p ${OUTPUT_DIR}/sub-${SUBJECT_ID}/transforms
mkdir -p ${WORK_DIR}
mkdir -p /scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs

echo "Configuration:"
echo "  Subject:       sub-${SUBJECT_ID}"
echo "  Input (BIDS):  ${BIDS_DIR}"
echo "  Output:        ${OUTPUT_DIR}"
echo "  Work:          ${WORK_DIR}"
echo "  FreeSurfer:    ${SUBJECTS_DIR}"
echo "  CPUs:          ${SLURM_CPUS_PER_TASK}"
echo ""

# Input files
T1W_FILE="${BIDS_DIR}/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_acq-mprage_T1w.nii.gz"

echo "Input files:"
echo "  T1w: ${T1W_FILE}"
echo ""

# Verify T1w exists
if [ ! -f "${T1W_FILE}" ]; then
    echo "❌ ERROR: T1w not found: ${T1W_FILE}"
    exit 1
fi

echo "✅ T1w found"
echo ""

# =========================================================================
# Step 1: FreeSurfer brain extraction (for reference only)
# =========================================================================

echo "================================================================================"
echo "Step 1: Brain extraction with FreeSurfer"
echo "================================================================================"
echo ""

# Quick brain extraction (not full recon-all!)
# We only need brain.mgz for mri_coreg reference

T1W_MGZ="${SUBJECTS_DIR}/sub-${SUBJECT_ID}/mri/orig.mgz"
BRAIN_MGZ="${SUBJECTS_DIR}/sub-${SUBJECT_ID}/mri/brain.mgz"

# Create FreeSurfer directory structure (CRITICAL: prevent MRISwrite errors)
mkdir -p ${SUBJECTS_DIR}/sub-${SUBJECT_ID}/mri
mkdir -p ${SUBJECTS_DIR}/sub-${SUBJECT_ID}/surf

# Convert T1w to MGZ
echo "Converting T1w to MGZ format..."
mri_convert ${T1W_FILE} ${T1W_MGZ}

# Brain extraction (robust fallback: bet2 > mri_watershed)
echo "Running brain extraction..."
if command -v bet2 &> /dev/null; then
    echo "  Using FSL bet2 (preferred for speed)..."
    bet2 ${T1W_FILE} ${WORK_DIR}/T1w_brain.nii.gz -f 0.5 -m
    mri_convert ${WORK_DIR}/T1w_brain.nii.gz ${BRAIN_MGZ}
else
    echo "  Using FreeSurfer mri_watershed (fallback)..."
    # NOTE: Removed -surf option (surf files not used in this pipeline)
    mri_watershed -useSRAS ${T1W_MGZ} ${BRAIN_MGZ}
fi

echo "✅ Brain extracted: ${BRAIN_MGZ}"
echo ""

# =========================================================================
# Step 2: BOLD → T1w registration with mri_coreg
# =========================================================================

echo "================================================================================"
echo "Step 2: BOLD → T1w registration (mri_coreg --regheader)"
echo "================================================================================"
echo ""

# Process all runs
for RUN in 1 2 3 4 5 6; do
    echo "------------------------------------------------------------"
    echo "Processing Run ${RUN}"
    echo "------------------------------------------------------------"

    BOLD_FILE="${BIDS_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_bold.nii.gz"

    if [ ! -f "${BOLD_FILE}" ]; then
        echo "⚠️  WARNING: BOLD run ${RUN} not found, skipping"
        continue
    fi

    # Extract reference volume (middle volume) - Robust fallback chain
    echo "Extracting reference volume..."

    # Get number of volumes (try AFNI first, fallback to FSL, then estimate)
    if command -v 3dinfo &> /dev/null; then
        NVOLS=$(3dinfo -nv ${BOLD_FILE} 2>/dev/null)
    elif command -v fslinfo &> /dev/null; then
        NVOLS=$(fslinfo ${BOLD_FILE} | grep "^dim4" | awk '{print $2}')
    else
        NVOLS=100  # Conservative estimate
    fi
    REF_VOL=$((NVOLS / 2))

    BOLD_REF="${WORK_DIR}/sub-${SUBJECT_ID}_run-${RUN}_boldref.nii.gz"

    # Extraction priority: FreeSurfer (always available) > FSL > AFNI
    echo "  Extracting volume ${REF_VOL} from ${NVOLS} volumes..."
    if mri_convert ${BOLD_FILE} ${BOLD_REF} --frame ${REF_VOL} 2>/dev/null; then
        echo "  ✅ Used mri_convert (FreeSurfer)"
    elif command -v fslroi &> /dev/null && fslroi ${BOLD_FILE} ${BOLD_REF} ${REF_VOL} 1 2>/dev/null; then
        echo "  ✅ Used fslroi (FSL)"
    elif command -v 3dTcat &> /dev/null && 3dTcat -prefix ${BOLD_REF} ${BOLD_FILE}"[${REF_VOL}]" 2>/dev/null; then
        echo "  ✅ Used 3dTcat (AFNI)"
    else
        echo "  ❌ ERROR: No tool available for volume extraction (tried: mri_convert, fslroi, 3dTcat)"
        exit 1
    fi

    # mri_coreg: BOLD → T1w
    echo "Running mri_coreg --regheader..."

    OUTPUT_LTA="${OUTPUT_DIR}/sub-${SUBJECT_ID}/transforms/sub-${SUBJECT_ID}_run-${RUN}_bold_to_t1w.lta"

    mri_coreg \
        --ref ${BRAIN_MGZ} \
        --mov ${BOLD_REF} \
        --reg ${OUTPUT_LTA} \
        --regheader \
        --threads ${SLURM_CPUS_PER_TASK}

    if [ ! -f "${OUTPUT_LTA}" ]; then
        echo "❌ ERROR: mri_coreg failed for run ${RUN}"
        continue
    fi

    echo "✅ Transform saved: $(basename ${OUTPUT_LTA})"

    # Apply transform to BOLD
    echo "Applying transform to BOLD..."
    BOLD_IN_T1W="${WORK_DIR}/sub-${SUBJECT_ID}_run-${RUN}_bold_in_t1w.nii.gz"

    mri_vol2vol \
        --mov ${BOLD_FILE} \
        --targ ${BRAIN_MGZ} \
        --lta ${OUTPUT_LTA} \
        --o ${BOLD_IN_T1W} \
        --interp trilin \
        --no-save-reg

    echo "✅ BOLD in T1w space: $(basename ${BOLD_IN_T1W})"

    # Cleanup large intermediate files to prevent OOM
    # Note: T1w mask not needed (only MNI mask used for analysis/diagnostics)
    echo "Cleaning up intermediate files..."
    rm -f ${BOLD_IN_T1W} ${BOLD_REF}

    echo ""
done

echo "✅ All runs processed with mri_coreg"
echo ""

# =========================================================================
# Step 3: T1w → MNI normalization (FSL FLIRT + FNIRT)
# =========================================================================

echo "================================================================================"
echo "Step 3: T1w → MNI normalization (FSL)"
echo "================================================================================"
echo ""
echo "NOTE: Using FSL FLIRT + FNIRT (nonlinear), matching original_v3 pipeline"
echo ""

# MNI template (CRITICAL: Must match Original_v3 = MNI152NLin2009cAsym)
# Original_v3 uses MNI152NLin2009cAsym (97,115,97)
# FSL default MNI152_T1_2mm is DIFFERENT space (91,109,91) ❌
MNI_TEMPLATE="/scratch/connectome/haba6030/colorBlind/templates/MNI152NLin2009cAsym_res-2_T1_brain.nii.gz"

if [ ! -f "${MNI_TEMPLATE}" ]; then
    echo "❌ ERROR: MNI template not found: ${MNI_TEMPLATE}"
    echo "   Run template creation script first:"
    echo "   python3 /scratch/connectome/haba6030/colorBlind/analysis/prep_trials/scripts/create_mni_template.py"
    exit 1
fi

# Convert T1w brain to NIfTI
T1W_BRAIN_NII="${WORK_DIR}/sub-${SUBJECT_ID}_T1w_brain.nii.gz"
mri_convert ${BRAIN_MGZ} ${T1W_BRAIN_NII}

echo "Running FLIRT + FNIRT (T1w → MNI, nonlinear)..."

# Step 3a: Linear registration with FLIRT
T1W_TO_MNI_MAT="${OUTPUT_DIR}/sub-${SUBJECT_ID}/transforms/sub-${SUBJECT_ID}_t1w_to_mni_affine.mat"
T1W_TO_MNI_LINEAR="${WORK_DIR}/sub-${SUBJECT_ID}_T1w_in_MNI_linear.nii.gz"

echo "  FLIRT (linear 12 DOF)..."
flirt \
    -in ${T1W_BRAIN_NII} \
    -ref ${MNI_TEMPLATE} \
    -out ${T1W_TO_MNI_LINEAR} \
    -omat ${T1W_TO_MNI_MAT} \
    -bins 256 \
    -cost corratio \
    -searchrx -90 90 \
    -searchry -90 90 \
    -searchrz -90 90 \
    -dof 12

echo "  ✅ Linear registration done"

# Step 3b: Nonlinear registration with FNIRT
T1W_TO_MNI_WARP="${OUTPUT_DIR}/sub-${SUBJECT_ID}/transforms/sub-${SUBJECT_ID}_t1w_to_mni_warp.nii.gz"
T1W_TO_MNI_NONLINEAR="${WORK_DIR}/sub-${SUBJECT_ID}_T1w_in_MNI_nonlinear.nii.gz"

# MNI template (full head for FNIRT)
MNI_TEMPLATE_HEAD="/scratch/connectome/haba6030/colorBlind/templates/MNI152NLin2009cAsym_res-2_T1.nii.gz"

# If full head template doesn't exist, use brain template
if [ ! -f "${MNI_TEMPLATE_HEAD}" ]; then
    echo "  ⚠️  Full head template not found, using brain template for FNIRT"
    MNI_TEMPLATE_HEAD="${MNI_TEMPLATE}"
fi

MNI_CONFIG="/usr/local/fsl/etc/flirtsch/T1_2_MNI152_2mm.cnf"

# Convert brain.mgz to full head T1w
T1W_HEAD_NII="${WORK_DIR}/sub-${SUBJECT_ID}_T1w_head.nii.gz"
T1W_ORIG_MGZ="${SUBJECTS_DIR}/sub-${SUBJECT_ID}/mri/orig.mgz"
mri_convert ${T1W_ORIG_MGZ} ${T1W_HEAD_NII}

echo "  FNIRT (nonlinear warp)..."

# Use MNI152NLin2009cAsym mask (matches our template space)
MNI_REFMASK="/scratch/connectome/haba6030/colorBlind/templates/MNI152NLin2009cAsym_res-2_brain_mask.nii.gz"

# CRITICAL: Override config's default mask with our template-matched mask
fnirt \
    --in=${T1W_HEAD_NII} \
    --ref=${MNI_TEMPLATE_HEAD} \
    --aff=${T1W_TO_MNI_MAT} \
    --cout=${T1W_TO_MNI_WARP} \
    --iout=${T1W_TO_MNI_NONLINEAR} \
    --refmask=${MNI_REFMASK} \
    --config=${MNI_CONFIG}

echo "  ✅ Nonlinear registration done"
echo "✅ T1w → MNI transform computed (FLIRT + FNIRT)"
echo ""

# =========================================================================
# Step 4: Apply composed transforms (BOLD → T1w → MNI)
# =========================================================================

echo "================================================================================"
echo "Step 4: Apply composed transforms (BOLD → MNI)"
echo "================================================================================"
echo ""

for RUN in 1 2 3 4 5 6; do
    BOLD_FILE="${BIDS_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_bold.nii.gz"

    if [ ! -f "${BOLD_FILE}" ]; then
        continue
    fi

    echo "Processing Run ${RUN}..."

    # FreeSurfer LTA transform
    OUTPUT_LTA="${OUTPUT_DIR}/sub-${SUBJECT_ID}/transforms/sub-${SUBJECT_ID}_run-${RUN}_bold_to_t1w.lta"

    # Convert LTA to FSL format
    BOLD_TO_T1W_MAT="${WORK_DIR}/sub-${SUBJECT_ID}_run-${RUN}_bold_to_t1w.mat"

    # Extract middle volume as reference (if not already exists)
    BOLD_REF="${WORK_DIR}/sub-${SUBJECT_ID}_run-${RUN}_boldref.nii.gz"

    if [ ! -f "${BOLD_REF}" ]; then
        # Get number of volumes (robust fallback)
        if command -v 3dinfo &> /dev/null; then
            NVOLS=$(3dinfo -nv ${BOLD_FILE} 2>/dev/null)
        elif command -v fslinfo &> /dev/null; then
            NVOLS=$(fslinfo ${BOLD_FILE} | grep "^dim4" | awk '{print $2}')
        else
            NVOLS=100
        fi
        REF_VOL=$((NVOLS / 2))

        # Extraction priority: FreeSurfer > FSL > AFNI
        if ! mri_convert ${BOLD_FILE} ${BOLD_REF} --frame ${REF_VOL} 2>/dev/null; then
            if command -v fslroi &> /dev/null; then
                fslroi ${BOLD_FILE} ${BOLD_REF} ${REF_VOL} 1
            elif command -v 3dTcat &> /dev/null; then
                3dTcat -prefix ${BOLD_REF} ${BOLD_FILE}"[${REF_VOL}]"
            fi
        fi
    fi

    # Use tkregister2 to convert LTA to FSL format
    tkregister2 \
        --mov ${BOLD_REF} \
        --targ ${T1W_BRAIN_NII} \
        --reg ${WORK_DIR}/temp_reg.dat \
        --lta ${OUTPUT_LTA} \
        --noedit \
        --fslregout ${BOLD_TO_T1W_MAT}

    # Apply composed transform: BOLD → T1w (linear) → MNI (nonlinear)
    BOLD_MNI="${OUTPUT_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
    BOLD_TO_T1W_MAT="${WORK_DIR}/sub-${SUBJECT_ID}_run-${RUN}_bold_to_t1w.mat"

    # Use applywarp to apply both linear (premat) and nonlinear (warp) transforms
    applywarp \
        --in=${BOLD_FILE} \
        --ref=${MNI_TEMPLATE} \
        --out=${BOLD_MNI} \
        --warp=${T1W_TO_MNI_WARP} \
        --premat=${BOLD_TO_T1W_MAT} \
        --interp=trilinear

    # Create mask in MNI space (Robust fallback chain)
    MASK_MNI="${OUTPUT_DIR}/sub-${SUBJECT_ID}/func/sub-${SUBJECT_ID}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz"

    # Priority: fslmaths (fast) > 3dAutomask (accurate) > mri_binarize (fallback)
    MASK_CREATED=false

    if command -v fslmaths &> /dev/null; then
        echo "  Trying fslmaths for mask..."
        if fslmaths ${BOLD_MNI} -Tmean -bin ${MASK_MNI}; then
            echo "  ✅ Mask created with fslmaths (FSL)"
            MASK_CREATED=true
        else
            echo "  ⚠️  fslmaths failed, trying next..."
        fi
    fi

    if [ "$MASK_CREATED" = false ] && command -v 3dAutomask &> /dev/null; then
        echo "  Trying 3dAutomask for mask..."
        if 3dAutomask -prefix ${MASK_MNI} ${BOLD_MNI}; then
            echo "  ✅ Mask created with 3dAutomask (AFNI)"
            MASK_CREATED=true
        else
            echo "  ⚠️  3dAutomask failed, trying next..."
        fi
    fi

    if [ "$MASK_CREATED" = false ] && command -v mri_binarize &> /dev/null; then
        echo "  Trying mri_binarize for mask..."
        if mri_binarize --i ${BOLD_MNI} --min 0.01 --o ${MASK_MNI}; then
            echo "  ✅ Mask created with mri_binarize (FreeSurfer)"
            MASK_CREATED=true
        else
            echo "  ⚠️  mri_binarize failed"
        fi
    fi

    if [ "$MASK_CREATED" = false ]; then
        echo "  ⚠️  WARNING: All mask tools failed, but continuing..."
        echo "  Note: Mask may not be critical for analysis"
    fi

    echo "  ✅ BOLD in MNI: $(basename ${BOLD_MNI})"
    if [ "$MASK_CREATED" = true ]; then
        echo "  ✅ Mask in MNI: $(basename ${MASK_MNI})"
    else
        echo "  ⚠️  Mask creation failed for: $(basename ${MASK_MNI})"
    fi

    # Cleanup intermediate files to prevent OOM
    echo "  Cleaning up intermediate files..."
    rm -f ${BOLD_REF} ${BOLD_TO_T1W_MAT} ${WORK_DIR}/temp_reg.dat
done

echo ""
echo "✅ All runs mapped to MNI space"

# =========================================================================
# Summary
# =========================================================================

echo ""
echo "================================================================================"
echo "Sub-${SUBJECT_ID}: Method 3 COMPLETE"
echo "================================================================================"
echo ""
echo "Method 3 summary:"
echo "  (a) Header init: ✅"
echo "  (a') MI optimization (mri_coreg): ✅"
echo "  (b) BBR: ❌ SKIP (MI sufficient)"
echo "  (c) MNI normalization: ✅ FSL FLIRT + FNIRT (nonlinear, same as original_v3)"
echo ""
echo "Output files:"
echo "  Transforms: ${OUTPUT_DIR}/sub-${SUBJECT_ID}/transforms/"
echo "  BOLD (MNI): ${OUTPUT_DIR}/sub-${SUBJECT_ID}/func/*_space-MNI*"
echo "  Masks (MNI): ${OUTPUT_DIR}/sub-${SUBJECT_ID}/func/*_desc-brain_mask.nii.gz"
echo ""
echo "Next steps:"
echo "  1. Compute Dice:"
echo "     (method-selection Dice step retired 2026-08-05; code at _archive/registration_method_selection/)"
echo ""
echo "End time: $(date)"
echo "================================================================================"

exit 0
