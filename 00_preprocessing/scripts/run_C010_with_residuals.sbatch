#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase1_procrustes_decoding/run_C010_with_residuals.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver for the primary-pipeline amplitudes.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=C010_residuals
#SBATCH --qos=shared
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --array=1-40
#SBATCH --output=logs/C010_residuals_%A_%a.out
#SBATCH --error=logs/C010_residuals_%A_%a.err

# Array job for all 40 subject-ROI pairs

# Subject-ROI combinations (40 total)
SUBJECTS=(01 01 01 01 02 02 02 02 03 03 03 03 04 04 04 04 05 05 05 05 06 06 06 06 07 07 07 07 08 08 08 08 09 09 09 09 10 10 10 10)
ROIS=(V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4 V1 V2 V3 V4)

# Get current subject and ROI (array index starts at 1)
IDX=$((SLURM_ARRAY_TASK_ID - 1))
SUBJECT=${SUBJECTS[$IDX]}
ROI=${ROIS[$IDX]}

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject: sub-${SUBJECT}"
echo "ROI: ${ROI}"
echo "========================================"

# Activate conda
source ~/.bashrc
conda activate nilearn

# Run pipeline
cd /scratch/connectome/haba6030/colorBlind/analysis/validation/preprocess_detrend_temp

python run_full_dataset_C010.py \
    --subject ${SUBJECT} \
    --roi ${ROI}

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Success: sub-${SUBJECT} ${ROI}"
else
    echo "✗ Failed: sub-${SUBJECT} ${ROI} (exit code: $EXIT_CODE)"
fi

exit $EXIT_CODE
