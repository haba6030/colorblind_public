#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase1_procrustes_decoding/run_all_C010_with_residuals.sh  (development repository, commit 53c81c2)
# Role      : Batch wrapper for the SLURM driver.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
# Run C010 pipeline for all subjects and ROIs (with residuals)

SUBJECTS="01 02 03 04 05 06 07 08 09 10"
ROIS="V1 V2 V3 V4"

echo "Running C010 Pipeline with Residuals"
echo "====================================="
echo ""

for sub in $SUBJECTS; do
    for roi in $ROIS; do
        echo "Processing sub-${sub} ${roi}..."
        python run_full_dataset_C010.py --subject ${sub} --roi ${roi}

        if [ $? -ne 0 ]; then
            echo "ERROR: Failed for sub-${sub} ${roi}"
        fi
    done
done

echo ""
echo "✅ All pairs completed!"
