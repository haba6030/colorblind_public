#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase3_decoder_comparing/model_comparison_validation/scripts/loro_baseline_srm.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver (SRM space).
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=loro_cmp
#SBATCH --nodelist=node2
#SBATCH --array=1-10%5
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/connectome/haba6030/colorBlind/analysis/phase3_decoder_comparing/model_comparison_validation/logs/decoder_cmp_%a_%j.out
#SBATCH --error=/scratch/connectome/haba6030/colorBlind/analysis/phase3_decoder_comparing/model_comparison_validation/logs/decoder_cmp_%a_%j.err

# Environment
source ~/.bashrc
conda activate nilearn

# Paths
SCRIPT_DIR=/scratch/connectome/haba6030/colorBlind/analysis/phase3_decoder_comparing/model_comparison_validation/scripts
BASELINE_DIR=/scratch/connectome/haba6030/colorBlind/derivatives/full_dataset_C010
OUTPUT_DIR=/scratch/connectome/haba6030/colorBlind/analysis/phase3_decoder_comparing/model_comparison_validation/results/loro/srm

# Subject mapping (array index 1-10 → subject ID)
SUBJECTS=(01 02 03 04 05 06 07 08 09 10)
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID-1]}

echo "=============================="
echo "Job: $SLURM_JOB_ID | Array: $SLURM_ARRAY_TASK_ID"
echo "Subject: sub-${SUBJECT}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================="

# Run model comparison (all 6 models, all 4 ROIs, both alignments)
python ${SCRIPT_DIR}/loro_baseline.py \
    --baseline_dir ${BASELINE_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --subject ${SUBJECT} \
    --rois V1 V2 V3 V4 \
    --models LDA Ridge KernelRidge SVM MLP ForwardEncoding \
    --alignment srm

echo "=============================="
echo "End: $(date)"
echo "=============================="
