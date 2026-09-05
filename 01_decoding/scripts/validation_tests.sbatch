#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 01_decoding
# Manuscript: Results section 1; Figure 4; Methods 'Hue-channel basis model', 'Two decoding schemes'; Supplementary S5 (decoders), S6 (GCV), S7 (cross-validation), S16 (effect sizes), S17 (alignment robustness)
# Original  : analysis/phase3_decoder_comparing/model_comparison_validation/scripts/validation_tests.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver for validation_tests.py.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=decoder_val
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --output=/scratch/connectome/haba6030/colorBlind/analysis/phase3_decoder_comparing/model_comparison_validation/logs/decoder_val_%j.out
#SBATCH --error=/scratch/connectome/haba6030/colorBlind/analysis/phase3_decoder_comparing/model_comparison_validation/logs/decoder_val_%j.err

# Environment
source ~/.bashrc
conda activate nilearn

# Paths
BASE=/scratch/connectome/haba6030/colorBlind/analysis/phase3_decoder_comparing/model_comparison_validation
SCRIPT_DIR=${BASE}/scripts
BASELINE_DIR=/scratch/connectome/haba6030/colorBlind/derivatives/full_dataset_C010
PERF_DIR=${BASE}/results/loro/procrustes
OUTPUT_DIR=${BASE}/results/validation
LOCO_DIR=${BASE}/results/loco/procrustes
GP_DIR=${BASE}/results/FE_group_prior/loco_nested

echo "=============================="
echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================="

# 1. LORO validation: permutation + generalization
echo "--- LORO Validation ---"
python ${SCRIPT_DIR}/validation_tests.py \
    --baseline_dir ${BASELINE_DIR} \
    --performance_dir ${PERF_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --alignment procrustes \
    --tests permutation generalization

# 2. LOCO + GP validation
echo "--- LOCO + GP Validation ---"
python ${SCRIPT_DIR}/validation_tests.py \
    --baseline_dir ${BASELINE_DIR} \
    --performance_dir ${PERF_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --alignment procrustes \
    --loco_dir ${LOCO_DIR} \
    --gp_dir ${GP_DIR} \
    --tests loco_bootstrap loco_group loco_permutation gp_comparison

echo "=============================="
echo "End: $(date)"
echo "=============================="
