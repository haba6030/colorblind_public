#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/run_exp2_C010.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver (amplitudes).
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=exp2_C010
#SBATCH --qos=shared
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1:00:00
#SBATCH --no-requeue
#SBATCH --array=1
#SBATCH --output=logs/exp2_C010_%A_%a.out
#SBATCH --error=logs/exp2_C010_%A_%a.err

# exp2 per-condition C010 amplitude extraction.
# array maps to ROI: 1=V1 2=V2 3=V3 4=V4 . Test with --array=1, then 1-4.
# Usage: sbatch [--array=1-4] run_exp2_C010.sbatch <native|matched> <subject>
#   sub-08: sbatch --array=1-4 run_exp2_C010.sbatch native       (default subject 08)
#   sub-09: sbatch --array=1-4 run_exp2_C010.sbatch native 09
ROIS=(V1 V2 V3 V4)
ROI=${ROIS[$((SLURM_ARRAY_TASK_ID - 1))]}
MV=${1:-native}    # native (masknone n coverage) | matched (exp1 maskfunc)
SUBJECT=${2:-08}   # CVD subject id (08 deutan / 09 protan)

echo "========================================"
echo "Job $SLURM_JOB_ID  task $SLURM_ARRAY_TASK_ID  sub-${SUBJECT}  ROI ${ROI}  variant ${MV}"
echo "========================================"

source ~/.bashrc
conda activate nilearn

cd /scratch/connectome/haba6030/colorBlind/analysis/phase6_behavioral_analysis/exp2_neural/scripts

python exp2_C010_conditions.py --subject ${SUBJECT} --roi ${ROI} --mask-variant ${MV}
