#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/run_exp2_analysis.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver (hc_likeness).
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=exp2_hclike
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/connectome/haba6030/colorBlind/analysis/phase6_behavioral_analysis/exp2_neural/scripts
#SBATCH --output=logs/exp2_hclike_%j.out
#SBATCH --error=logs/exp2_hclike_%j.err

# Usage: sbatch run_exp2_analysis.sbatch <native|matched> <subject>   (subject default 08)
VARIANT=${1:-native}
SUBJECT=${2:-08}
source ~/.bashrc
conda activate nilearn
python exp2_hc_likeness.py --variant ${VARIANT} --subject ${SUBJECT}
