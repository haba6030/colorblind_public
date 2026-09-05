#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/run_exp2_convergent.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver (convergent geometry).
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=exp2_conv
#SBATCH --qos=shared
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:30:00
#SBATCH --no-requeue
#SBATCH --output=logs/exp2_conv_%j.out
#SBATCH --error=logs/exp2_conv_%j.err

# Usage: sbatch run_exp2_convergent.sbatch <native|matched> <subject>   (subject default 08)
VARIANT=${1:-native}
SUBJECT=${2:-08}
source ~/.bashrc
conda activate nilearn
cd /scratch/connectome/haba6030/colorBlind/analysis/phase6_behavioral_analysis/exp2_neural/scripts
# brainiak SRM needs MPI init -> mpirun -np 1 (bare python forbidden, project CLAUDE.md §9)
mpirun -np 1 python exp2_convergent.py --variant ${VARIANT} --subject ${SUBJECT}
