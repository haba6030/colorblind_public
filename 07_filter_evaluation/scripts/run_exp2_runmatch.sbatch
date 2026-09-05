#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 07_filter_evaluation
# Manuscript: Results section 7; Figure 7; Supplementary S14 (design), S15 (tab:exp2_8afc, tab:exp2_loro, tab:exp2_geometry), Figures S2-S3
# Original  : analysis/phase6_behavioral_analysis/exp2_neural/scripts/run_exp2_runmatch.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver (run-matched geometry).
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=exp2_runmatch
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/connectome/haba6030/colorBlind/analysis/phase6_behavioral_analysis/exp2_neural/scripts
#SBATCH --output=/scratch/connectome/haba6030/colorBlind/analysis/phase6_behavioral_analysis/exp2_neural/scripts/logs/exp2_runmatch_%j.out
#SBATCH --error=/scratch/connectome/haba6030/colorBlind/analysis/phase6_behavioral_analysis/exp2_neural/scripts/logs/exp2_runmatch_%j.err
source ~/.bashrc
conda activate nilearn
mpirun -np 1 python exp2_runmatched_geometry.py sub-08 sub-09
echo "EXIT=$?"
