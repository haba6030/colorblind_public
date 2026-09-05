#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 05_identifiability
# Manuscript: Methods 'Identifiability and recovery'; Supplementary S12 (tab:identifiability); S13 control leave-one-out magnitude anchor
# Original  : analysis/phase5_filter_optimization/scripts/run_null_label_perm.sbatch  (development repository, commit 53c81c2)
# Role      : SLURM driver (Test 2c).
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=null_perm
#SBATCH --nodelist=node2
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/scratch/connectome/haba6030/colorBlind/analysis/phase5_filter_optimization
#SBATCH --output=logs/null_label_perm_%j.out
#SBATCH --error=logs/null_label_perm_%j.err

source ~/.bashrc
conda activate nilearn

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

mkdir -p results/redteam logs

mpirun -np 1 python scripts/null_label_permutation.py \
    --n-perm 1000 \
    --candidates all
