#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/hmc_reanalysis/server_recovered/identity_check.sh  (development repository, commit 53c81c2)
# Role      : Checks that the composed transform reduces to the primary transform when the motion term is identity.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
# hmc_v2 기준 볼륨 항등 검사 — node1 에서 실행한다.
# 기준 볼륨의 MCFLIRT 행렬은 항등이므로 hmc_v2 출력이 정본과 같아야 한다.
export FSLDIR=/usr/local/fsl; export PATH=$FSLDIR/bin:$PATH; source $FSLDIR/etc/fslconf/fsl.sh
R=/storage/connectome/haba6030
T=/tmp/idcheck_$$; mkdir -p $T
for SUB in 01 08 09; do
  RUN=1
  C=$R/fmriprep_out_method3_header_mi/sub-$SUB/func/sub-${SUB}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
  H=$R/fmriprep_out_method3_hmc_v2/sub-$SUB/func/sub-${SUB}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
  if [ ! -f "$C" ] || [ ! -f "$H" ]; then echo "sub-$SUB MISSING"; continue; fi
  NV=$(fslinfo $C | awk "/^dim4/{print \$2}"); REF=$((NV/2))
  fslroi $C $T/c $REF 1; fslroi $H $T/h $REF 1
  fslmaths $T/h -sub $T/c $T/d
  fslroi $C $T/c2 $((REF+1)) 1; fslmaths $T/c2 -sub $T/c $T/d2
  echo "sub-$SUB NV=$NV REF=$REF"
  echo "   mean signal                       : $(fslstats $T/c -M)"
  echo "   |hmc - canon| at reference volume : $(fslstats $T/d -a -M)  range $(fslstats $T/d -R)"
  echo "   |vol(REF+1) - vol(REF)| (scale)   : $(fslstats $T/d2 -a -M)"
done
