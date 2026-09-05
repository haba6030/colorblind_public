#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/hmc_reanalysis/server_recovered/analyze_hmc.sh  (development repository, commit 53c81c2)
# Role      : tSNR comparison between the two pipelines -> hmc_summary.csv.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
# HMC arm vs canonical: ROI-atlas overlap + tSNR, all subjects x runs
export FSLDIR=/usr/local/fsl; export PATH=$FSLDIR/bin:$PATH; source $FSLDIR/etc/fslconf/fsl.sh
R=/storage/connectome/haba6030; S=/scratch/connectome/haba6030/colorBlind
O=$R/pilot/hmc_full; G=$O/figs; mkdir -p $G; cd $O

echo "subject,run,roi,n_atlas,n_canon,n_hmc,tsnr_canon,tsnr_hmc" > $O/hmc_summary.csv

for SUB in 01 02 03 04 05 06 07 08 09; do
  D=$S/analysis/roi_masks/method3_header_mi/sub-$SUB/roi_pipeline
  for RUN in 1 2 3 4 5 6; do
    C=$R/fmriprep_out_method3_header_mi/sub-$SUB/func/sub-${SUB}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
    H=$R/fmriprep_out_method3_hmc_v2/sub-$SUB/func/sub-${SUB}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
    [ -f "$C" ] && [ -f "$H" ] || continue
    T=/tmp/an_${SUB}_${RUN}_$$; rm -rf $T; mkdir -p $T

    for TAG in c h; do
      if [ "$TAG" = "c" ]; then IM=$C; else IM=$H; fi
      fslmaths $IM -Tmean $T/m_$TAG
      fslmaths $IM -Tstd  $T/s_$TAG
      fslmaths $T/m_$TAG -div $T/s_$TAG -nan $T/t_$TAG
      thr=$(fslstats $T/m_$TAG -P 40)
      fslmaths $T/m_$TAG -thr $thr -bin $T/bm_$TAG
    done

    for V in V1 V2 V3 hV4; do
      A=$D/${V}_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjTrue.nii.gz
      [ -f "$A" ] || continue
      na=$(fslstats $A -V | awk '{print $1}')
      fslmaths $A -mul $T/bm_c $T/ic; nc=$(fslstats $T/ic -V | awk '{print $1}')
      fslmaths $A -mul $T/bm_h $T/ih; nh=$(fslstats $T/ih -V | awk '{print $1}')
      tc=$(fslstats $T/t_c -k $A -M)
      th=$(fslstats $T/t_h -k $A -M)
      echo "sub-$SUB,$RUN,$V,$na,$nc,$nh,$tc,$th" >> $O/hmc_summary.csv
    done

    if [ "$RUN" = "1" ]; then
      fslmaths $D/V1_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjTrue.nii.gz -bin $T/r1
      fslmaths $D/hV4_mask_thr50_intnearest_binTrue_masknone_gmTrue_subjTrue.nii.gz -bin -mul 2 $T/r4
      fslmaths $T/r1 -add $T/r4 -bin $T/roi
      slicer $T/m_c $T/roi -A 1500 $G/ROI_sub-${SUB}_a_CANON.png
      slicer $T/m_h $T/roi -A 1500 $G/ROI_sub-${SUB}_b_HMC.png
      fslmaths $T/t_h -sub $T/t_c $T/tdiff
      slicer $T/tdiff -l render1 -A 1200 $G/TSNRDIFF_sub-${SUB}.png
    fi
    rm -rf $T
  done
  echo "sub-$SUB analyzed"
done
echo DONE > $O/.analyzed
