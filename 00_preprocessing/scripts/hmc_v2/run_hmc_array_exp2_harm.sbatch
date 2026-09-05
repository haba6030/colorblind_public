#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/hmc_reanalysis/server_recovered/run_hmc_array_exp2_harm.sbatch  (development repository, commit 53c81c2)
# Role      : Session-2 harmonised + head-motion-corrected arm.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
#SBATCH --job-name=hmc_exp2_harm
#SBATCH --array=1-16%8
#SBATCH --nodelist=node2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --chdir=/storage/connectome/haba6030/pilot/hmc_exp2_harm
#SBATCH --output=/storage/connectome/haba6030/pilot/hmc_exp2_harm/logs/hmc_%A_%a.out
#SBATCH --error=/storage/connectome/haba6030/pilot/hmc_exp2_harm/logs/hmc_%A_%a.err

# exp2 realignment arm on the DEFACED (anat-harmonized) session-2 BIDS.
# Single interpolation: mcflirt -mats -> convert_xfm -concat with bold->t1w
# -> one applywarp per volume. Same chain as fmriprep_out_method3_hmc_v2 (exp1).
# NOTE: transforms MUST come from the harmonized anat run, not fmriprep_out_method3_2nd,
# otherwise the defacing harmonisation has no effect on T1w->MNI.

export FSLDIR=/usr/local/fsl; export PATH=$FSLDIR/bin:$PATH; source $FSLDIR/etc/fslconf/fsl.sh
export FREESURFER_HOME=/usr/local/freesurfer/7.2.0; export PATH=$FREESURFER_HOME/bin:$PATH
export FS_LICENSE=/storage/connectome/haba6030/fs_license.txt

R=/storage/connectome/haba6030
S=/scratch/connectome/haba6030/colorBlind
MNI=$S/templates/MNI152NLin2009cAsym_res-2_T1_brain.nii.gz
SUBS=(08 09)
I=$((SLURM_ARRAY_TASK_ID-1)); SUB=${SUBS[$((I/8))]}; RUN=$((I%8+1))

OUT=$R/fmriprep_out_method3_2nd_harm_hmc/sub-$SUB/func; mkdir -p $OUT
FINAL=$OUT/sub-${SUB}_task-rsvp_run-${RUN}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
[ -f "$FINAL" ] && { echo "exists, skip"; exit 0; }

BOLD=$R/bids_2nd_defaced/sub-$SUB/func/sub-${SUB}_task-rsvp_run-${RUN}_bold.nii.gz
[ -f "$BOLD" ] || { echo "no BOLD $BOLD"; exit 1; }
T=$R/fmriprep_out_method3_2nd_harm/sub-$SUB/transforms
WK=$R/fmriprep_work_method3_2nd_harm_sub-$SUB
export SUBJECTS_DIR=$WK/freesurfer_subjects
[ -f "$T/sub-${SUB}_t1w_to_mni_warp.nii.gz" ] || { echo "no warp; run anat stage first"; exit 1; }

W=/tmp/hmc2h_${SUB}_${RUN}_$$; rm -rf $W; mkdir -p $W/v; cd $W
NV=$(fslinfo $BOLD | awk "/^dim4/{print \$2}"); REF=$((NV/2))
echo "sub-$SUB run-$RUN NV=$NV REF=$REF"

mcflirt -in $BOLD -out mc -refvol $REF -mats -plots
fslroi $BOLD boldref $REF 1
tkregister2 --mov boldref.nii.gz --targ $WK/sub-${SUB}_T1w_brain.nii.gz --reg tmp.dat \
  --lta $T/sub-${SUB}_run-${RUN}_bold_to_t1w.lta --noedit --fslregout b2t.mat > /dev/null

for v in $(seq 0 $((NV-1))); do
  vv=$(printf "%04d" $v)
  convert_xfm -omat v/M_$vv.mat -concat b2t.mat mc.mat/MAT_$vv
  fslroi $BOLD v/in_$vv $v 1
  applywarp --in=v/in_$vv --ref=$MNI --premat=v/M_$vv.mat \
            --warp=$T/sub-${SUB}_t1w_to_mni_warp.nii.gz --interp=trilinear --out=v/o_$vv
done
fslmerge -t $FINAL $(ls v/o_*.nii.gz | sort)
cp mc.par $OUT/sub-${SUB}_task-rsvp_run-${RUN}_desc-motion.par 2>/dev/null
cd /; rm -rf $W
echo "DONE sub-$SUB run-$RUN"
