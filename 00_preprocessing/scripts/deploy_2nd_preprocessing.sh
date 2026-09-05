#!/bin/bash
# ============================================================================
# Public release copy: https://github.com/haba6030/colorblind_public
# Stage     : 00_preprocessing
# Manuscript: Methods 'MRI acquisition and preprocessing', 'ROI definition and response estimation'; Supplementary S2 (pipelines), S3 (ROI coverage)
# Original  : analysis/phase0_preprocessing/scripts/deploy_2nd_preprocessing.sh  (development repository, commit 53c81c2)
# Role      : Deploys the session-2 preprocessing jobs.
# Server paths, partitions and the input data are those of the development
# environment; this file documents the job configuration and is not runnable here.
# ============================================================================
# Deploy + submit 2nd-dataset (exp2) Method-3 preprocessing.
# Run LOCALLY once the server (node3 via ProxyJump) is reachable.
#   bash deploy_2nd_preprocessing.sh           # upload data+script, do NOT submit
#   bash deploy_2nd_preprocessing.sh --submit  # also sbatch-submit
set -euo pipefail

LOCAL_REPO="/Users/jinilkim/Library/CloudStorage/OneDrive-Personal/Projects/colorBlind_analysis"
LOCAL_BIDS="${LOCAL_REPO}/data/bids_2nd/"
LOCAL_SBATCH="${LOCAL_REPO}/analysis/phase0_preprocessing/scripts/run_method3_header_mi_2nd.sbatch"

SRV=node3
SRV_BIDS="/storage/connectome/haba6030/bids_2nd"
SRV_SCRIPTDIR="/scratch/connectome/haba6030/colorBlind/analysis/phase0_preprocessing/scripts"
SRV_LOGDIR="/scratch/connectome/haba6030/colorBlind/analysis/prep_trials/logs"

echo "==> 0. connectivity"
ssh -o ConnectTimeout=30 "$SRV" 'hostname' || { echo "server unreachable"; exit 1; }

echo "==> 1. make server dirs"
ssh "$SRV" "mkdir -p '$SRV_BIDS' '$SRV_SCRIPTDIR' '$SRV_LOGDIR'"

echo "==> 2. upload bids_2nd (~603M, rsync resumable)"
rsync -avz --partial --progress "$LOCAL_BIDS" "${SRV}:${SRV_BIDS}/"

echo "==> 3. upload sbatch"
scp "$LOCAL_SBATCH" "${SRV}:${SRV_SCRIPTDIR}/"

echo "==> 4. verify inputs on server"
ssh "$SRV" "
  echo 'T1w:'; ls -la '${SRV_BIDS}/sub-08/anat/sub-08_acq-mprage_T1w.nii.gz'
  echo 'BOLD runs:'; ls '${SRV_BIDS}/sub-08/func/'*_task-rsvp_run-*_bold.nii.gz | wc -l
  echo 'MNI template:'; ls -la /scratch/connectome/haba6030/colorBlind/templates/MNI152NLin2009cAsym_res-2_T1_brain.nii.gz
"

if [[ "${1:-}" == "--submit" ]]; then
  echo "==> 5. submit"
  ssh "$SRV" "cd '$SRV_SCRIPTDIR' && sbatch run_method3_header_mi_2nd.sbatch && squeue -u haba6030"
else
  echo "==> upload done. To submit:"
  echo "    ssh $SRV \"cd '$SRV_SCRIPTDIR' && sbatch run_method3_header_mi_2nd.sbatch\""
fi
