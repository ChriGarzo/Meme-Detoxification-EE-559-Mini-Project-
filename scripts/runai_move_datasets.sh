#!/usr/bin/env bash
set -e

# =============================================================================
# Move uploaded datasets from home PVC to scratch PVC.
# Run this after scp-ing datasets to /home/<user>/datasets_upload/
#
# Usage: bash scripts/runai_move_datasets.sh <UID_NUMBER>
# =============================================================================

if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER>"
    exit 1
fi

UID_NUM="$1"
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

echo "=== Moving datasets to /scratch/ ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo ""

runai submit hmr-move-datasets \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- bash /home/${USERNAME}/hateful_meme_rewriting/scripts/move_datasets_to_scratch.sh

echo "Job submitted. Follow logs with:"
echo "  runai logs hmr-move-datasets -p course-ee-559-${USERNAME} --follow"
