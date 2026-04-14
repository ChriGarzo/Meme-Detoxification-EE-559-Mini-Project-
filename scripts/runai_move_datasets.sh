#!/usr/bin/env bash
set -e

# =============================================================================
# Move uploaded datasets from home PVC to scratch PVC.
# Run this after scp-ing datasets to /home/<user>/datasets_upload/
#
# Usage: bash scripts/runai_move_datasets.sh <UID_NUMBER> [dataset]
#
#   dataset   One of: mami | mmhs150k | all  (default: all)
#
# Examples:
#   bash scripts/runai_move_datasets.sh 314305 mami       ← only MAMI
#   bash scripts/runai_move_datasets.sh 314305 mmhs150k   ← only MMHS150K
#   bash scripts/runai_move_datasets.sh 314305            ← both (default)
# =============================================================================

if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER> [dataset]"
    exit 1
fi

UID_NUM="$1"
DATASET="${2:-all}"
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"
JOB_NAME="hmr-move-${DATASET}"

echo "=== Moving datasets to /scratch/ ==="
echo "  User:    ${USERNAME} (UID: ${UID_NUM})"
echo "  Dataset: ${DATASET}"
echo ""

runai submit ${JOB_NAME} \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- bash -c "DATASET=${DATASET} HOME_UPLOAD=/home/${USERNAME}/datasets_upload bash /home/${USERNAME}/hateful_meme_rewriting/scripts/move_datasets_to_scratch.sh"

echo "Job submitted. Follow logs with:"
echo "  runai logs ${JOB_NAME} -p course-ee-559-${USERNAME} --follow"
