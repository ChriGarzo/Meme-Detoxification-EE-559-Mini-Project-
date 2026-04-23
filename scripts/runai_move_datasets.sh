#!/usr/bin/env bash
set -e

# =============================================================================
# Move uploaded datasets from home PVC to scratch PVC.
# Run this after scp-ing datasets to /home/<user>/datasets_upload/
#
# Usage:
#   bash scripts/runai_move_datasets.sh <UID_NUMBER> [dataset]   # home code path
#   bash scripts/runai_move_datasets.sh [dataset]                # scratch code path
#
#   dataset   One of: mami | mmhs150k | all  (default: all)
#
# Examples:
#   bash scripts/runai_move_datasets.sh 314305 mami       ← only MAMI
#   bash scripts/runai_move_datasets.sh 314305 mmhs150k   ← only MMHS150K
#   bash scripts/runai_move_datasets.sh 314305            ← both (default)
#   bash scripts/runai_move_datasets.sh mami              ← scratch mode + MAMI
# =============================================================================

if [ "$#" -gt 2 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER> [dataset]   # use /home/\${USER}/hateful_meme_rewriting"
    echo "  bash $0 [dataset]                # use /scratch/hateful_meme_rewriting"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

# --- Path/UID mode selection ---
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
    DATASET="${2:-all}"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
    DATASET="${1:-all}"
fi

case "${DATASET}" in
    mami|mmhs150k|all) ;;
    *)
        echo "ERROR: Invalid dataset '${DATASET}'."
        echo "Allowed values: mami | mmhs150k | all"
        exit 1
        ;;
esac

SCRIPT_PATH="${CODE_ROOT}/scripts/move_datasets_to_scratch.sh"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/scripts/move_datasets_to_scratch.sh" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

JOB_NAME="hmr-move-${DATASET}"

echo "=== Moving datasets to /scratch/ ==="
echo "  User:    ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:    ${MODE_LABEL}"
echo "  Code:    ${CODE_ROOT}"
echo "  Dataset: ${DATASET}"
echo ""

runai submit ${JOB_NAME} \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- bash -c "DATASET=${DATASET} HOME_UPLOAD=/home/${USERNAME}/datasets_upload bash ${SCRIPT_PATH}"

echo "Job submitted. Follow logs with:"
echo "  runai logs ${JOB_NAME} -p course-ee-559-${USERNAME} --follow"
