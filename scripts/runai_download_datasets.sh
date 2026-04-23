#!/usr/bin/env bash
set -e

# =============================================================================
# Dataset Download + Scratch Setup
#
# Run this ONCE before starting any pipeline stage.
# Downloads HarMeme automatically. Prints instructions for MAMI and MMHS150K.
#
# Usage:
#   bash scripts/runai_download_datasets.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_download_datasets.sh                # scratch code path
# =============================================================================

if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER>   # use /home/\${USER}/hateful_meme_rewriting"
    echo "  bash $0                # use /scratch/hateful_meme_rewriting"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

# --- Path/UID mode selection ---
if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

SCRIPT_PATH="${CODE_ROOT}/scripts/setup_scratch.sh"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/scripts/setup_scratch.sh" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Dataset Download + Scratch Setup ==="
echo "  User:   ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:   ${MODE_LABEL}"
echo "  Code:   ${CODE_ROOT}"
echo "  Image:  ${IMAGE}"
echo "  Script: ${SCRIPT_PATH}"
echo ""

runai submit hmr-download-datasets \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- bash ${SCRIPT_PATH}

echo ""
echo "Job submitted. Follow logs with:"
echo "  runai logs hmr-download-datasets -p course-ee-559-${USERNAME} --follow"
