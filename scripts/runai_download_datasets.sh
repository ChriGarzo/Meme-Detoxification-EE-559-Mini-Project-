#!/usr/bin/env bash
set -e

# =============================================================================
# Dataset Download + Scratch Setup
#
# Run this ONCE before starting any pipeline stage.
# Downloads HarMeme automatically. Prints instructions for MAMI and MMHS150K.
#
# Usage: bash scripts/runai_download_datasets.sh <UID_NUMBER>
# =============================================================================

if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER>"
    echo "  Get your UID with: id -u"
    exit 1
fi

UID_NUM="$1"
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
SCRIPT_PATH="/home/${USERNAME}/hateful_meme_rewriting/scripts/setup_scratch.sh"

echo "=== Dataset Download + Scratch Setup ==="
echo "  User:   ${USERNAME} (UID: ${UID_NUM})"
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
