#!/usr/bin/env bash
set -e

# =============================================================================
# Build Stage 2 training dataset from Stage 1 outputs (no GPU needed)
#
# Usage: bash scripts/runai_build_stage2_dataset.sh <UID_NUMBER>
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#
# Example:
#   bash scripts/runai_build_stage2_dataset.sh 123456
#
# Note: Run this AFTER all Stage 1 jobs have completed.
#       Combines all per-dataset stage1 outputs into a single train/val split
#       stored at /scratch/hmr_stage2_dataset/ (shared across the group).
# =============================================================================

# --- Validate args ---
if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER>"
    echo "  Get your UID with: id -u"
    exit 1
fi

UID_NUM="$1"

# --- Configuration (do NOT edit these) ---
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"

echo "=== Build Stage 2 Dataset ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Group: ${GROUP_NUM}"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-build-stage2-dataset \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --cpu-request 4 \
    --memory-request 16Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/data/preprocess/build_stage2_dataset.py \
        --stage1_dir /scratch/hmr_stage1_output \
        --output_dir /scratch/hmr_stage2_dataset \
        --hf_cache /scratch/hf_cache

echo "Build Stage 2 dataset job submitted."
