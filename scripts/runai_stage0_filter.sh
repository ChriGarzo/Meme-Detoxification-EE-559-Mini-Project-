#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 0: OCR + CLIP meme filtering (GPU accelerated)
#
# Uses EasyOCR + CLIP — both run significantly faster on GPU.
# For MMHS150K (~150K images) CPU would take many hours; GPU brings it to ~1h.
#
# Usage: bash scripts/runai_stage0_filter.sh <UID_NUMBER> [dataset]
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#               (on the jumphost: ssh <username>@jumphost.rcp.epfl.ch then: id -u)
#   dataset     One of: harmeme | mami | mmhs150k  (default: harmeme)
#
# Example:
#   bash scripts/runai_stage0_filter.sh 123456 harmeme
#   bash scripts/runai_stage0_filter.sh 123456 mami
#   bash scripts/runai_stage0_filter.sh 123456 mmhs150k
#
# Note: USERNAME is taken automatically from $USER (your current Unix username).
#       GROUP_NUM is fixed at 31 for the whole group.
# =============================================================================

# --- Validate args ---
if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER> [dataset]"
    echo "  Get your UID with: id -u"
    exit 1
fi

UID_NUM="$1"
DATASET="${2:-harmeme}"   # default to harmeme if not provided

# --- Configuration (do NOT edit these) ---
USERNAME="${USER}"        # automatically uses your current Unix username
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

echo "=== Stage 0: Meme Filtering (GPU) ==="
echo "  User:    ${USERNAME} (UID: ${UID_NUM})"
echo "  Group:   ${GROUP_NUM}"
echo "  Dataset: ${DATASET}"
echo "  Image:   ${IMAGE}"
echo ""

runai submit hmr-stage0-${DATASET} \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --gpu 1 \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/data/preprocess/filter_meme_images.py \
        --dataset ${DATASET} \
        --images_dir /scratch/hmr_data/${DATASET}/images \
        --output_manifest /scratch/hmr_data/${DATASET}/manifest.csv \
        --hf_cache /scratch/hf_cache \
        --save_examples /scratch/hmr_data/${DATASET}/filter_examples

echo "Stage 0 (${DATASET}) submitted with GPU."
echo "Follow logs with:"
echo "  runai logs hmr-stage0-${DATASET} -p course-ee-559-${USERNAME} --follow"
