#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 0: OCR + CLIP meme filtering (GPU accelerated)
#
# Uses EasyOCR + CLIP — both run significantly faster on GPU.
# For MMHS150K (~150K images) CPU would take many hours; GPU brings it to ~1h.
#
# Usage:
#   bash scripts/runai_stage0_filter.sh <UID_NUMBER> [dataset]   # home code path
#   bash scripts/runai_stage0_filter.sh [dataset]                # scratch code path
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#               (on the jumphost: ssh <username>@jumphost.rcp.epfl.ch then: id -u)
#   dataset     One of: harmeme | mami | mmhs150k  (default: harmeme)
#
# Example:
#   bash scripts/runai_stage0_filter.sh 123456 harmeme
#   bash scripts/runai_stage0_filter.sh 123456 mami
#   bash scripts/runai_stage0_filter.sh 123456 mmhs150k
#   bash scripts/runai_stage0_filter.sh mmhs150k
#
# Note: USERNAME is taken automatically from $USER (your current Unix username).
#       GROUP_NUM is fixed at 31 for the whole group.
# =============================================================================

# --- Validate args ---
if [ "$#" -gt 2 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER> [dataset]   # use /home/\${USER}/hateful_meme_rewriting"
    echo "  bash $0 [dataset]                # use /scratch/hateful_meme_rewriting"
    exit 1
fi

# --- Configuration (do NOT edit these) ---
USERNAME="${USER}"        # automatically uses your current Unix username
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

# --- Path/UID mode selection ---
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
    DATASET="${2:-harmeme}"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
    DATASET="${1:-harmeme}"
fi

case "${DATASET}" in
    harmeme|mami|mmhs150k) ;;
    *)
        echo "ERROR: Invalid dataset '${DATASET}'."
        echo "Allowed values: harmeme | mami | mmhs150k"
        exit 1
        ;;
esac

SCRIPT_PATH="${CODE_ROOT}/data/preprocess/filter_meme_images.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/data/preprocess/filter_meme_images.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Stage 0: Meme Filtering (GPU) ==="
echo "  User:    ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:    ${MODE_LABEL}"
echo "  Code:    ${CODE_ROOT}"
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
    --command -- python3 ${SCRIPT_PATH} \
        --dataset ${DATASET} \
        --images_dir /scratch/hmr_data/${DATASET}/images \
        --output_manifest /scratch/hmr_data/${DATASET}/manifest.csv \
        --hf_cache /scratch/hf_cache

echo "Stage 0 (${DATASET}) submitted with GPU."
echo "Follow logs with:"
echo "  runai logs hmr-stage0-${DATASET} -p course-ee-559-${USERNAME} --follow"
