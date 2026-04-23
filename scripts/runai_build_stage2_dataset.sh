#!/usr/bin/env bash
set -e

# =============================================================================
# Build Stage 2 training dataset from Stage 1 outputs (no GPU needed)
#
# Usage:
#   bash scripts/runai_build_stage2_dataset.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_build_stage2_dataset.sh                # scratch code path
#
# Example:
#   bash scripts/runai_build_stage2_dataset.sh 123456
#
# Note: Run this AFTER all Stage 1 jobs have completed.
#       Combines all per-dataset stage1 outputs into a single train/val split
#       stored at /scratch/hmr_stage2_dataset/ (shared across the group).
# =============================================================================

# --- Validate args ---
if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER>   # use /home/\${USER}/hateful_meme_rewriting"
    echo "  bash $0                # use /scratch/hateful_meme_rewriting"
    exit 1
fi

# --- Configuration (do NOT edit these) ---
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

SCRIPT_PATH="${CODE_ROOT}/data/preprocess/build_stage2_dataset.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/data/preprocess/build_stage2_dataset.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Build Stage 2 Dataset ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo "  Group: ${GROUP_NUM}"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-build-stage2-dataset \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --cpu 4 \
    --memory 16Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- python3 ${SCRIPT_PATH} \
        --stage1_dir /scratch/hmr_stage1_output \
        --output_dir /scratch/hmr_stage2_dataset \
        --hf_cache /scratch/hf_cache

echo "Build Stage 2 dataset job submitted."
