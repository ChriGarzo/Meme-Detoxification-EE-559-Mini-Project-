#!/usr/bin/env bash
set -e

# =============================================================================
# Build unified 80/10/10 stratified splits across HarMeme, MAMI, MMHS150K.
#
# IMPORTANT: Run AFTER Stage 0 has completed for ALL THREE datasets:
#   bash scripts/runai_stage0_filter.sh <UID> harmeme
#   bash scripts/runai_stage0_filter.sh <UID> mami
#   bash scripts/runai_stage0_filter.sh <UID> mmhs150k
#
# The script reads per-dataset Stage 0 manifests from the default paths:
#   /scratch/hmr_data/harmeme/manifest.csv
#   /scratch/hmr_data/mami/manifest.csv
#   /scratch/hmr_data/mmhs150k/manifest.csv
#
# Only images that passed Stage 0 filtering (kept=True) are included in splits.
#
# Usage:
#   bash scripts/runai_build_unified_splits.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_build_unified_splits.sh                # scratch code path
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

SCRIPT_PATH="${CODE_ROOT}/data/preprocess/build_unified_splits.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/data/preprocess/build_unified_splits.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Build Unified Splits (post-Stage-0) ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo "  Reads Stage 0 manifests from /scratch/hmr_data/<dataset>/manifest.csv"
echo ""

runai submit hmr-build-unified-splits \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- python3 \
        ${SCRIPT_PATH} \
        --harmeme_dir           /scratch/hmr_data/harmeme \
        --mami_dir              /scratch/hmr_data/mami \
        --mmhs150k_dir          /scratch/hmr_data/mmhs150k \
        --harmeme_manifest      /scratch/hmr_data/harmeme/manifest.csv \
        --mami_manifest         /scratch/hmr_data/mami/manifest.csv \
        --mmhs150k_manifest     /scratch/hmr_data/mmhs150k/manifest.csv \
        --output_dir            /scratch/hmr_data/unified_splits \
        --train_ratio           0.80 \
        --val_ratio             0.10 \
        --seed                  42

echo ""
echo "Job submitted. Follow logs with:"
echo "  runai logs hmr-build-unified-splits -p course-ee-559-${USERNAME} --follow"
