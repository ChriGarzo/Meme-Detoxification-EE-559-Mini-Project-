#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 1: LLaVA explanation generation + pseudo-rewrites (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_stage1_explain.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_stage1_explain.sh                # scratch code path
#
# Example:
#   bash scripts/runai_stage1_explain.sh 123456
#
# Note: USERNAME is taken automatically from $USER.
#       Submits one job per dataset (harmeme, mami, mmhs150k).
#       Assumes Stage 0 has already been run for all datasets.
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

SCRIPT_PATH="${CODE_ROOT}/inference/run_stage1.py"
# On login nodes, /scratch may not be mounted at that absolute path even though
# it is mounted as /scratch inside the RunAI job container. Validate against the
# local repo as fallback in scratch mode.
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/inference/run_stage1.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Stage 1: LLaVA Explanation Generation ==="
echo "  User:     ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:     ${MODE_LABEL}"
echo "  Code:     ${CODE_ROOT}"
echo "  Group:    ${GROUP_NUM}"
echo "  Input:    /scratch/hmr_data/unified_splits/unified_train.csv (hateful only)"
echo "  Image:    ${IMAGE}"
echo ""

runai submit hmr-stage1 \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools a100-40g \
    --gpu 1 \
    --cpu 4 \
    --memory 32Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- python3 ${SCRIPT_PATH} \
        --dataset train \
        --manifest_path /scratch/hmr_data/unified_splits/unified_train.csv \
        --images_dir /scratch/hmr_data \
        --output_dir /scratch/hmr_stage1_output \
        --hf_cache /scratch/hf_cache \
        --load_in_4bit \
        --hateful_only

echo ""
echo "Stage 1 job submitted. Wait for completion before running build_stage2_dataset."
echo "Follow logs with:"
echo "  runai logs hmr-stage1 -p course-ee-559-${USERNAME} --follow"
