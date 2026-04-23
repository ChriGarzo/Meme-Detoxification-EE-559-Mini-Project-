#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 4: Train explanation proxy network (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_train_proxy.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_train_proxy.sh                # scratch code path
#
# Example:
#   bash scripts/runai_train_proxy.sh 123456
#
# Note: Run this AFTER Stage 2 Phase 2 (full condition) has completed.
#       Trains a lightweight CLIP→BART-hidden-state MLP to bypass LLaVA at deployment.
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

SCRIPT_PATH="${CODE_ROOT}/training/train_proxy.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/training/train_proxy.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Stage 4: Proxy Network Training ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo "  Group: ${GROUP_NUM}"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-train-proxy \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools a100-40g \
    --gpu 1 \
    --cpu 8 \
    --memory 40Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- python3 ${SCRIPT_PATH} \
        --stage1_output_dir /scratch/hmr_stage1_output \
        --stage2_dataset_dir /scratch/hmr_stage2_dataset \
        --bart_checkpoint_dir /scratch/hmr_stage2_phase2_full_checkpoint \
        --output_dir /scratch/hmr_proxy_checkpoint \
        --hf_cache /scratch/hf_cache \
        --num_train_epochs 20 \
        --batch_size 64 \
        --learning_rate 1e-3 \
        --seed 42

echo "Proxy network training job submitted."
