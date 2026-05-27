#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 4: Train explanation proxy network (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_train_proxy.sh <UID_NUMBER>
#   bash scripts/runai_train_proxy.sh
#
# Example:
#   bash scripts/runai_train_proxy.sh 123456
#
# Note: Run this AFTER Stage 2 Phase 2 (full condition) has completed.
#       Trains a lightweight CLIP→BART soft-token MLP to bypass LLaVA at deployment.
#       Set PROXY_NUM_SOFT_TOKENS=32 to try a longer proxy encoder memory.
#
# Optional environment variables:
#   BART_SUFFIX    suffix appended to the BART checkpoint dir name (default: "")
#                  e.g. BART_SUFFIX=_explicit_detox reads hmr_stage2_phase2_full_explicit_detox_checkpoint
#   PROXY_SUFFIX   suffix appended to the proxy output dir name (default: same as BART_SUFFIX)
#                  e.g. PROXY_SUFFIX=_explicit_detox writes hmr_proxy_checkpoint_explicit_detox
# =============================================================================

# --- Validate args ---
if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER>"
    echo "  bash $0"
    exit 1
fi

# --- Configuration (do NOT edit these) ---
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
PROXY_NUM_SOFT_TOKENS="${PROXY_NUM_SOFT_TOKENS:-16}"
BART_SUFFIX="${BART_SUFFIX:-}"
PROXY_SUFFIX="${PROXY_SUFFIX:-${BART_SUFFIX}}"
INPUT_FORMAT="${INPUT_FORMAT:-legacy}"
TASK_PREFIX="${TASK_PREFIX:-}"
JOB_SUFFIX="${PROXY_SUFFIX//_/-}"
TASK_PREFIX_ARGS=()
if [ -n "${TASK_PREFIX}" ]; then
    TASK_PREFIX_ARGS=(--task_prefix "${TASK_PREFIX}")
fi

# --- Path/UID mode selection ---
if [ -n "$1" ]; then
    UID_NUM="$1"
else
    UID_NUM="$(id -u)"
fi

# If this launcher is run from the shared scratch checkout, use the scratch
# checkout inside the RunAI pod even when UID is provided. The personal home
# checkout may exist but be stale.
if [[ "${REPO_ROOT_LOCAL}" == /mnt/course-ee-559/* || "${REPO_ROOT_LOCAL}" == /scratch/* ]]; then
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
else
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
fi

SCRIPT_PATH="${CODE_ROOT}/training/train_proxy.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ -f "${REPO_ROOT_LOCAL}/training/train_proxy.py" ]; then
        echo "Note: ${SCRIPT_PATH} is not visible on this node."
        echo "      Using the shared scratch code path inside the RunAI pod: /scratch/hateful_meme_rewriting"
        CODE_ROOT="/scratch/hateful_meme_rewriting"
        SCRIPT_PATH="${CODE_ROOT}/training/train_proxy.py"
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
echo "  Soft tokens: ${PROXY_NUM_SOFT_TOKENS}"
echo "  Input fmt:   ${INPUT_FORMAT}"
echo "  BART suffix: ${BART_SUFFIX}"
echo "  Proxy suffix:${PROXY_SUFFIX}"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-train-proxy${JOB_SUFFIX} \
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
        --stage1_output_dir /scratch/stages/hmr_stage1_output \
        --stage2_dataset_dir /scratch/stages/hmr_stage2_dataset \
        --bart_checkpoint_dir /scratch/stages/hmr_stage2_phase2_full${BART_SUFFIX}_checkpoint \
        --output_dir /scratch/stages/hmr_proxy_checkpoint${PROXY_SUFFIX} \
        --hf_cache /scratch/hf_cache \
        --num_train_epochs 20 \
        --batch_size 64 \
        --learning_rate 1e-3 \
        --num_soft_tokens ${PROXY_NUM_SOFT_TOKENS} \
        --input_format ${INPUT_FORMAT} \
        "${TASK_PREFIX_ARGS[@]}" \
        --seed 42

echo "Proxy network training job submitted."
