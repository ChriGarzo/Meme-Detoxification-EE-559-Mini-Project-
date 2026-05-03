#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 3: Full evaluation (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_evaluate.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_evaluate.sh                # scratch code path
#
# Example:
#   bash scripts/runai_evaluate.sh 123456
#
# Note: Run this AFTER all Stage 2 Phase 2 jobs have completed.
#       Runs inference for:
#         - non-finetuned BART-large under all 4 visual-evidence conditions
#         - finetuned BART-large under all 4 visual-evidence conditions
#       Inference is restricted to /scratch/hmr_stage2_dataset/val.jsonl.
#       Then compares those outputs against the validation target_text
#       pseudo-rewrites.
#       Metrics include text STA and VisualBERT multimodal toxicity.
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

EVAL_SCRIPT="${CODE_ROOT}/evaluation/evaluate.py"
if [ ! -f "${EVAL_SCRIPT}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/evaluation/evaluate.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Evaluation script not found at: ${EVAL_SCRIPT}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

JOB_SCRIPT="${CODE_ROOT}/scripts/run_evaluate_job.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/scripts/run_evaluate_job.sh" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Evaluation job script not found at: ${JOB_SCRIPT}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Stage 3: Full Evaluation ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo "  Group: ${GROUP_NUM}"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-evaluate \
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
    --command -- bash "${JOB_SCRIPT}" "${CODE_ROOT}"

echo "Full evaluation job submitted."
echo "Watch logs with:"
echo "  runai logs hmr-evaluate --follow"
echo "When complete, outputs should be under:"
echo "  /scratch/hmr_eval_results"
