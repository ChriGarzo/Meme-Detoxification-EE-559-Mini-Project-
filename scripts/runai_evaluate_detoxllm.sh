#!/usr/bin/env bash
set -e

# =============================================================================
# DetoxLLM evaluation: run DetoxLLM inference ONLY, then evaluate against
# the already-existing canonical BART results.
#
# Usage:
#   bash scripts/runai_evaluate_detoxllm.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_evaluate_detoxllm.sh                # scratch code path
#
# Example:
#   bash scripts/runai_evaluate_detoxllm.sh 123456
#
# What this does:
#   1. Runs DetoxLLM (UBC-NLP/DetoxLLM-7B) on the 358 validation texts
#      (skip-if-complete: re-submitting is safe).
#   2. Calls evaluate.py pointing at the EXISTING canonical BART dirs —
#      no BART inference is re-run.
#   3. Writes results to /scratch/hmr_eval_results_detoxllm/.
#
# Prerequisites:
#   - runai_evaluate.sh has already completed (canonical BART dirs exist).
#   - UBC-NLP/DetoxLLM-7B accessible via HuggingFace cache at /scratch/hf_cache.
#
# Note: requires an A100-40G GPU because DetoxLLM is a 7B causal LM.
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

JOB_SCRIPT="${CODE_ROOT}/scripts/run_detoxllm_eval_job.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/scripts/run_detoxllm_eval_job.sh" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Job script not found at: ${JOB_SCRIPT}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== DetoxLLM evaluation (DetoxLLM inference only, reuses existing BART results) ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo "  Group: ${GROUP_NUM}"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-evaluate--detoxllm \
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

echo "DetoxLLM evaluation job submitted."
echo "Watch logs with:"
echo "  runai logs hmr-evaluate--detoxllm --follow"
echo "When complete, outputs at:"
echo "  /scratch/hmr_eval_results_detoxllm/"
