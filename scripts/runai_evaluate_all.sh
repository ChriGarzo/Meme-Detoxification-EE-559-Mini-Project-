#!/usr/bin/env bash
set -e

# =============================================================================
# Unified evaluation: BART base + finetuned + DetoxLLM + Proxy (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_evaluate_all.sh [<UID_NUMBER>]
#
# Optional environment variables:
#   INPUT_FORMAT       BART encoder input format (default: legacy)
#   CHECKPOINT_SUFFIX  suffix appended to checkpoint dir names (default: "")
#   EVAL_SUFFIX        suffix appended to all output dir names (default: "")
#   PROXY_CHECKPOINT   path to proxy .pt file (default: /scratch/hmr_proxy_checkpoint/best_proxy.pt)
#
# Prerequisites:
#   - Stage 2 Phase 2 (all 4 conditions) completed
#   - Proxy network training completed
# =============================================================================

if [ "$#" -gt 1 ]; then
    echo "Usage: bash $0 [<UID_NUMBER>]"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
INPUT_FORMAT="${INPUT_FORMAT:-explicit_detox}"
TASK_PREFIX="${TASK_PREFIX:-}"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-_explicit_detox}"
EVAL_SUFFIX="${EVAL_SUFFIX:-_explicit_detox}"
PROXY_CHECKPOINT="${PROXY_CHECKPOINT:-/scratch/stages/hmr_proxy_checkpoint_explicit_detox/best_proxy.pt}"
PROXY_TEXT_PROMPT_FORMAT="${PROXY_TEXT_PROMPT_FORMAT:-none_explicit_detox}"
JOB_SUFFIX="${EVAL_SUFFIX//_/-}"

if [ -n "${1:-}" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

JOB_SCRIPT="${CODE_ROOT}/scripts/run_evaluate_all_job.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    if [ -f "${REPO_ROOT_LOCAL}/scripts/run_evaluate_all_job.sh" ]; then
        echo "Note: /scratch not visible on this node; validated local repo at ${REPO_ROOT_LOCAL}."
        CODE_ROOT="/scratch/hateful_meme_rewriting"
        JOB_SCRIPT="${CODE_ROOT}/scripts/run_evaluate_all_job.sh"
    else
        echo "ERROR: Job script not found at: ${JOB_SCRIPT}"
        exit 1
    fi
fi

echo "=== Unified Evaluation ==="
echo "  User:         ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:         ${MODE_LABEL}"
echo "  Code:         ${CODE_ROOT}"
echo "  Input fmt:    ${INPUT_FORMAT}"
echo "  Ckpt suffix:  ${CHECKPOINT_SUFFIX}"
echo "  Eval suffix:  ${EVAL_SUFFIX}"
echo "  Proxy prompt: ${PROXY_TEXT_PROMPT_FORMAT}"
echo "  Image:        ${IMAGE}"
echo ""

runai submit hmr-evaluate-all${JOB_SUFFIX} \
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
    --command -- env \
        INPUT_FORMAT="${INPUT_FORMAT}" \
        TASK_PREFIX="${TASK_PREFIX}" \
        CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX}" \
        EVAL_SUFFIX="${EVAL_SUFFIX}" \
        PROXY_CHECKPOINT="${PROXY_CHECKPOINT}" \
        PROXY_TEXT_PROMPT_FORMAT="${PROXY_TEXT_PROMPT_FORMAT}" \
        bash "${JOB_SCRIPT}" "${CODE_ROOT}"

echo "Evaluation job submitted: hmr-evaluate-all${JOB_SUFFIX}"
echo "Monitor with:"
echo "  runai logs hmr-evaluate-all${JOB_SUFFIX} -p course-ee-559-${USERNAME} --follow"
echo "Results will be at: /scratch/hmr_eval_results${EVAL_SUFFIX}"
