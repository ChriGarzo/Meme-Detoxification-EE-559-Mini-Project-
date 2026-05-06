#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 5: Proxy + BART-full evaluation (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_evaluate_proxy.sh <UID_NUMBER>             # explicit detox prompt
#   bash scripts/runai_evaluate_proxy.sh <UID_NUMBER> legacy      # legacy none prompt
#   bash scripts/runai_evaluate_proxy.sh                          # explicit detox prompt, current UID
#
# Optional experiment switches:
#   CHECKPOINT_SUFFIX=_explicit_detox EVAL_SUFFIX=_explicit_detox_proxy bash scripts/runai_evaluate_proxy.sh <UID>
#   PROXY_TEXT_PROMPT_FORMAT=none_legacy EVAL_SUFFIX=_proxy bash scripts/runai_evaluate_proxy.sh <UID>
# =============================================================================

if [ "$#" -gt 2 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER> [explicit|legacy]"
    echo "  bash $0 [explicit|legacy]"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-}"

MODE="${2:-}"
if [ -n "${1:-}" ] && [[ ! "$1" =~ ^[0-9]+$ ]]; then
    MODE="$1"
    UID_NUM="$(id -u)"
elif [ -n "${1:-}" ]; then
    UID_NUM="$1"
else
    UID_NUM="$(id -u)"
fi

case "${MODE:-explicit}" in
    explicit|none_explicit_detox)
        DEFAULT_PROMPT_FORMAT="none_explicit_detox"
        DEFAULT_EVAL_SUFFIX="_proxy_text_explicit"
        ;;
    legacy|none_legacy)
        DEFAULT_PROMPT_FORMAT="none_legacy"
        DEFAULT_EVAL_SUFFIX="_proxy"
        ;;
    *)
        echo "ERROR: Unknown proxy prompt mode: ${MODE}"
        echo "Use one of: explicit, legacy"
        exit 1
        ;;
esac

PROXY_TEXT_PROMPT_FORMAT="${PROXY_TEXT_PROMPT_FORMAT:-${DEFAULT_PROMPT_FORMAT}}"
EVAL_SUFFIX="${EVAL_SUFFIX:-${DEFAULT_EVAL_SUFFIX}}"
JOB_SUFFIX="${EVAL_SUFFIX//_/-}"

if [[ "${REPO_ROOT_LOCAL}" == /mnt/course-ee-559/* || "${REPO_ROOT_LOCAL}" == /scratch/* ]]; then
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
else
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
fi

JOB_SCRIPT="${CODE_ROOT}/scripts/run_proxy_evaluate_job.sh"
if [ ! -f "${JOB_SCRIPT}" ]; then
    if [ -f "${REPO_ROOT_LOCAL}/scripts/run_proxy_evaluate_job.sh" ]; then
        echo "Note: ${JOB_SCRIPT} is not visible on this node."
        echo "      Using the shared scratch code path inside the RunAI pod: /scratch/hateful_meme_rewriting"
        CODE_ROOT="/scratch/hateful_meme_rewriting"
        JOB_SCRIPT="${CODE_ROOT}/scripts/run_proxy_evaluate_job.sh"
    else
        echo "ERROR: Proxy evaluation job script not found at: ${JOB_SCRIPT}"
        exit 1
    fi
fi

echo "=== Proxy+BART Evaluation ==="
echo "  User:        ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:        ${MODE_LABEL}"
echo "  Code:        ${CODE_ROOT}"
echo "  Ckpt suffix: ${CHECKPOINT_SUFFIX}"
echo "  Eval suffix: ${EVAL_SUFFIX}"
echo "  Text prompt: ${PROXY_TEXT_PROMPT_FORMAT}"
echo "  Image:       ${IMAGE}"
echo ""

runai submit hmr-evaluate-proxy${JOB_SUFFIX} \
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
        CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX}" \
        EVAL_SUFFIX="${EVAL_SUFFIX}" \
        PROXY_TEXT_PROMPT_FORMAT="${PROXY_TEXT_PROMPT_FORMAT}" \
        bash "${JOB_SCRIPT}" "${CODE_ROOT}"

echo "Proxy+BART evaluation job submitted."
echo "Watch logs with:"
echo "  runai logs hmr-evaluate-proxy${JOB_SUFFIX} --follow"
echo "When complete, outputs should be under:"
echo "  /scratch/hmr_eval_results_proxy${EVAL_SUFFIX}"
