#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 1 (rewrites-only + sharded)
#
# Reads existing explanations shard and generates pseudo-rewrites only.
#
# Usage:
#   SHARD_ID=0 NUM_SHARDS=8 bash scripts/runai_stage1_rewrites_only_sharded.sh <UID_NUMBER>
#   SHARD_ID=0 NUM_SHARDS=8 bash scripts/runai_stage1_rewrites_only_sharded.sh
#
# Optional env vars:
#   EXPLANATIONS_PATH=/scratch/hmr_stage1_output/train_explanations_shard00of08.jsonl
#   STAGE1_BATCH_SIZE=4
#   SCORE_BATCH_SIZE=8
#   REWRITE_MAX_ATTEMPTS=2
#   REWRITE_CANDIDATES_PER_ATTEMPT=3
#   REWRITE_TEMPERATURE=0.75
#   REWRITE_TOP_P=0.92
#   STA_THRESHOLD=0.45
#   BERTSCORE_MIN=0.25
#   BERTSCORE_MAX=1.0
#   MIN_LEXICAL_CHANGE=0.0
#   MAX_CHAR_SIMILARITY=1.0
#   MIN_TOXICITY_DROP=0.0
#   MIN_SOURCE_TOXICITY_FOR_DROP=0.2
# =============================================================================

if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER>"
    echo "  bash $0"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-4}"
SCORE_BATCH_SIZE="${SCORE_BATCH_SIZE:-8}"
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_ID="${SHARD_ID:-0}"
MULTIMODAL_MODEL_NAME="${MULTIMODAL_MODEL_NAME:-chiragmittal92/visualbert-hateful-memes-finetuned-model}"
REWRITE_MAX_ATTEMPTS="${REWRITE_MAX_ATTEMPTS:-2}"
REWRITE_CANDIDATES_PER_ATTEMPT="${REWRITE_CANDIDATES_PER_ATTEMPT:-4}"
REWRITE_TEMPERATURE="${REWRITE_TEMPERATURE:-0.75}"
REWRITE_TOP_P="${REWRITE_TOP_P:-0.92}"

STA_THRESHOLD="${STA_THRESHOLD:-0.25}"
BERTSCORE_MIN="${BERTSCORE_MIN:-0.15}"
BERTSCORE_MAX="${BERTSCORE_MAX:-1.0}"
MIN_LEXICAL_CHANGE="${MIN_LEXICAL_CHANGE:-0.0}"
MAX_CHAR_SIMILARITY="${MAX_CHAR_SIMILARITY:-1.0}"
MIN_TOXICITY_DROP="${MIN_TOXICITY_DROP:-0.05}"
MIN_SOURCE_TOXICITY_FOR_DROP="${MIN_SOURCE_TOXICITY_FOR_DROP:-0.2}"
EXPLANATIONS_PATH="${EXPLANATIONS_PATH:-}"

if [ "${NUM_SHARDS}" -lt 1 ]; then
    echo "ERROR: NUM_SHARDS must be >= 1 (got ${NUM_SHARDS})"
    exit 1
fi
if [ "${SHARD_ID}" -lt 0 ] || [ "${SHARD_ID}" -ge "${NUM_SHARDS}" ]; then
    echo "ERROR: SHARD_ID must be in [0, NUM_SHARDS-1] (got SHARD_ID=${SHARD_ID}, NUM_SHARDS=${NUM_SHARDS})"
    exit 1
fi

if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

SCRIPT_PATH="${CODE_ROOT}/inference/run_stage1_rewrites_only_sharded.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/inference/run_stage1_rewrites_only_sharded.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

JOB_NAME="hmr-stage1-rw-s${SHARD_ID}"

echo "=== Stage 1: Rewrites Only (Sharded) ==="
echo "  User:      ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:      ${MODE_LABEL}"
echo "  Code:      ${CODE_ROOT}"
echo "  Group:     ${GROUP_NUM}"
echo "  Input:     /scratch/hmr_stage1_output/train_explanations_shardXXofYY.jsonl"
echo "  Image:     ${IMAGE}"
echo "  Job:       ${JOB_NAME}"
echo "  Shard:     ${SHARD_ID}/${NUM_SHARDS}"
echo "  Batches:   gen=${STAGE1_BATCH_SIZE}, score=${SCORE_BATCH_SIZE}"
echo "  Rewrite:   attempts=${REWRITE_MAX_ATTEMPTS}, candidates=${REWRITE_CANDIDATES_PER_ATTEMPT}, temp=${REWRITE_TEMPERATURE}, top_p=${REWRITE_TOP_P}"
echo "  MM model:  ${MULTIMODAL_MODEL_NAME}"
if [ -n "${EXPLANATIONS_PATH}" ]; then
  echo "  Exp path:  ${EXPLANATIONS_PATH}"
fi
echo ""

CMD=(
  python3 "${SCRIPT_PATH}"
  --dataset train
  --images_dir /scratch/hmr_data
  --output_dir /scratch/hmr_stage1_output
  --hf_cache /scratch/hf_cache
  --multimodal_model_name "${MULTIMODAL_MODEL_NAME}"
  --batch_size "${STAGE1_BATCH_SIZE}"
  --score_batch_size "${SCORE_BATCH_SIZE}"
  --num_shards "${NUM_SHARDS}"
  --shard_id "${SHARD_ID}"
  --sta_threshold "${STA_THRESHOLD}"
  --bertscore_min "${BERTSCORE_MIN}"
  --bertscore_max "${BERTSCORE_MAX}"
  --min_lexical_change "${MIN_LEXICAL_CHANGE}"
  --max_char_similarity "${MAX_CHAR_SIMILARITY}"
  --min_toxicity_drop "${MIN_TOXICITY_DROP}"
  --min_source_toxicity_for_drop "${MIN_SOURCE_TOXICITY_FOR_DROP}"
  --rewrite_max_attempts "${REWRITE_MAX_ATTEMPTS}"
  --rewrite_candidates_per_attempt "${REWRITE_CANDIDATES_PER_ATTEMPT}"
  --rewrite_temperature "${REWRITE_TEMPERATURE}"
  --rewrite_top_p "${REWRITE_TOP_P}"
  --load_in_4bit
  --hateful_only
)

if [ -n "${EXPLANATIONS_PATH}" ]; then
  CMD+=(--explanations_path "${EXPLANATIONS_PATH}")
fi

runai submit "${JOB_NAME}" \
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
    --command -- "${CMD[@]}"

echo ""
echo "Rewrites-only shard job submitted."
echo "Follow logs with:"
echo "  runai logs ${JOB_NAME} -p course-ee-559-${USERNAME} --follow"
