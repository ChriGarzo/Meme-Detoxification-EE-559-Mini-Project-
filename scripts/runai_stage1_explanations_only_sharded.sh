#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 1 (explanations-only + sharded)
#
# Usage:
#   SHARD_ID=0 NUM_SHARDS=8 bash scripts/runai_stage1_explanations_only_sharded.sh <UID_NUMBER>
#   SHARD_ID=0 NUM_SHARDS=8 bash scripts/runai_stage1_explanations_only_sharded.sh
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
NUM_SHARDS="${NUM_SHARDS:-8}"
SHARD_ID="${SHARD_ID:-0}"
EXPLAIN_MAX_RETRIES="${EXPLAIN_MAX_RETRIES:-0}"

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

SCRIPT_PATH="${CODE_ROOT}/inference/run_stage1_explanations_only_sharded.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/inference/run_stage1_explanations_only_sharded.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

JOB_NAME="hmr-stage1-exp-s${SHARD_ID}"

echo "=== Stage 1: Explanations Only (Sharded) ==="
echo "  User:      ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:      ${MODE_LABEL}"
echo "  Code:      ${CODE_ROOT}"
echo "  Group:     ${GROUP_NUM}"
echo "  Input:     /scratch/hmr_data/unified_splits/unified_train.csv"
echo "  Image:     ${IMAGE}"
echo "  Job:       ${JOB_NAME}"
echo "  Shard:     ${SHARD_ID}/${NUM_SHARDS}"
echo "  Batch:     ${STAGE1_BATCH_SIZE}"
echo "  Retries:   ${EXPLAIN_MAX_RETRIES}"
echo ""

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
    --command -- python3 ${SCRIPT_PATH} \
        --dataset train \
        --manifest_path /scratch/hmr_data/unified_splits/unified_train.csv \
        --images_dir /scratch/hmr_data \
        --output_dir /scratch/hmr_stage1_output \
        --hf_cache /scratch/hf_cache \
        --batch_size ${STAGE1_BATCH_SIZE} \
        --num_shards ${NUM_SHARDS} \
        --shard_id ${SHARD_ID} \
        --explain_max_retries ${EXPLAIN_MAX_RETRIES} \
        --load_in_4bit \
        --hateful_only

echo ""
echo "Explanations-only shard job submitted."
echo "Follow logs with:"
echo "  runai logs ${JOB_NAME} -p course-ee-559-${USERNAME} --follow"
