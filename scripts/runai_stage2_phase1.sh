#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 2 Phase 1: BART ParaDetox warm-up (GPU A100-40G)
#
# Usage:
#   bash scripts/runai_stage2_phase1.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_stage2_phase1.sh                # scratch code path
#
# Example:
#   bash scripts/runai_stage2_phase1.sh 123456
#
# Note: Run this BEFORE runai_stage2_phase2.sh.
#       Submits a single job that fine-tunes facebook/bart-large on the
#       full s-nlp/paradetox dataset.
#
#       The resulting checkpoint at /scratch/hmr_stage2_phase1_checkpoint
#       is passed to Phase 2 via --phase1_checkpoint_dir, giving the model
#       a detoxification prior before meme-specific conditioning.
#
#       Output: /scratch/hmr_stage2_phase1_checkpoint
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

SCRIPT_PATH="${CODE_ROOT}/training/train_stage2_phase1.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/training/train_stage2_phase1.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Stage 2 Phase 1: BART ParaDetox Warm-up ==="
echo "  User:    ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:    ${MODE_LABEL}"
echo "  Code:    ${CODE_ROOT}"
echo "  Group:   ${GROUP_NUM}"
echo "  Image:   ${IMAGE}"
echo "  Output:  /scratch/hmr_stage2_phase1_checkpoint"
echo ""

runai submit hmr-stage2-phase1 \
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
        --output_dir      /scratch/hmr_stage2_phase1_checkpoint \
        --hf_cache        /scratch/hf_cache \
        --num_train_epochs            3 \
        --per_device_train_batch_size 16 \
        --learning_rate   1e-5 \
        --warmup_steps    1000 \
        --weight_decay    0.01 \
        --bertscore_min   0.5 \
        --seed            42

echo ""
echo "Phase 1 job submitted: hmr-stage2-phase1"
echo "Monitor with:  runai logs hmr-stage2-phase1 --tail 50"
echo "Check status:  runai get hmr-stage2-phase1"
echo ""
echo "Once complete, run Phase 2 with:"
echo "  bash scripts/runai_stage2_phase2.sh ${UID_NUM}"
