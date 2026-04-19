#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 2 Phase 1: BART ParaDetox warm-up (GPU A100-40G)
#
# Usage: bash scripts/runai_stage2_phase1.sh <UID_NUMBER>
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#
# Example:
#   bash scripts/runai_stage2_phase1.sh 123456
#
# Note: Run this AFTER runai_build_stage2_dataset.sh has completed.
#       Pre-trains BART on ParaDetox before conditioning fine-tune.
# =============================================================================

# --- Validate args ---
if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER>"
    echo "  Get your UID with: id -u"
    exit 1
fi

UID_NUM="$1"

# --- Configuration (do NOT edit these) ---
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"

echo "=== Stage 2 Phase 1: BART ParaDetox Warm-up ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Group: ${GROUP_NUM}"
echo "  Image: ${IMAGE}"
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
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/training/train_stage2_phase1.py \
        --output_dir /scratch/hmr_stage2_phase1_checkpoint \
        --hf_cache /scratch/hf_cache \
        --num_train_epochs 2 \
        --per_device_train_batch_size 16 \
        --learning_rate 5e-5 \
        --warmup_steps 500 \
        --weight_decay 0.01 \
        --seed 42

echo "Stage 2 Phase 1 (ParaDetox warm-up) job submitted."
