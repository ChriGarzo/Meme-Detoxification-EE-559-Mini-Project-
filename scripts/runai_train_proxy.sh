#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 4: Train explanation proxy network (GPU A100-40G)
#
# Usage: bash scripts/runai_train_proxy.sh <UID_NUMBER>
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#
# Example:
#   bash scripts/runai_train_proxy.sh 123456
#
# Note: Run this AFTER Stage 2 Phase 2 (full condition) has completed.
#       Trains a lightweight CLIP→BART-hidden-state MLP to bypass LLaVA at deployment.
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

echo "=== Stage 4: Proxy Network Training ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
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
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/training/train_proxy.py \
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
