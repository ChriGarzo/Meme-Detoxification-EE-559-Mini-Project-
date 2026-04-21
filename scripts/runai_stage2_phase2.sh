#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 2 Phase 2: BART conditioning fine-tune, all 4 conditions (GPU A100-40G)
#
# Usage: bash scripts/runai_stage2_phase2.sh <UID_NUMBER>
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#
# Example:
#   bash scripts/runai_stage2_phase2.sh 123456
#
# Note: Run this AFTER runai_stage2_phase1.sh has completed.
#       Submits 4 jobs (one per conditioning condition).
#       Conditions: full | target_only | attack_only | none
#         full        — uses all explanation fields: target_group + attack_type + implicit_meaning
#         target_only — uses only the target_group field
#         attack_only — uses only the attack_type field
#         none        — no explanation conditioning (text-only baseline)
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

# Ablation conditions (must match MemeRewriter.format_input and run_stage2.py)
CONDITIONS=("full" "target_only" "attack_only" "none")

echo "=== Stage 2 Phase 2: BART Conditioning Fine-tune ==="
echo "  User:       ${USERNAME} (UID: ${UID_NUM})"
echo "  Group:      ${GROUP_NUM}"
echo "  Conditions: ${CONDITIONS[*]}"
echo "  Image:      ${IMAGE}"
echo ""

for CONDITION in "${CONDITIONS[@]}"; do
    SAFE_CONDITION="${CONDITION//_/-}"
    echo "Submitting Stage 2 Phase 2 for condition: ${CONDITION}"

    runai submit hmr-stage2-phase2-${SAFE_CONDITION} \
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
        --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/training/train_stage2_phase2.py \
            --condition ${CONDITION} \
            --phase1_checkpoint_dir /scratch/hmr_stage2_phase1_checkpoint \
            --dataset_dir /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_stage2_phase2_${CONDITION}_checkpoint \
            --hf_cache /scratch/hf_cache \
            --num_train_epochs 5 \
            --per_device_train_batch_size 8 \
            --learning_rate 2e-5 \
            --warmup_steps 50 \
            --weight_decay 0.01 \
            --seed 42
done

echo ""
echo "All Stage 2 Phase 2 jobs submitted."
