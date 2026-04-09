#!/usr/bin/env bash
set -e

# === Configuration (edit these) ===
USERNAME="garzone"
UID_NUM="<YOUR_UID>"      # Get with: ssh garzone@jumphost.rcp.epfl.ch then: id
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

# Stage 2 Phase 2: BART conditioning fine-tune (GPU A100-40G)
# Loops over all 4 conditions to train separate models

CONDITIONS=("text_only" "image_text" "image_embedding" "full")

for CONDITION in "${CONDITIONS[@]}"; do
    echo "Submitting Stage 2 Phase 2 for condition: ${CONDITION}"

    runai submit hmr-stage2-phase2-${CONDITION} \
        --run-as-uid ${UID_NUM} \
        --image ${IMAGE} \
        --node-pools a100-40g \
        --gpu 1 \
        --cpu-request 8 \
        --memory-request 40Gi \
        --existing-pvc claimname=home,path=/home/${USERNAME} \
        --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
        --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
        --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
        --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/training/train_stage2_phase2.py \
            --condition ${CONDITION} \
            --checkpoint_dir /scratch/hmr_stage2_phase1_checkpoint \
            --dataset_path /scratch/hmr_stage2_dataset \
            --output_dir /scratch/hmr_stage2_phase2_${CONDITION}_checkpoint \
            --hf_cache /scratch/hf_cache \
            --num_train_epochs 5 \
            --per_device_train_batch_size 8 \
            --learning_rate 2e-5 \
            --seed 42
done

echo "All Stage 2 Phase 2 jobs submitted."
