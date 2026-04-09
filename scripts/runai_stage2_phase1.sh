#!/usr/bin/env bash
set -e

# === Configuration (edit these) ===
USERNAME="garzone"
UID_NUM="<YOUR_UID>"      # Get with: ssh garzone@jumphost.rcp.epfl.ch then: id
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

# Stage 2 Phase 1: BART ParaDetox warm-up (GPU A100-40G)
# Pre-trains BART on ParaNMT data before conditioning fine-tune

runai submit hmr-stage2-phase1 \
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
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/training/train_stage2_phase1.py \
        --output_dir /scratch/hmr_stage2_phase1_checkpoint \
        --hf_cache /scratch/hf_cache \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --learning_rate 5e-5 \
        --seed 42

echo "Stage 2 Phase 1 (ParaDetox warm-up) job submitted."
