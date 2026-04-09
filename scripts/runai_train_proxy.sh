#!/usr/bin/env bash
set -e

# === Configuration (edit these) ===
USERNAME="garzone"
UID_NUM="<YOUR_UID>"      # Get with: ssh garzone@jumphost.rcp.epfl.ch then: id
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

# Proxy network training (GPU A100-40G)
# Trains lightweight proxy model to predict Stage 2 outputs without LLaVA

runai submit hmr-train-proxy \
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
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/training/train_proxy.py \
        --stage1_output_dir /scratch/hmr_stage1_output \
        --stage2_data /scratch/hmr_stage2_dataset \
        --output_dir /scratch/hmr_proxy_checkpoint \
        --hf_cache /scratch/hf_cache \
        --num_train_epochs 10 \
        --per_device_train_batch_size 16 \
        --learning_rate 1e-4 \
        --seed 42

echo "Proxy network training job submitted."
