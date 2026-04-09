#!/usr/bin/env bash
set -e

# === Configuration (edit these) ===
USERNAME="garzone"
UID_NUM="<YOUR_UID>"      # Get with: ssh garzone@jumphost.rcp.epfl.ch then: id
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

# Stage 0: OCR + CLIP meme filtering (no GPU needed)
# Run once per dataset: change --dataset flag

DATASET="${1:-harmeme}"  # Pass dataset name as argument

runai submit hmr-stage0-${DATASET} \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/data/preprocess/filter_meme_images.py \
        --dataset ${DATASET} \
        --images_dir /scratch/hmr_data/${DATASET}/images \
        --output_manifest /scratch/hmr_data/${DATASET}/manifest.csv \
        --hf_cache /scratch/hf_cache
