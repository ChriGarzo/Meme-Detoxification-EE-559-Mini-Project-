#!/usr/bin/env bash
set -e

# === Configuration (edit these) ===
USERNAME="garzone"
UID_NUM="<YOUR_UID>"      # Get with: ssh garzone@jumphost.rcp.epfl.ch then: id
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

# Stage 1: LLaVA inference (GPU A100-40G)
# Loops over all datasets in DATASETS array

DATASETS=("harmeme" "reddit" "twitter" "other")  # Adjust as needed

for DATASET in "${DATASETS[@]}"; do
    echo "Submitting Stage 1 for dataset: ${DATASET}"

    runai submit hmr-stage1-${DATASET} \
        --run-as-uid ${UID_NUM} \
        --image ${IMAGE} \
        --node-pools a100-40g \
        --gpu 1 \
        --cpu-request 4 \
        --memory-request 32Gi \
        --existing-pvc claimname=home,path=/home/${USERNAME} \
        --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
        --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
        --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
        --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/inference/run_stage1.py \
            --dataset ${DATASET} \
            --manifest_path /scratch/hmr_data/${DATASET}/manifest.csv \
            --images_dir /scratch/hmr_data/${DATASET}/images \
            --output_dir /scratch/hmr_stage1_output/${DATASET} \
            --hf_cache /scratch/hf_cache
done

echo "All Stage 1 jobs submitted."
