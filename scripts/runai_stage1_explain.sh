#!/usr/bin/env bash
set -e

# =============================================================================
# Stage 1: LLaVA explanation generation + pseudo-rewrites (GPU A100-40G)
#
# Usage: bash scripts/runai_stage1_explain.sh <UID_NUMBER>
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#
# Example:
#   bash scripts/runai_stage1_explain.sh 123456
#
# Note: USERNAME is taken automatically from $USER.
#       Submits one job per dataset (harmeme, mami, mmhs150k).
#       Assumes Stage 0 has already been run for all datasets.
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

# Datasets to process (must match Stage 0 runs)
DATASETS=("harmeme" "mami" "mmhs150k")

echo "=== Stage 1: LLaVA Explanation Generation ==="
echo "  User:     ${USERNAME} (UID: ${UID_NUM})"
echo "  Group:    ${GROUP_NUM}"
echo "  Datasets: ${DATASETS[*]}"
echo "  Image:    ${IMAGE}"
echo ""

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

echo ""
echo "All Stage 1 jobs submitted. Wait for completion before running build_stage2_dataset."
