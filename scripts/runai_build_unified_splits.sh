#!/usr/bin/env bash
set -e

# =============================================================================
# Build unified 80/10/10 stratified splits across HarMeme, MAMI, MMHS150K.
#
# IMPORTANT: Run AFTER Stage 0 has completed for ALL THREE datasets:
#   bash scripts/runai_stage0_filter.sh <UID> harmeme
#   bash scripts/runai_stage0_filter.sh <UID> mami
#   bash scripts/runai_stage0_filter.sh <UID> mmhs150k
#
# The script reads per-dataset Stage 0 manifests from the default paths:
#   /scratch/hmr_data/harmeme/manifest.csv
#   /scratch/hmr_data/mami/manifest.csv
#   /scratch/hmr_data/mmhs150k/manifest.csv
#
# Only images that passed Stage 0 filtering (kept=True) are included in splits.
#
# Usage: bash scripts/runai_build_unified_splits.sh <UID_NUMBER>
# =============================================================================

if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER>"
    exit 1
fi

UID_NUM="$1"
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

echo "=== Build Unified Splits (post-Stage-0) ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Reads Stage 0 manifests from /scratch/hmr_data/<dataset>/manifest.csv"
echo ""

runai submit hmr-build-unified-splits \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- python3 \
        /home/${USERNAME}/hateful_meme_rewriting/data/preprocess/build_unified_splits.py \
        --harmeme_dir           /scratch/hmr_data/harmeme \
        --mami_dir              /scratch/hmr_data/mami \
        --mmhs150k_dir          /scratch/hmr_data/mmhs150k \
        --harmeme_manifest      /scratch/hmr_data/harmeme/manifest.csv \
        --mami_manifest         /scratch/hmr_data/mami/manifest.csv \
        --mmhs150k_manifest     /scratch/hmr_data/mmhs150k/manifest.csv \
        --output_dir            /scratch/hmr_data/unified_splits \
        --train_ratio           0.80 \
        --val_ratio             0.10 \
        --seed                  42

echo ""
echo "Job submitted. Follow logs with:"
echo "  runai logs hmr-build-unified-splits -p course-ee-559-${USERNAME} --follow"
