#!/usr/bin/env bash
set -e

# =============================================================================
# Sample kept / discarded images from Stage 0 filter manifests.
#
# Run this after ALL three Stage 0 jobs have completed.
# Copies 50 kept and 50 discarded images per dataset into:
#
#   /scratch/hmr_data/filtering_results/
#       kept/
#           harmeme/
#           mami/
#           mmhs150k/
#       discarded/
#           harmeme/
#           mami/
#           mmhs150k/
#
# You can then scp the whole filtering_results/ folder to your laptop.
#
# Usage: bash scripts/runai_sample_filter_examples.sh <UID_NUMBER>
#
# Example:
#   bash scripts/runai_sample_filter_examples.sh 123456
# =============================================================================

if [ -z "$1" ]; then
    echo "ERROR: Missing UID_NUMBER argument."
    echo "Usage: bash $0 <UID_NUMBER>"
    echo "  Get your UID with: id -u"
    exit 1
fi

UID_NUM="$1"
USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"

echo "=== Sample Filter Examples ==="
echo "  User:   ${USERNAME} (UID: ${UID_NUM})"
echo "  Output: /scratch/hmr_data/filtering_results/"
echo ""

runai submit hmr-sample-filter-examples \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- python3 /home/${USERNAME}/hateful_meme_rewriting/data/preprocess/sample_filter_examples.py \
        --harmeme_manifest  /scratch/hmr_data/harmeme/manifest.csv \
        --mami_manifest     /scratch/hmr_data/mami/manifest.csv \
        --mmhs150k_manifest /scratch/hmr_data/mmhs150k/manifest.csv \
        --output_dir        /scratch/hmr_data/filtering_results \
        --n_examples        50

echo "Job submitted."
echo "Follow logs with:"
echo "  runai logs hmr-sample-filter-examples -p course-ee-559-${USERNAME} --follow"
echo ""
echo "Once complete, copy results to your laptop with:"
echo "  scp -r ${USERNAME}@jumphost.rcp.epfl.ch:/scratch/hmr_data/filtering_results/ ."
