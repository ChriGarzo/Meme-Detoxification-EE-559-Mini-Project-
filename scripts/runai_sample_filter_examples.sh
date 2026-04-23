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
# Usage:
#   bash scripts/runai_sample_filter_examples.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_sample_filter_examples.sh                # scratch code path
#
# Example:
#   bash scripts/runai_sample_filter_examples.sh 123456
# =============================================================================

if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER>   # use /home/\${USER}/hateful_meme_rewriting"
    echo "  bash $0                # use /scratch/hateful_meme_rewriting"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

# --- Path/UID mode selection ---
if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

SCRIPT_PATH="${CODE_ROOT}/data/preprocess/sample_filter_examples.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/data/preprocess/sample_filter_examples.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Sample Filter Examples ==="
echo "  User:   ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:   ${MODE_LABEL}"
echo "  Code:   ${CODE_ROOT}"
echo "  Output: /scratch/hmr_data/filtering_results/"
echo ""

runai submit hmr-sample-filter-examples \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- python3 ${SCRIPT_PATH} \
        --harmeme_manifest  /scratch/hmr_data/harmeme/manifest.csv \
        --mami_manifest     /scratch/hmr_data/mami/manifest.csv \
        --mmhs150k_manifest /scratch/hmr_data/mmhs150k/manifest.csv \
        --output_dir        /home/${USERNAME}/filtering_results \
        --n_examples        50

echo "Job submitted."
echo "Follow logs with:"
echo "  runai logs hmr-sample-filter-examples -p course-ee-559-${USERNAME} --follow"
echo ""
echo "Once complete, copy results to your laptop with:"
echo "  scp -r ${USERNAME}@jumphost.rcp.epfl.ch:~/filtering_results/ ."
