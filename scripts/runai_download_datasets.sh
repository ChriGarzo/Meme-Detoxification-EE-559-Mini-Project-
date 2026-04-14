#!/usr/bin/env bash
set -e

# =============================================================================
# Dataset Download + Scratch Setup
#
# Run this ONCE before starting any pipeline stage.
# Downloads HarMeme automatically. Prints instructions for MAMI and MMHS150K
# which require manual steps (form request / direct download).
#
# Usage: bash scripts/runai_download_datasets.sh <UID_NUMBER>
#
#   UID_NUMBER  Your numeric Unix UID. Get it with: id -u
#
# Example:
#   bash scripts/runai_download_datasets.sh 314305
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
IMAGE="registry.rcp.epfl.ch/ee-559-${USERNAME}/hmr:v0.1"

echo "=== Dataset Download + Scratch Setup ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Image: ${IMAGE}"
echo ""

runai submit hmr-download-datasets \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- bash -c "
        set -e

        echo '============================================================'
        echo '  Setting up /scratch directory structure'
        echo '============================================================'
        mkdir -p /scratch/hf_cache
        mkdir -p /scratch/hmr_data/harmeme/images
        mkdir -p /scratch/hmr_data/mami/images
        mkdir -p /scratch/hmr_data/mmhs150k/images
        mkdir -p /scratch/hmr_stage1_output
        mkdir -p /scratch/hmr_stage2_dataset
        mkdir -p /scratch/hmr_stage2_phase1_checkpoint
        mkdir -p /scratch/hmr_stage2_phase2_full_checkpoint
        mkdir -p /scratch/hmr_stage2_phase2_target_only_checkpoint
        mkdir -p /scratch/hmr_stage2_phase2_attack_only_checkpoint
        mkdir -p /scratch/hmr_stage2_phase2_none_checkpoint
        mkdir -p /scratch/hmr_proxy_checkpoint
        mkdir -p /scratch/hmr_eval_results
        echo 'Directory structure created.'

        echo ''
        echo '============================================================'
        echo '  Downloading HarMeme (MOMENTA) from GitHub...'
        echo '============================================================'
        if [ -f /scratch/hmr_data/harmeme/momenta.zip ]; then
            echo 'momenta.zip already exists, skipping download.'
        else
            wget -q --show-progress \
                -O /scratch/hmr_data/harmeme/momenta.zip \
                https://github.com/LCS2-IIITD/MOMENTA/archive/refs/heads/main.zip
            echo 'Download complete.'
        fi

        echo 'Extracting...'
        python3 -c \"
import zipfile, os
z = zipfile.ZipFile('/scratch/hmr_data/harmeme/momenta.zip')
z.extractall('/scratch/hmr_data/harmeme/')
print('Extraction complete.')
print('Contents:', os.listdir('/scratch/hmr_data/harmeme/'))
\"

        echo ''
        echo '============================================================'
        echo '  HarMeme extracted. Listing top-level structure:'
        echo '============================================================'
        find /scratch/hmr_data/harmeme/MOMENTA-main -maxdepth 3 -type d | head -30

        echo ''
        echo '============================================================'
        echo '  DONE. Scratch structure ready.'
        echo ''
        echo '  NEXT STEPS FOR MAMI AND MMHS150K:'
        echo '  - MAMI: request access at https://forms.gle/AGWMiGicBHiQx4q98'
        echo '    then upload images to /scratch/hmr_data/mami/images/'
        echo '  - MMHS150K: download from https://gombru.github.io/2019/10/09/MMHS/'
        echo '    then upload images to /scratch/hmr_data/mmhs150k/images/'
        echo '============================================================'
    "

echo ""
echo "Download job submitted. Follow logs with:"
echo "  runai logs hmr-download-datasets -p course-ee-559-${USERNAME} --follow"
