#!/usr/bin/env bash
set -e

echo "============================================================"
echo "  Setting up /scratch directory structure"
echo "============================================================"
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
echo "Directory structure created."

echo ""
echo "============================================================"
echo "  Downloading HarMeme (MOMENTA) from GitHub..."
echo "============================================================"
if [ -f /scratch/hmr_data/harmeme/momenta.zip ]; then
    echo "momenta.zip already exists, skipping download."
else
    python3 -c "
import urllib.request, sys
url = 'https://github.com/LCS2-IIITD/MOMENTA/archive/refs/heads/main.zip'
dest = '/scratch/hmr_data/harmeme/momenta.zip'
print('Downloading from', url)
def progress(count, block_size, total_size):
    pct = count * block_size * 100 // total_size
    sys.stdout.write(f'\r  {pct}%  ({count * block_size // 1024 // 1024} MB)')
    sys.stdout.flush()
urllib.request.urlretrieve(url, dest, reporthook=progress)
print('\nDownload complete.')
"
fi

echo "Extracting..."
python3 -c "
import zipfile, os
z = zipfile.ZipFile('/scratch/hmr_data/harmeme/momenta.zip')
z.extractall('/scratch/hmr_data/harmeme/')
print('Extraction complete.')
print('Contents:', os.listdir('/scratch/hmr_data/harmeme/'))
"

echo ""
echo "============================================================"
echo "  HarMeme extracted. Folder structure:"
echo "============================================================"
find /scratch/hmr_data/harmeme/MOMENTA-main -maxdepth 3 -type d

echo ""
echo "============================================================"
echo "  DONE. /scratch is ready."
echo ""
echo "  MAMI: request at https://forms.gle/AGWMiGicBHiQx4q98"
echo "    then place images in /scratch/hmr_data/mami/images/"
echo "  MMHS150K: download from https://gombru.github.io/2019/10/09/MMHS/"
echo "    then place images in /scratch/hmr_data/mmhs150k/images/"
echo "============================================================"
