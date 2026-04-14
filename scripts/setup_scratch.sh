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
echo "  Downloading HarMeme (di-dimitrov/mmf) from GitHub..."
echo "============================================================"
if [ -f /scratch/hmr_data/harmeme/mmf.zip ]; then
    echo "mmf.zip already exists, skipping download."
else
    python3 -c "
import urllib.request, sys
url = 'https://github.com/di-dimitrov/mmf/archive/refs/heads/master.zip'
dest = '/scratch/hmr_data/harmeme/mmf.zip'
print('Downloading from', url)
def progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        mb = count * block_size // 1024 // 1024
        sys.stdout.write(f'\r  {pct}%  ({mb} MB)')
        sys.stdout.flush()
urllib.request.urlretrieve(url, dest, reporthook=progress)
print('\nDownload complete.')
"
fi

echo "Extracting..."
python3 -c "
import zipfile, os, shutil

z = zipfile.ZipFile('/scratch/hmr_data/harmeme/mmf.zip')
z.extractall('/scratch/hmr_data/harmeme/')
print('Extraction complete.')

# Move images to the expected path
src_images = '/scratch/hmr_data/harmeme/mmf-master/data/datasets/memes/defaults/images'
dst_images = '/scratch/hmr_data/harmeme/images'
if os.path.isdir(src_images):
    if not os.path.isdir(dst_images) or not os.listdir(dst_images):
        shutil.copytree(src_images, dst_images, dirs_exist_ok=True)
        print(f'Images copied to {dst_images}')
    else:
        print(f'Images already at {dst_images}, skipping copy.')
    print(f'Image count: {len(os.listdir(dst_images))}')
else:
    print(f'WARNING: images folder not found at {src_images}')
    print('Contents of mmf-master/data/datasets/memes/defaults/:')
    base = '/scratch/hmr_data/harmeme/mmf-master/data/datasets/memes/defaults'
    if os.path.isdir(base):
        for f in os.listdir(base):
            print(' ', f)

# Copy annotations too
src_ann = '/scratch/hmr_data/harmeme/mmf-master/data/datasets/memes/defaults/annotations'
dst_ann = '/scratch/hmr_data/harmeme/annotations'
if os.path.isdir(src_ann):
    shutil.copytree(src_ann, dst_ann, dirs_exist_ok=True)
    print(f'Annotations copied to {dst_ann}')
    print(f'Annotation files: {os.listdir(dst_ann)}')
"

echo ""
echo "============================================================"
echo "  HarMeme setup complete. Final structure:"
echo "============================================================"
echo "Images:      $(ls /scratch/hmr_data/harmeme/images/ 2>/dev/null | wc -l) files"
echo "Annotations: $(ls /scratch/hmr_data/harmeme/annotations/ 2>/dev/null)"

echo ""
echo "============================================================"
echo "  DONE. /scratch is ready."
echo ""
echo "  MAMI: request at https://forms.gle/AGWMiGicBHiQx4q98"
echo "    then place images in /scratch/hmr_data/mami/images/"
echo "  MMHS150K: download from https://gombru.github.io/2019/10/09/MMHS/"
echo "    then place images in /scratch/hmr_data/mmhs150k/images/"
echo "============================================================"
