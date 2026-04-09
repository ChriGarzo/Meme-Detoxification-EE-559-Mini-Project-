#!/usr/bin/env bash
set -e

echo "============================================"
echo "  Dataset Download Script"
echo "  Hateful Meme Rewriting Pipeline"
echo "============================================"
echo ""

# Parse output directory
OUTPUT_DIR="${1:-.}"

echo "WARNING: By downloading these datasets, you agree to use them"
echo "only for research purposes and in accordance with their respective"
echo "licenses and terms of use."
echo ""
read -p "Do you want to proceed? (y/n): " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "=== Dataset 1: HarMeme (MOMENTA) ==="
echo "Cloning from official GitHub repository..."
HARMEME_DIR="${OUTPUT_DIR}/harmeme"
if [ ! -d "${HARMEME_DIR}" ]; then
    git clone https://github.com/LCS2-IIITD/MOMENTA.git "${HARMEME_DIR}"
    echo "HarMeme downloaded to ${HARMEME_DIR}"
else
    echo "HarMeme directory already exists, skipping."
fi

echo ""
echo "=== Dataset 2: MAMI ==="
echo "MAMI dataset must be manually requested."
echo "Please fill the form at: https://forms.gle/AGWMiGicBHiQx4q98"
echo "Once approved, place the dataset at your chosen --mami_dir path."

echo ""
echo "=== Dataset 3: MMHS150K ==="
echo "Please download MMHS150K from: https://gombru.github.io/2019/10/09/MMHS/"
echo "Do NOT attempt to re-scrape from Twitter."
echo "Place the downloaded data in: ${OUTPUT_DIR}/mmhs150k/"

echo ""
echo "=== Dataset 4: ParaDetox ==="
echo "ParaDetox will be auto-downloaded from HuggingFace during training."
echo "No manual action needed."

echo ""
echo "Done! Please ensure all datasets are properly placed before running the pipeline."
