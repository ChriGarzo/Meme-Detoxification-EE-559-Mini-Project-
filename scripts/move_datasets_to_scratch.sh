#!/usr/bin/env bash
set -e

# =============================================================================
# Move uploaded datasets from /home/<user>/datasets_upload/ to /scratch/
#
# Run this INSIDE a RunAI job (scratch is not accessible from the login node).
# Expected upload layout in /home/<user>/datasets_upload/:
#
#   datasets_upload/
#   ├── MAMI/
#   │   ├── training/        ← images + training annotation file (.csv or .xlsx)
#   │   └── test/            ← images only (no labels used)
#   └── MMHS150K/            (or "MMHS150K dataset")
#       ├── img_resized/     ← images named by tweet_id
#       ├── MMHS150K_GT.json
#       └── splits/
#
# Usage: called by runai_move_datasets.sh — do not run directly.
#   DATASET env var controls which dataset to move: mami | mmhs150k | all (default: all)
# =============================================================================

HOME_UPLOAD="/home/${USER}/datasets_upload"
DATASET="${DATASET:-all}"

echo "============================================================"
echo "  Moving datasets from ${HOME_UPLOAD} to /scratch/"
echo "  Dataset: ${DATASET}"
echo "============================================================"

move_mami() {
    echo ""
    echo "--- MAMI ---"
    MAMI_SRC="${HOME_UPLOAD}/MAMI"

    if [ ! -d "${MAMI_SRC}" ]; then
        echo "  [SKIP] ${MAMI_SRC} not found. Upload MAMI first."
        return
    fi

    EXISTING=$(ls /scratch/hmr_data/mami/images/ 2>/dev/null | wc -l)
    if [ "$EXISTING" -gt 0 ]; then
        echo "  [OK] MAMI already in /scratch/ ($EXISTING images). Skipping."
        return
    fi

    echo "  Copying MAMI training images..."
    mkdir -p /scratch/hmr_data/mami/images
    cp -v ${MAMI_SRC}/training/*.jpg /scratch/hmr_data/mami/images/ 2>/dev/null || \
    cp -v ${MAMI_SRC}/training/*.png /scratch/hmr_data/mami/images/ 2>/dev/null || true

    echo "  Copying MAMI annotations..."
    mkdir -p /scratch/hmr_data/mami/annotations
    find ${MAMI_SRC} -maxdepth 2 \( -name "*.csv" -o -name "*.xlsx" -o -name "*.xls" \) \
        -exec cp -v {} /scratch/hmr_data/mami/annotations/ \;

    COUNT=$(ls /scratch/hmr_data/mami/images/ | wc -l)
    echo "  [OK] MAMI copied: ${COUNT} images"
}

move_mmhs150k() {
    echo ""
    echo "--- MMHS150K ---"
    MMHS_SRC=""
    for CANDIDATE in "${HOME_UPLOAD}/MMHS150K" "${HOME_UPLOAD}/MMHS150K dataset" "${HOME_UPLOAD}/MMHS150K_dataset"; do
        if [ -d "${CANDIDATE}" ]; then
            MMHS_SRC="${CANDIDATE}"
            break
        fi
    done

    if [ -z "${MMHS_SRC}" ]; then
        echo "  [SKIP] MMHS150K folder not found in ${HOME_UPLOAD}. Upload MMHS150K first."
        return
    fi

    EXISTING=$(ls /scratch/hmr_data/mmhs150k/images/ 2>/dev/null | wc -l)
    if [ "$EXISTING" -gt 0 ]; then
        echo "  [OK] MMHS150K already in /scratch/ ($EXISTING images). Skipping."
        return
    fi

    echo "  Copying MMHS150K images (this may take a while — ~150K files)..."
    mkdir -p /scratch/hmr_data/mmhs150k/images
    cp -r ${MMHS_SRC}/img_resized/. /scratch/hmr_data/mmhs150k/images/

    echo "  Copying MMHS150K annotations..."
    mkdir -p /scratch/hmr_data/mmhs150k/annotations
    cp -v ${MMHS_SRC}/MMHS150K_GT.json /scratch/hmr_data/mmhs150k/annotations/ 2>/dev/null || true
    if [ -d "${MMHS_SRC}/splits" ]; then
        cp -r ${MMHS_SRC}/splits/. /scratch/hmr_data/mmhs150k/annotations/splits/
    fi

    COUNT=$(ls /scratch/hmr_data/mmhs150k/images/ | wc -l)
    echo "  [OK] MMHS150K copied: ${COUNT} images"
}

# --- Run selected dataset(s) ---
case "${DATASET}" in
    mami)      move_mami ;;
    mmhs150k)  move_mmhs150k ;;
    all)       move_mami; move_mmhs150k ;;
    *)
        echo "ERROR: Unknown dataset '${DATASET}'. Use: mami | mmhs150k | all"
        exit 1
        ;;
esac

# --- Summary ---
echo ""
echo "============================================================"
echo "  SUMMARY"
echo "============================================================"
HARM_COUNT=$(ls /scratch/hmr_data/harmeme/images/  2>/dev/null | wc -l)
MAMI_COUNT=$(ls /scratch/hmr_data/mami/images/     2>/dev/null | wc -l)
MMHS_COUNT=$(ls /scratch/hmr_data/mmhs150k/images/ 2>/dev/null | wc -l)
echo "  HarMeme:  ${HARM_COUNT} images"
echo "  MAMI:     ${MAMI_COUNT} images"
echo "  MMHS150K: ${MMHS_COUNT} images"
echo "============================================================"
