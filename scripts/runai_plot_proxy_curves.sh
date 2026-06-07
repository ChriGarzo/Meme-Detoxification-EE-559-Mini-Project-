#!/usr/bin/env bash
set -e

# =============================================================================
# Plot Stage 3 proxy network training curves (CPU-only, fast)
#
# Reads training_history.json from multiseed proxy checkpoint directories and
# saves 4 PNG plots to /scratch/plots/stage_3_training_plots:
#   stage3_train_loss.png          — per-seed train MSE + mean±std band
#   stage3_val_loss.png            — per-seed val MSE + mean±std band
#   stage3_train_vs_val.png        — mean train vs mean val on same axes
#   stage3_generalization_gap.png  — mean(val−train) with ±std band
#
# Usage:
#   bash scripts/runai_plot_proxy_curves.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_plot_proxy_curves.sh                # scratch code path
#
# Optional:
#   CHECKPOINT_SUFFIX=_explicit_detox bash scripts/runai_plot_proxy_curves.sh <UID>
# =============================================================================

if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage:"
    echo "  bash $0 <UID_NUMBER>"
    echo "  bash $0"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-_explicit_detox}"
STAGES_ROOT="${STAGES_ROOT:-/scratch/stages}"
PLOT_DIR="/scratch/hateful_meme_rewriting/training_plots/stage_3_training_plots"

if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

SCRIPT_PATH="${CODE_ROOT}/analysis/recover_training_metrics.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ -f "${REPO_ROOT_LOCAL}/analysis/recover_training_metrics.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Plot Stage 3 Proxy Training Curves ==="
echo "  User:        ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:        ${MODE_LABEL}"
echo "  Code:        ${CODE_ROOT}"
echo "  Stages root: ${STAGES_ROOT}"
echo "  Ckpt suffix: ${CHECKPOINT_SUFFIX}"
echo "  Output:      ${PLOT_DIR}"
echo "  Image:       ${IMAGE}"
echo ""

runai submit hmr-plot-proxy-curves \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools cpu \
    --cpu 2 \
    --memory 8Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- bash -c "
        mkdir -p /tmp/mplconfig-${UID_NUM}
        export MPLCONFIGDIR=/tmp/mplconfig-${UID_NUM}
        pip install matplotlib --quiet --break-system-packages 2>/dev/null || true
        python3 ${SCRIPT_PATH} \
            --stage stage3 \
            --scratch_root ${STAGES_ROOT} \
            --checkpoint_suffix=\"${CHECKPOINT_SUFFIX}\" \
            --output_dir   ${PLOT_DIR}
    "

echo ""
echo "Job submitted. Watch progress with:"
echo "  runai logs hmr-plot-proxy-curves --follow"
echo ""
echo "When complete, plots are under:"
echo "  ${PLOT_DIR}"
