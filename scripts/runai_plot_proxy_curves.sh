#!/usr/bin/env bash
set -e

# =============================================================================
# Plot ExplanationProxy training curves (CPU-only, fast)
#
# Reads /scratch/hmr_proxy_checkpoint/training_history.json and eval_results.json
# and saves PNG plots to /scratch/hmr_proxy_training_plots.
#
# Usage:
#   bash scripts/runai_plot_proxy_curves.sh <UID_NUMBER>
#   bash scripts/runai_plot_proxy_curves.sh
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
PROXY_CHECKPOINT_DIR="${PROXY_CHECKPOINT_DIR:-/scratch/hmr_proxy_checkpoint}"
PLOT_DIR="${PLOT_DIR:-/scratch/hmr_proxy_training_plots}"

if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

SCRIPT_PATH="${CODE_ROOT}/analysis/plot_proxy_training.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ -f "${REPO_ROOT_LOCAL}/analysis/plot_proxy_training.py" ]; then
        echo "Note: ${SCRIPT_PATH} is not visible on this node."
        echo "      Using the shared scratch code path inside the RunAI pod: /scratch/hateful_meme_rewriting"
        CODE_ROOT="/scratch/hateful_meme_rewriting"
        SCRIPT_PATH="${CODE_ROOT}/analysis/plot_proxy_training.py"
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        exit 1
    fi
fi

echo "=== Plot Proxy Training Curves ==="
echo "  User:       ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:       ${MODE_LABEL}"
echo "  Code:       ${CODE_ROOT}"
echo "  Proxy ckpt: ${PROXY_CHECKPOINT_DIR}"
echo "  Output:     ${PLOT_DIR}"
echo "  Image:      ${IMAGE}"
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
            --proxy_checkpoint_dir ${PROXY_CHECKPOINT_DIR} \
            --output_dir ${PLOT_DIR}
    "

echo ""
echo "Job submitted. Watch progress with:"
echo "  runai logs hmr-plot-proxy-curves --follow"
echo ""
echo "When complete, plots are under:"
echo "  ${PLOT_DIR}"
