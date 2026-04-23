#!/usr/bin/env bash
set -e

# =============================================================================
# Plot training curves from Stage 2 checkpoints (CPU-only, fast)
#
# Reads trainer_state.json / training_history.json from all Stage 2 checkpoint
# directories on scratch and saves PNG plots to /scratch/hmr_training_plots.
#
# Run this AFTER any Stage 2 training completes.  Works even on runs that
# predate training_history.json — it falls back to trainer_state.json inside
# the HuggingFace checkpoint subdirs.
#
# Usage:
#   bash scripts/runai_plot_curves.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_plot_curves.sh                # scratch code path
#
# Example:
#   bash scripts/runai_plot_curves.sh 123456
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

SCRIPT_PATH="${CODE_ROOT}/analysis/recover_training_metrics.py"
if [ ! -f "${SCRIPT_PATH}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/analysis/recover_training_metrics.py" ]; then
        echo "Note: /scratch path not visible on this node; using local repo check at ${REPO_ROOT_LOCAL}."
    else
        echo "ERROR: Script not found at: ${SCRIPT_PATH}"
        echo "Check that the repository exists at ${CODE_ROOT}."
        exit 1
    fi
fi

echo "=== Plot Training Curves ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo "  Image: ${IMAGE}"
echo "  Output: /scratch/hmr_training_plots"
echo ""

runai submit hmr-plot-curves \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools cpu \
    --cpu 2 \
    --memory 8Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g31,path=/scratch \
    --command -- bash -c "
        pip install matplotlib --quiet --break-system-packages 2>/dev/null || true
        python3 ${SCRIPT_PATH} \
            --scratch_root /scratch \
            --output_dir   /scratch/hmr_training_plots
    "

echo ""
echo "Job submitted.  Watch progress with:"
echo "  runai logs hmr-plot-curves --follow"
echo ""
echo "Once done, copy plots to your local machine with:"
echo "  # From a NEW terminal on your laptop (not inside the cluster):"
echo "  kubectl cp <POD_NAME>:/scratch/hmr_training_plots ./hmr_training_plots"
echo ""
echo "Or submit a second job to list the generated files:"
echo "  runai submit hmr-list-plots \\"
echo "      --run-as-uid ${UID_NUM} \\"
echo "      --image ${IMAGE} \\"
echo "      --cpu 1 --memory 1Gi \\"
echo "      --existing-pvc claimname=course-ee-559-scratch-g31,path=/scratch \\"
echo "      --command -- bash -c 'ls -lh /scratch/hmr_training_plots/'"
