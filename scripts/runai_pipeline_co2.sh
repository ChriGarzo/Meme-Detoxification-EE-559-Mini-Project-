#!/usr/bin/env bash
set -e

# =============================================================================
# Pipeline CO2 aggregation — reads existing CodeCarbon CSVs and training
# history files to produce a full pipeline emission summary.
#
# Usage:
#   bash scripts/runai_pipeline_co2.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_pipeline_co2.sh                # scratch code path
#
# No GPU required: the script only reads files, it does not run any model.
#
# Prerequisites:
#   - All pipeline stages have completed at least once.
#   - Emissions CSVs exist under /scratch/hmr_stage1_output/ and
#     /scratch/hmr_eval_*/ (produced automatically during inference).
#   - /scratch/hmr_stage2_*/training_history.json exists
#     (produced by train_stage2.py).
#
# Outputs:
#   /scratch/hmr_co2_summary/pipeline_co2_summary.json
#   /scratch/hmr_co2_summary/pipeline_co2_summary.tsv
# =============================================================================

if [ "$#" -gt 1 ]; then
    echo "ERROR: Too many arguments."
    echo "Usage: bash $0 [<UID_NUMBER>]"
    exit 1
fi

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

SCRIPT="${CODE_ROOT}/analysis/aggregate_pipeline_co2.py"
if [ ! -f "${SCRIPT}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/analysis/aggregate_pipeline_co2.py" ]; then
        echo "Note: /scratch path not visible on this node; local check passed."
    else
        echo "ERROR: Script not found at ${SCRIPT}"
        exit 1
    fi
fi

echo "=== Pipeline CO2 Aggregation ==="
echo "  User:  ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:  ${MODE_LABEL}"
echo "  Code:  ${CODE_ROOT}"
echo ""

runai submit hmr-pipeline-co2 \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools cpu \
    --cpu 4 \
    --memory 8Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --command -- bash -c "
        cd ${CODE_ROOT} &&
        python3 analysis/aggregate_pipeline_co2.py \
            --scratch_dir /scratch \
            --output_dir  /scratch/hmr_co2_summary
    "

echo "Pipeline CO2 job submitted."
echo "Watch logs with:"
echo "  runai logs hmr-pipeline-co2 --follow"
echo "When complete, results at:"
echo "  /scratch/hmr_co2_summary/pipeline_co2_summary.tsv"
