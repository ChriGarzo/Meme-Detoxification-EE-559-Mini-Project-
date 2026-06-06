#!/usr/bin/env bash
set -e

# =============================================================================
# Screencast demo: CLIP Proxy + BART-large FT Full inference on 20 test images
#
# Runs the finetuned proxy+BART pipeline on 20 randomly sampled test examples,
# printing clean per-example logs with STA, ΔToxicity, SIM, and CLIP Score.
#
# Usage:
#   bash scripts/runai_screencast_demo.sh
#
# Optional environment variables:
#   N_SAMPLES       number of test images to sample (default: 20)
#   SEED            random seed for sampling (default: 42)
#   EVAL_SUFFIX     output directory suffix   (default: "")
# =============================================================================

USERNAME="${USER}"
GROUP_NUM="31"
IMAGE="registry.rcp.epfl.ch/ee-559-garzone/hmr:v0.1"
REPO_ROOT_LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

SEED="${SEED:-42}"
EVAL_SUFFIX="${EVAL_SUFFIX:-}"

CODE_ROOT="/scratch/hateful_meme_rewriting"
HF_CACHE="/scratch/hf_cache"
STAGES_ROOT="/scratch/stages"
OUTPUT_DIR="/scratch/eval_results/screencast_demo${EVAL_SUFFIX}"

INPUT_JSONL="${STAGES_ROOT}/hmr_stage2_dataset/test.jsonl"
BART_CHECKPOINT="${STAGES_ROOT}/hmr_stage2_phase2_full_explicit_detox_checkpoint"
PROXY_CHECKPOINT="${STAGES_ROOT}/hmr_proxy_checkpoint_explicit_detox/best_proxy.pt"

JOB_NAME="hmr-screencast-demo"

echo "=== Screencast Demo Job ==="
echo "  User             : ${USERNAME}"
echo "  Code             : ${CODE_ROOT}"
echo "  Test JSONL       : ${INPUT_JSONL}"
echo "  BART checkpoint  : ${BART_CHECKPOINT}"
echo "  Proxy checkpoint : ${PROXY_CHECKPOINT}"
echo "  HF cache         : ${HF_CACHE}"
echo "  Output           : ${OUTPUT_DIR}"
echo "  Seed             : ${SEED}"
echo "  Image            : ${IMAGE}"
echo ""

runai submit "${JOB_NAME}" \
    --run-as-uid "$(id -u)" \
    --image "${IMAGE}" \
    --node-pools a100-40g \
    --gpu 1 \
    --cpu 8 \
    --memory 40Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- python3 "${CODE_ROOT}/inference/run_screencast_demo.py" \
        --input_jsonl    "${INPUT_JSONL}" \
        --bart_checkpoint  "${BART_CHECKPOINT}" \
        --proxy_checkpoint "${PROXY_CHECKPOINT}" \
        --hf_cache       "${HF_CACHE}" \
        --output_dir     "${OUTPUT_DIR}" \
        --seed           "${SEED}"

echo ""
echo "Job submitted: ${JOB_NAME}"
echo ""
echo "Follow logs with:"
echo "  runai logs ${JOB_NAME} -p course-ee-559-${USERNAME} --follow"
echo ""
echo "Results will be saved to:"
echo "  ${OUTPUT_DIR}/screencast_results.jsonl"
