#!/usr/bin/env bash
set -e

# =============================================================================
# Single-inference benchmark: wall-clock time and CO2 per rewrite for
# LLaVA-Next 7B, DetoxLLM 7B, and BART-finetuned 400M.
#
# Usage:
#   bash scripts/runai_benchmark_inference.sh <UID_NUMBER>   # home code path
#   bash scripts/runai_benchmark_inference.sh                # scratch code path
#
# Requires an A100-40G GPU (both LLaVA and DetoxLLM are 7B models in fp16).
# Models are loaded and released one at a time; peak VRAM stays within 40 GB.
#
# What this runs (for each model in sequence):
#   1. Load model
#   2. Run N_WARMUP warmup inferences (GPU warm-up, not measured)
#   3. Run N_BENCH timed inferences under a CodeCarbon tracker
#   4. Report mean/std latency and estimated CO2 per inference
#
# Prerequisites:
#   - Fine-tuned BART checkpoint exists at /scratch/hmr_stage2_full_checkpoint
#   - HuggingFace models cached at /scratch/hf_cache
#     (LLaVA and DetoxLLM are downloaded automatically on first run)
#
# Outputs:
#   /scratch/hmr_inference_benchmark/inference_benchmark.json
#   /scratch/hmr_inference_benchmark/benchmark.log
#
# Optional env vars:
#   SKIP_LLAVA=1      skip LLaVA benchmark (saves ~20 min)
#   SKIP_DETOXLLM=1   skip DetoxLLM benchmark
#   N_WARMUP=N        warmup passes (default 3)
#   N_BENCH=N         timed passes  (default 10)
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

SKIP_LLAVA="${SKIP_LLAVA:-0}"
SKIP_DETOXLLM="${SKIP_DETOXLLM:-0}"
N_WARMUP="${N_WARMUP:-3}"
N_BENCH="${N_BENCH:-10}"

if [ -n "$1" ]; then
    UID_NUM="$1"
    CODE_ROOT="/home/${USERNAME}/hateful_meme_rewriting"
    MODE_LABEL="home"
else
    UID_NUM="$(id -u)"
    CODE_ROOT="/scratch/hateful_meme_rewriting"
    MODE_LABEL="scratch"
fi

SCRIPT="${CODE_ROOT}/analysis/benchmark_single_inference.py"
if [ ! -f "${SCRIPT}" ]; then
    if [ "${MODE_LABEL}" = "scratch" ] && [ -f "${REPO_ROOT_LOCAL}/analysis/benchmark_single_inference.py" ]; then
        echo "Note: /scratch path not visible on this node; local check passed."
    else
        echo "ERROR: Script not found at ${SCRIPT}"
        exit 1
    fi
fi

SKIP_FLAGS=""
[ "${SKIP_LLAVA}" = "1" ] && SKIP_FLAGS="${SKIP_FLAGS} --skip_llava"
[ "${SKIP_DETOXLLM}" = "1" ] && SKIP_FLAGS="${SKIP_FLAGS} --skip_detoxllm"

echo "=== Single-Inference Benchmark ==="
echo "  User:      ${USERNAME} (UID: ${UID_NUM})"
echo "  Mode:      ${MODE_LABEL}"
echo "  Code:      ${CODE_ROOT}"
echo "  n_warmup:  ${N_WARMUP}"
echo "  n_bench:   ${N_BENCH}"
echo "  Skip LLaVA:    ${SKIP_LLAVA}"
echo "  Skip DetoxLLM: ${SKIP_DETOXLLM}"
echo ""

runai submit hmr-benchmark-inference \
    --run-as-uid ${UID_NUM} \
    --image ${IMAGE} \
    --node-pools a100-40g \
    --gpu 1 \
    --cpu 8 \
    --memory 40Gi \
    --existing-pvc claimname=home,path=/home/${USERNAME} \
    --existing-pvc claimname=course-ee-559-scratch-g${GROUP_NUM},path=/scratch \
    --existing-pvc claimname=course-ee-559-shared-ro,path=/shared-ro \
    --existing-pvc claimname=course-ee-559-shared-rw,path=/shared-rw \
    --command -- bash -c "
        cd ${CODE_ROOT} &&
        python3 analysis/benchmark_single_inference.py \
            --validation_jsonl /scratch/hmr_stage2_dataset/val.jsonl \
            --checkpoint_dir   /scratch/hmr_stage2_full_checkpoint \
            --hf_cache         /scratch/hf_cache \
            --output_dir       /scratch/hmr_inference_benchmark \
            --n_warmup ${N_WARMUP} \
            --n_bench  ${N_BENCH} \
            ${SKIP_FLAGS}
    "

echo "Benchmark job submitted."
echo "Watch logs with:"
echo "  runai logs hmr-benchmark-inference --follow"
echo "When complete, results at:"
echo "  /scratch/hmr_inference_benchmark/inference_benchmark.json"
echo ""
echo "To skip LLaVA (fastest option if 7B download is slow):"
echo "  SKIP_LLAVA=1 bash $0"
