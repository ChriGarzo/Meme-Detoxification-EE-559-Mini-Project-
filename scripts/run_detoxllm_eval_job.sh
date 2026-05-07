#!/usr/bin/env bash
set -euo pipefail

# Inner job script for DetoxLLM-only evaluation.
# Unlike run_evaluate_job.sh, this script does NOT re-run BART inference.
# It runs only DetoxLLM inference (with skip-if-complete), then calls
# evaluate.py pointing at the already-existing canonical BART dirs.

CODE_ROOT="${1:-/scratch/hateful_meme_rewriting}"
HF_CACHE="${HF_CACHE:-/scratch/hf_cache}"
VALIDATION_JSONL="${VALIDATION_JSONL:-/scratch/hmr_stage2_dataset/val.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch}"

# DetoxLLM output dir is always fixed — no suffix variant needed.
DETOXLLM_OUTPUT_DIR="${OUTPUT_ROOT}/hmr_eval_detoxllm"
DETOXLLM_OUTPUT_FILE="${DETOXLLM_OUTPUT_DIR}/detoxllm_rewrites.jsonl"

# Canonical BART inference dirs (already produced by runai_evaluate.sh).
BART_BASE_DIRS=(
    "${OUTPUT_ROOT}/hmr_eval_bart_base_full"
    "${OUTPUT_ROOT}/hmr_eval_bart_base_target_only"
    "${OUTPUT_ROOT}/hmr_eval_bart_base_visual_only"
    "${OUTPUT_ROOT}/hmr_eval_bart_base_none"
)
BART_FINETUNED_DIRS=(
    "${OUTPUT_ROOT}/hmr_eval_stage2_full"
    "${OUTPUT_ROOT}/hmr_eval_stage2_target_only"
    "${OUTPUT_ROOT}/hmr_eval_stage2_visual_only"
    "${OUTPUT_ROOT}/hmr_eval_stage2_none"
)

EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/hmr_eval_results_detoxllm"

cd "${CODE_ROOT}"

echo "=== DetoxLLM evaluation job ==="
echo "Code root:    ${CODE_ROOT}"
echo "HF cache:     ${HF_CACHE}"
echo "Val JSONL:    ${VALIDATION_JSONL}"
echo "DetoxLLM out: ${DETOXLLM_OUTPUT_DIR}"
echo "Eval out:     ${EVAL_OUTPUT_DIR}"
echo ""
echo "BART dirs (existing, not re-run):"
for d in "${BART_BASE_DIRS[@]}" "${BART_FINETUNED_DIRS[@]}"; do
    echo "  $d"
done
echo ""

VAL_COUNT="$(wc -l < "${VALIDATION_JSONL}")"
echo "Validation examples: ${VAL_COUNT}"
echo ""

# ------------------------------------------------------------------
# Step 1: DetoxLLM inference (skip if already complete)
# ------------------------------------------------------------------
if [ -f "${DETOXLLM_OUTPUT_FILE}" ]; then
    DETOX_COUNT="$(wc -l < "${DETOXLLM_OUTPUT_FILE}")"
    if [ "${DETOX_COUNT}" = "${VAL_COUNT}" ]; then
        echo "DetoxLLM: output already complete (${DETOX_COUNT}/${VAL_COUNT}) — skipping inference."
    else
        echo "DetoxLLM: incomplete output (${DETOX_COUNT}/${VAL_COUNT}) — re-running inference."
        python3 "${CODE_ROOT}/baselines/run_detoxllm_baseline.py" \
            --validation_jsonl "${VALIDATION_JSONL}" \
            --output_dir "${DETOXLLM_OUTPUT_DIR}" \
            --hf_cache "${HF_CACHE}"
    fi
else
    echo "=== DetoxLLM inference (358 examples, one at a time) ==="
    python3 "${CODE_ROOT}/baselines/run_detoxllm_baseline.py" \
        --validation_jsonl "${VALIDATION_JSONL}" \
        --output_dir "${DETOXLLM_OUTPUT_DIR}" \
        --hf_cache "${HF_CACHE}"
fi

echo ""
echo "=== Final evaluation: LLaVA teacher vs BART base (ablation) vs finetuned BART vs DetoxLLM ==="
python3 "${CODE_ROOT}/evaluation/evaluate.py" \
    --validation_jsonl "${VALIDATION_JSONL}" \
    --bart_base_output_dirs   "${BART_BASE_DIRS[@]}" \
    --bart_finetuned_output_dirs "${BART_FINETUNED_DIRS[@]}" \
    --detoxllm_output_path "${DETOXLLM_OUTPUT_DIR}" \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --hf_cache "${HF_CACHE}"

echo "=== DetoxLLM evaluation complete ==="
echo "Results at: ${EVAL_OUTPUT_DIR}"
