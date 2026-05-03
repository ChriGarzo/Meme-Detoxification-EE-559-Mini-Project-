#!/usr/bin/env bash
set -euo pipefail

# Runs inside the RunAI evaluation pod.
# Keep this as a separate script so runai_evaluate.sh does not depend on a
# long nested bash -c string, which is brittle and hard to debug.

CODE_ROOT="${1:-/scratch/hateful_meme_rewriting}"
HF_CACHE="${HF_CACHE:-/scratch/hf_cache}"
STAGE1_DIR="${STAGE1_DIR:-/scratch/hmr_stage1_output}"
VALIDATION_JSONL="${VALIDATION_JSONL:-/scratch/hmr_stage2_dataset/val.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch}"

cd "${CODE_ROOT}"

echo "=== Stage 3 evaluation job ==="
echo "Code root: ${CODE_ROOT}"
echo "HF cache:  ${HF_CACHE}"
echo "Stage 1:   ${STAGE1_DIR}"
echo "Val JSONL: ${VALIDATION_JSONL}"
echo "Output:    ${OUTPUT_ROOT}/hmr_eval_results"
echo ""

VAL_COUNT="$(wc -l < "${VALIDATION_JSONL}")"
echo "Validation examples: ${VAL_COUNT}"
echo ""

run_stage2_if_needed() {
    local cond="$1"
    local checkpoint_dir="$2"
    local output_dir="$3"
    local output_file="${output_dir}/stage2_rewrites_${cond}.jsonl"

    if [ -f "${output_file}" ]; then
        local output_count
        output_count="$(wc -l < "${output_file}")"
        if [ "${output_count}" = "${VAL_COUNT}" ]; then
            echo "--- ${cond}: found ${output_file} (${output_count}/${VAL_COUNT}); skipping inference ---"
            return 0
        fi
        echo "--- ${cond}: found incomplete ${output_file} (${output_count}/${VAL_COUNT}); recomputing ---"
    else
        echo "--- ${cond}: ${output_file} not found; running inference ---"
    fi

    python3 "${CODE_ROOT}/inference/run_stage2.py" \
        --condition "${cond}" \
        --checkpoint_dir "${checkpoint_dir}" \
        --input_jsonl "${VALIDATION_JSONL}" \
        --output_dir "${output_dir}" \
        --hf_cache "${HF_CACHE}" \
        --batch_size 4
}

echo "=== Non-finetuned BART inference: all 4 conditions ==="
for COND in full target_only visual_only none; do
    echo "--- Base BART condition: ${COND} ---"
    run_stage2_if_needed \
        "${COND}" \
        "facebook/bart-large" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_${COND}"
done

echo "=== Finetuned BART inference: all 4 conditions ==="
for COND in full target_only visual_only none; do
    echo "--- Finetuned BART condition: ${COND} ---"
    run_stage2_if_needed \
        "${COND}" \
        "${OUTPUT_ROOT}/hmr_stage2_phase2_${COND}_checkpoint" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_${COND}"
done

echo "=== Final evaluation: LLaVA teacher vs base BART vs finetuned BART ==="
python3 "${CODE_ROOT}/evaluation/evaluate.py" \
    --validation_jsonl "${VALIDATION_JSONL}" \
    --bart_base_output_dirs \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_full" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_target_only" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_visual_only" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_none" \
    --bart_finetuned_output_dirs \
        "${OUTPUT_ROOT}/hmr_eval_stage2_full" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_target_only" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_visual_only" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_none" \
    --output_dir "${OUTPUT_ROOT}/hmr_eval_results" \
    --hf_cache "${HF_CACHE}"

echo "=== Stage 3 evaluation complete ==="
