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
INPUT_FORMAT="${INPUT_FORMAT:-legacy}"
TASK_PREFIX="${TASK_PREFIX:-}"
CHECKPOINT_SUFFIX="${CHECKPOINT_SUFFIX:-}"
EVAL_SUFFIX="${EVAL_SUFFIX:-}"
INCLUDE_DETOXLLM="${INCLUDE_DETOXLLM:-0}"
TASK_PREFIX_ARGS=()
if [ -n "${TASK_PREFIX}" ]; then
    TASK_PREFIX_ARGS=(--task_prefix "${TASK_PREFIX}")
fi

cd "${CODE_ROOT}"

echo "=== Stage 3 evaluation job ==="
echo "Code root:       ${CODE_ROOT}"
echo "HF cache:        ${HF_CACHE}"
echo "Stage 1:         ${STAGE1_DIR}"
echo "Val JSONL:       ${VALIDATION_JSONL}"
echo "Input fmt:       ${INPUT_FORMAT}"
echo "Eval suff:       ${EVAL_SUFFIX}"
echo "Ckpt suff:       ${CHECKPOINT_SUFFIX}"
echo "Include DetoxLLM: ${INCLUDE_DETOXLLM}"
echo "Output:          ${OUTPUT_ROOT}/hmr_eval_results${EVAL_SUFFIX}"
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
        --input_format "${INPUT_FORMAT}" \
        "${TASK_PREFIX_ARGS[@]}" \
        --batch_size 4
}

echo "=== Non-finetuned BART inference: all 4 conditions ==="
for COND in full target_only visual_only none; do
    echo "--- Base BART condition: ${COND} ---"
    run_stage2_if_needed \
        "${COND}" \
        "facebook/bart-large" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_${COND}${EVAL_SUFFIX}"
done

echo "=== Finetuned BART inference: all 4 conditions ==="
for COND in full target_only visual_only none; do
    echo "--- Finetuned BART condition: ${COND} ---"
    run_stage2_if_needed \
        "${COND}" \
        "${OUTPUT_ROOT}/hmr_stage2_phase2_${COND}${CHECKPOINT_SUFFIX}_checkpoint" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_${COND}${EVAL_SUFFIX}"
done

DETOXLLM_OUTPUT_DIR="${OUTPUT_ROOT}/hmr_eval_detoxllm${EVAL_SUFFIX}"
DETOXLLM_OUTPUT_FILE="${DETOXLLM_OUTPUT_DIR}/detoxllm_rewrites.jsonl"

if [ "${INCLUDE_DETOXLLM}" = "1" ]; then
    if [ -f "${DETOXLLM_OUTPUT_FILE}" ]; then
        DETOX_COUNT="$(wc -l < "${DETOXLLM_OUTPUT_FILE}")"
        if [ "${DETOX_COUNT}" = "${VAL_COUNT}" ]; then
            echo "--- DetoxLLM: found ${DETOXLLM_OUTPUT_FILE} (${DETOX_COUNT}/${VAL_COUNT}); skipping inference ---"
        else
            echo "--- DetoxLLM: found incomplete output (${DETOX_COUNT}/${VAL_COUNT}); recomputing ---"
            python3 "${CODE_ROOT}/baselines/run_detoxllm_baseline.py" \
                --validation_jsonl "${VALIDATION_JSONL}" \
                --output_dir "${DETOXLLM_OUTPUT_DIR}" \
                --hf_cache "${HF_CACHE}"
        fi
    else
        echo "=== DetoxLLM inference ==="
        python3 "${CODE_ROOT}/baselines/run_detoxllm_baseline.py" \
            --validation_jsonl "${VALIDATION_JSONL}" \
            --output_dir "${DETOXLLM_OUTPUT_DIR}" \
            --hf_cache "${HF_CACHE}"
    fi
fi

DETOXLLM_ARGS=()
if [ "${INCLUDE_DETOXLLM}" = "1" ] && [ -f "${DETOXLLM_OUTPUT_FILE}" ]; then
    DETOXLLM_ARGS=(--detoxllm_output_path "${DETOXLLM_OUTPUT_DIR}")
fi

echo "=== Final evaluation: LLaVA teacher vs BART base (ablation) vs finetuned BART vs DetoxLLM ==="
python3 "${CODE_ROOT}/evaluation/evaluate.py" \
    --validation_jsonl "${VALIDATION_JSONL}" \
    --bart_base_output_dirs \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_full${EVAL_SUFFIX}" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_target_only${EVAL_SUFFIX}" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_visual_only${EVAL_SUFFIX}" \
        "${OUTPUT_ROOT}/hmr_eval_bart_base_none${EVAL_SUFFIX}" \
    --bart_finetuned_output_dirs \
        "${OUTPUT_ROOT}/hmr_eval_stage2_full${EVAL_SUFFIX}" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_target_only${EVAL_SUFFIX}" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_visual_only${EVAL_SUFFIX}" \
        "${OUTPUT_ROOT}/hmr_eval_stage2_none${EVAL_SUFFIX}" \
    "${DETOXLLM_ARGS[@]}" \
    --output_dir "${OUTPUT_ROOT}/hmr_eval_results${EVAL_SUFFIX}" \
    --hf_cache "${HF_CACHE}"

echo "=== Stage 3 evaluation complete ==="
